"""Core audit logic for foldaudit.

Pure Python, no third-party dependencies. It never trains a model:
it only counts, compares rates, and checks orderings over the folds
it is given.
"""

from __future__ import annotations

import inspect
import math
import operator
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction

from .report import Flag, FoldResult, Report


@dataclass(frozen=True)
class Thresholds:
    """Configurable, heuristic thresholds.

    All of these are documented heuristics, not universal truths:
    a 20% class-rate deviation is a warning, not a law of nature.
    Change them to fit your problem; the report always quotes the
    actual numbers alongside the flag so you can judge for yourself.

    class_rate_deviation: warn when a fold's class rate differs from
        the overall rate by more than this (absolute, e.g. 0.20 = 20
        percentage points).
    class_rate_ratio: warn when a fold's class rate is more than this
        many times the overall rate, or less than 1/this of it, even
        when the absolute difference is below class_rate_deviation.
        This is the rare-class check: a class at 2% overall appearing
        at 15% in a fold is a 7.5x over-representation, but only a
        13-point absolute difference, so the absolute check alone
        misses it. Two guards keep this from being noise: the class
        must actually appear in the test fold (a missing class is
        already reported as missing_class_test), and the absolute
        difference must be at least 5 percentage points, so 1% -> 3%
        does not warn. At most one deviation flag is raised per
        class per fold: the absolute flag wins if both would fire.
    min_test_size: warn when a test fold has fewer rows than this.
    fold_size_deviation: warn when a fold's test size differs from
        the median test size by more than this fraction.
    """

    class_rate_deviation: float = 0.20
    class_rate_ratio: float = 2.0
    min_test_size: int = 10
    fold_size_deviation: float = 0.50

    def __post_init__(self):
        # FA-007: NaN/inf and wrong types must not silently disable a
        # check. A bool is not a number here, and a string is not
        # silently coerced. NaN fails the finite check for every
        # field, including min_test_size.
        for field_name, value in (
            ("class_rate_deviation", self.class_rate_deviation),
            ("class_rate_ratio", self.class_rate_ratio),
            ("min_test_size", self.min_test_size),
            ("fold_size_deviation", self.fold_size_deviation),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number, got {value!r}")  # noqa: TRY004
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value!r}")
        if not 0 <= self.class_rate_deviation <= 1:
            raise ValueError(
                "class_rate_deviation must be between 0 and 1, "
                f"got {self.class_rate_deviation}"
            )
        if self.class_rate_ratio < 1.0:
            raise ValueError(
                f"class_rate_ratio must be at least 1.0, got {self.class_rate_ratio}"
            )
        if self.min_test_size < 0:
            raise ValueError(
                f"min_test_size must be at least 0, got {self.min_test_size}"
            )
        if self.fold_size_deviation < 0:
            raise ValueError(
                f"fold_size_deviation must be at least 0, "
                f"got {self.fold_size_deviation}"
            )


def _is_missing(value):
    """True for None, NaN, NaT and pandas NA, without importing them.

    NaN is detected by self-inequality (NaT also satisfies it);
    pandas NA has type name NAType and its comparison returns NA,
    which is why the type-name check comes first.
    """
    if value is None:
        return True
    if type(value).__name__ in ("NaTType", "NAType", "NaT"):
        return True
    try:
        return bool(value != value)  # noqa: PLR0124
    except Exception:  # noqa: BLE001
        return False


def _is_invalid_label(value):
    """A label that cannot be a class: missing, or infinite."""
    if _is_missing(value):
        return True
    try:
        return bool(math.isinf(value))
    except (TypeError, ValueError):
        return False


def _length_of(value):
    """Length via shape[0] first (sparse matrices have no len())."""
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except (TypeError, IndexError, ValueError):
            pass
    if hasattr(value, "__len__"):
        try:
            return len(value)
        except TypeError:
            return None
    return None


def _is_sequence_of_ints(seq):
    return isinstance(seq, (list, tuple)) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in seq
    )


def _is_1d_int_sequence(seq):
    """True for a 1-D sequence of integer indices, list/tuple or not.

    P-006: a numpy integer array (and a stdlib fake with the same
    shape/tolist interface) is a 1-D int sequence, even though it
    is neither a list nor a tuple and its elements are not Python
    ints. A 2-D shape, a string, a mapping, or an element that is
    itself a sequence is not.
    """
    if isinstance(seq, (str, bytes, dict)):
        return False
    if isinstance(seq, (list, tuple)):
        return _is_sequence_of_ints(seq)
    shape = getattr(seq, "shape", None)
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            return False
        if len(shape) != 1:
            return False
    try:
        values = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    except TypeError:
        return False
    for v in values:
        if isinstance(v, (bool, list, tuple, dict, set)):
            return False
        try:
            operator.index(v)
        except TypeError:
            return False
    return True


def _convert_datetime64(values, original):
    """Convert numpy datetime64[ns] ints to ISO strings (FA-014).

    ``ndarray.tolist()`` on a datetime64[ns] array yields raw
    nanosecond integers, which are meaningless in a report. When
    the original object carries a datetime64 dtype, those ints are
    nanoseconds (or the unit in the dtype) since the epoch. The
    minimum int64 value is NaT and becomes None (missing).
    Date/datetime objects returned by tolist() are left alone.
    """
    dtype_str = str(getattr(original, "dtype", "") or "")
    if "datetime64" not in dtype_str:
        return values
    unit = "ns"
    if "[" in dtype_str:
        unit = dtype_str.split("[", 1)[1].rstrip("]")
    scales = {
        "ns": 1_000_000_000,
        "us": 1_000_000,
        "ms": 1_000,
        "s": 1,
        "m": 60,
        "h": 3600,
        "D": 86400,
        "W": 604800,
    }
    scale = scales.get(unit)
    if scale is None:
        return values
    out = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
        elif isinstance(v, int) and v == -9223372036854775808:
            out.append(None)
        elif isinstance(v, int):
            out.append(datetime.fromtimestamp(v / scale, tz=timezone.utc).isoformat())
        else:
            out.append(v)
    return out


def _as_list(value, length, name, convert_datetime=False):
    if value is None:
        return None
    _as_list_original = value
    # FA-008: a mapping, a string, or a 2-D shape is not a 1-D
    # sequence of values, and must fail naming the argument.
    if isinstance(value, dict):
        raise ValueError(f"{name} must be a one-dimensional sequence, got a mapping")  # noqa: TRY004
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence, not a string")  # noqa: TRY004
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(shape)}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        value = list(value)
    except TypeError:
        raise ValueError(
            f"{name} must be a one-dimensional sequence of length "
            f"{length}, got a scalar {value!r}"
        ) from None
    if convert_datetime:
        value = _convert_datetime64(value, _as_list_original)
    if len(value) != length:
        raise ValueError(f"{name} has length {len(value)}, expected {length}")
    # FA-008: nested sequences are a second dimension, and
    # unhashable values cannot be counted or grouped.
    # P-001: a hashable tuple is an atomic label, not a nested
    # value, so only list/dict/set elements are rejected here;
    # an unhashable tuple (e.g. containing a list) is still
    # rejected by the hash check below.
    for element in value:
        if isinstance(element, (list, dict, set)):
            raise ValueError(  # noqa: TRY004
                f"{name} must be one-dimensional, got a nested value {element!r}"
            )
        try:
            hash(element)
        except TypeError:
            raise ValueError(
                f"{name} contains an unhashable value {element!r}; "
                "values must be hashable"
            ) from None
    return value


def _counts(indices, labels):
    counts: Counter = Counter()
    for i in indices:
        counts[labels[i]] += 1
    return dict(counts)


def _rates(counts, total):
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _validate_one_fold(fold_no, train_idx, test_idx, n_samples):
    """Validate one fold's indices, naming the fold and side."""
    for side, indices in (("train", train_idx), ("test", test_idx)):
        for raw in indices:
            if isinstance(raw, bool):
                raise ValueError(  # noqa: TRY004
                    f"fold {fold_no} {side} set contains a non-integer index {raw!r}"
                )
            try:
                idx = operator.index(raw)
            except TypeError:
                raise ValueError(
                    f"fold {fold_no} {side} set contains a non-integer index {raw!r}"
                ) from None
            if idx < 0 or idx >= n_samples:
                raise ValueError(
                    f"fold {fold_no} {side} set contains index {idx}, "
                    f"but only {n_samples} samples were provided"
                )


def _validate_indices(folds, n_samples):
    """Fail with an actionable message on bad fold indices.

    A bare IndexError from deep inside the counting loop tells the
    user nothing about what they did wrong. This check runs once,
    before any counting, and names the fold, the side (train/test),
    and the offending index. It only has teeth when n_samples comes
    from X/y/groups/time; when no such input exists, n_samples was
    itself inferred from the largest index, so there is no external
    bound to check against.
    """
    for fold_no, (train_idx, test_idx) in enumerate(folds):
        _validate_one_fold(fold_no, train_idx, test_idx, n_samples)


def _iter_folds(cv, X, y, groups, time=None):
    """Normalise a splitter or an iterable of index pairs into folds.

    Splitter arguments are dispatched by parameter NAME (FA-001,
    FA-002, FA-003), not by positional count:
    - a parameter named y gets y, groups gets groups, and
      time/times gets time;
    - keyword-only parameters are passed by keyword;
    - a *args parameter, or an unavailable signature, falls back
      to the scikit-learn convention split(X, y, groups);
    - an unknown parameter with a default is left to its default;
      an unknown required parameter falls back to the next
      positional value (X, y, groups in order) or raises a clear
      error when there is no value to give it.
    Splitter errors themselves are never swallowed.
    """
    # Strings have a .split() method too, but a string is never a
    # splitter or a list of folds (FA-011).
    if isinstance(cv, (str, bytes)):
        raise TypeError(
            "cv must be a splitter with a .split() method or an "
            "iterable of (train_idx, test_idx) pairs, not a string"
        )
    split = getattr(cv, "split", None)
    if split is not None:
        try:
            params = list(inspect.signature(split).parameters.values())
        except (TypeError, ValueError):
            params = None
        available = {
            "X": X,
            "y": y,
            "groups": groups,
            "time": time,
            "times": time,
        }
        # FA-003: variadic or unknown signatures get the sklearn
        # convention, not zero arguments.
        if params is None or any(prm.kind == prm.VAR_POSITIONAL for prm in params):
            result = split(X, y, groups)
            for train, test in result:
                yield list(train), list(test)
            return
        args: list = []
        kwargs: dict = {}
        positional_fallback = [X, y, groups]
        fallback_pos = 0
        # P-005: positional slots already consumed. A skipped
        # defaulted positional slot, or a recognised argument
        # passed by keyword because of such a gap, is recorded
        # in pending, in parameter order. A later positional-only
        # parameter flushes pending into args in that order, so
        # no argument ever lands in the wrong slot. Recognised
        # positional-or-keyword arguments after a gap go by
        # keyword, e.g. split(X, unused=None, groups=None) is
        # called as split(X, groups=groups), not split(X, groups)
        # with groups landing in unused.
        pending: list = []  # (name or None, value, is_kwarg)
        gap = False

        def flush_pending():
            nonlocal gap
            for pending_name, pending_value, is_kwarg in pending:
                if is_kwarg:
                    kwargs.pop(pending_name, None)
                args.append(pending_value)
            pending.clear()
            gap = False

        for prm in params:
            if prm.kind == prm.VAR_KEYWORD:
                continue
            if prm.name in available:
                value = available[prm.name]
                if prm.kind == prm.KEYWORD_ONLY:
                    kwargs[prm.name] = value
                elif prm.kind == prm.POSITIONAL_ONLY:
                    if gap:
                        flush_pending()
                    args.append(value)
                elif gap:
                    kwargs[prm.name] = value
                    pending.append((prm.name, value, True))
                else:
                    args.append(value)
                continue
            if prm.kind == prm.KEYWORD_ONLY:
                if prm.default is not inspect.Parameter.empty:
                    continue
                raise TypeError(
                    "cannot determine a value for keyword-only "
                    f"parameter {prm.name!r} of {split!r}; audit() "
                    "can supply X, y, groups and time"
                )
            # Positional-only or positional-or-keyword, unknown name.
            if prm.default is not inspect.Parameter.empty:
                if prm.kind == prm.POSITIONAL_ONLY:
                    # A positional slot must be filled; use the
                    # next fallback value so the call shape holds.
                    if gap:
                        flush_pending()
                    if fallback_pos >= len(positional_fallback):
                        raise TypeError(
                            "cannot determine a value for parameter "
                            f"{prm.name!r} of {split!r}"
                        )
                    args.append(positional_fallback[fallback_pos])
                    fallback_pos += 1
                else:
                    # Skipped defaulted slot: record it, do not
                    # let the next argument fill it positionally.
                    pending.append((None, prm.default, False))
                    gap = True
                continue
            if fallback_pos >= len(positional_fallback):
                raise TypeError(
                    "cannot determine a value for required parameter "
                    f"{prm.name!r} of {split!r}; audit() can supply "
                    "X, y, groups and time"
                )
            if prm.kind == prm.POSITIONAL_ONLY:
                if gap:
                    flush_pending()
                args.append(positional_fallback[fallback_pos])
            elif gap:
                # Passable by keyword under its own name, so it
                # cannot land in a skipped slot.
                kwargs[prm.name] = positional_fallback[fallback_pos]
                pending.append((prm.name, positional_fallback[fallback_pos], True))
            else:
                args.append(positional_fallback[fallback_pos])
            fallback_pos += 1
        result = split(*args, **kwargs)
        for train, test in result:
            yield list(train), list(test)
        return
    # FA-011: any iterable of pairs is accepted, not only lists;
    # malformed folds name their fold.
    try:
        pairs = list(cv)
    except TypeError:
        raise TypeError(
            "cv must be a splitter with a .split() method or an "
            "iterable of (train_idx, test_idx) pairs"
        ) from None
    # A single pair (train, test) of int sequences is one fold,
    # not two malformed folds. P-006: the sequences need not be
    # lists/tuples; two 1-D numpy integer arrays count too.
    if len(pairs) == 2 and all(_is_1d_int_sequence(s) for s in pairs):
        pairs = [tuple(pairs)]
    for fold_no, pair in enumerate(pairs):
        try:
            train, test = pair
        except (TypeError, ValueError):
            raise ValueError(
                f"fold {fold_no} is not a (train_idx, test_idx) pair: {pair!r}"
            ) from None
        yield list(train), list(test)


def _fmt_percent(rate: float) -> str:
    """Adaptive percent: rates under 1% never print as 0% (FA-022)."""
    if 0 < rate < 0.01:
        return f"{rate:.2%}"
    return f"{rate:.0%}"


def _fmt_ratio(ratio: float) -> str:
    """Two significant digits, never scientific notation (FA-022)."""
    text = f"{ratio:.2g}"
    if "e" in text:
        text = f"{ratio:.1f}"
    return text


def _fmt_num(value) -> str:
    """Median-style number without a trailing .0 (FA-017)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _frac(value) -> Fraction:
    """Exact fraction from a decimal threshold, e.g. 0.05 -> 1/20."""
    return Fraction(str(value))


_MISSING_GROUP = object()


def _display_group(group):
    return "missing" if group is _MISSING_GROUP else group


def audit(
    cv, X=None, y=None, groups=None, time=None, thresholds: Thresholds | None = None
) -> Report:
    """Audit a cross-validation split.

    Parameters
    ----------
    cv:
        A splitter with ``.split()`` (e.g. KFold, StratifiedKFold,
        GroupKFold, TimeSeriesSplit) or an iterable of
        ``(train_idx, test_idx)`` pairs. Splitter arguments are
        dispatched by parameter name: a parameter named ``groups``
        receives ``groups`` and a parameter named ``time``/``times``
        receives ``time`` (FA-001).
    X:
        Features, only used to infer the number of samples and to
        pass to a splitter. May be None when ``cv`` is a list of
        index pairs and ``y`` is given.
    y:
        Labels, one per sample. Required for class checks; if None,
        class checks are skipped. Missing or infinite labels raise
        ValueError (FA-004).
    groups:
        Group label per sample. Missing group ids are normalised to
        one missing group and warned about (``missing_group_ids``);
        a group on both sides is group leakage (FA-005).
    time:
        Time value per sample, for the train-before-test check.
        String times raise ValueError; missing times are warned
        about (``missing_time``) and excluded from min/max, so the
        verdict never depends on index order (FA-006, FA-027).
        Equal train-max/test-min timestamps warn as
        ``time_tie_at_boundary``: a tie is not treated as a proven
        leak, but it is not silently passed either (FA-018).
    thresholds:
        A :class:`Thresholds` object; defaults are used when None.

    Returns
    -------
    Report with a per-fold table, flags, and a verdict.
    """
    if thresholds is None:
        thresholds = Thresholds()
    elif not isinstance(thresholds, Thresholds):
        raise TypeError(
            f"thresholds must be a Thresholds object, got {type(thresholds).__name__}"
        )

    # P-002: materialise a generator/iterator time input once,
    # before anything consumes it. A generator has no length and
    # cannot be re-iterated, so the string-time validation pass
    # previously consumed it and the audit pass then saw an
    # empty input. Re-iterable inputs (lists, arrays, ranges)
    # are left alone.
    if (
        time is not None
        and not isinstance(time, (str, bytes, dict))
        and not hasattr(time, "shape")
        and not hasattr(time, "tolist")
    ):
        try:
            if iter(time) is time:
                time = list(time)
        except TypeError:
            pass

    # Infer the number of samples, shape-first so sparse X works
    # (FA-009: len() of a sparse matrix raises TypeError).
    n_samples = None
    for candidate in (y, groups, time, X):
        candidate_len = _length_of(candidate)
        if candidate_len is not None:
            n_samples = candidate_len
            break
    # FA-010: X must agree with the other inputs; a shorter X was
    # previously accepted silently because folds came from X alone.
    if X is not None and n_samples is not None:
        x_len = _length_of(X)
        if x_len is not None and x_len != n_samples:
            raise ValueError(
                f"X has length {x_len}, expected {n_samples} "
                "(length of y, groups or time)"
            )
    if n_samples is None:
        # Fall back to the largest index seen in the folds.
        folds_preview = list(_iter_folds(cv, X, y, groups, time))
        n_samples = 1 + max(
            (i for pair in folds_preview for idx in pair for i in idx),
            default=-1,
        )
        folds_iter = iter(folds_preview)
    else:
        folds_iter = _iter_folds(cv, X, y, groups, time)

    labels = _as_list(y, n_samples, "y") if y is not None else None
    group_list = _as_list(groups, n_samples, "groups") if groups is not None else None
    # FA-027: string times are checked on the RAW input, before
    # FA-014 converts datetime64 ints to ISO strings (which are
    # also strings, but are not user-supplied string times).
    if time is not None:
        time_shape = getattr(time, "shape", None)
        if time_shape is None or len(time_shape) <= 1:
            raw_time = time.tolist() if hasattr(time, "tolist") else time
            try:
                raw_time_list = list(raw_time)
            except TypeError:
                raw_time_list = []
            for pos, value in enumerate(raw_time_list):
                if isinstance(value, (str, bytes)):
                    raise ValueError(  # noqa: TRY004
                        f"time value at position {pos} is a string "
                        f"({value!r}); pass numeric timestamps or "
                        "datetime objects, not strings"
                    )
    time_list = (
        _as_list(time, n_samples, "time", convert_datetime=True)
        if time is not None
        else None
    )

    # FA-004: missing or infinite labels are rejected, as in
    # scikit-learn. Counting them would create one phantom class
    # per NaN row with ndarray/Series input.
    if labels is not None:
        for pos, label in enumerate(labels):
            if _is_invalid_label(label):
                raise ValueError(
                    f"y contains a missing or invalid label at position "
                    f"{pos}: {label!r}; labels must be present and finite"
                )

    overall_counts = _counts(range(n_samples), labels) if labels else {}
    overall_rates = _rates(overall_counts, n_samples) if labels else {}

    # FA-015: a float target with more than half its values unique
    # is a continuous target, not a classification. Class checks
    # would flood the report, so they are skipped with one warning.
    skip_class_checks = False
    continuous_flag = None
    if (
        labels is not None
        and len(overall_counts) > n_samples / 2
        and any(isinstance(v, float) and not isinstance(v, bool) for v in labels)
    ):
        skip_class_checks = True
        continuous_flag = Flag(
            "WARNING",
            "continuous_target",
            f"y has {len(overall_counts)} distinct float values "
            f"for {n_samples} rows, so it looks continuous; "
            "class checks are skipped.",
            fold=None,
        )

    fold_results: list[FoldResult] = []
    all_flags: list[Flag] = []
    if continuous_flag is not None:
        all_flags.append(continuous_flag)
    test_seen: Counter = Counter()
    # FA-024: test sets are kept as bitmasks (fast pairwise overlap
    # in C) when n_samples is manageable, otherwise as sets.
    use_bitmask = n_samples <= 250000
    test_masks: list[int] = []
    test_sets: list[set] = []

    fold_no = -1
    for fold_no, (train_idx, test_idx) in enumerate(folds_iter):
        _validate_one_fold(fold_no, train_idx, test_idx, n_samples)
        flags: list[Flag] = []

        def add(severity, code, message, _fold=fold_no, _flags=flags):
            flag = Flag(severity, code, message, fold=_fold)
            _flags.append(flag)
            all_flags.append(flag)

        train_counts = _counts(train_idx, labels) if labels else {}
        test_counts = _counts(test_idx, labels) if labels else {}
        train_rates = _rates(train_counts, len(train_idx)) if labels else {}
        test_rates = _rates(test_counts, len(test_idx)) if labels else {}

        # Groups, computed before the labels block so the pooled
        # (LOO/LOGO) rule below can use them (FA-026).
        train_groups: set = set()
        test_groups: set = set()
        n_train_groups = n_test_groups = None
        if group_list is not None:
            train_groups = {
                _MISSING_GROUP if _is_missing(group_list[i]) else group_list[i]
                for i in train_idx
            }
            test_groups = {
                _MISSING_GROUP if _is_missing(group_list[i]) else group_list[i]
                for i in test_idx
            }
            n_train_groups = len(train_groups)
            n_test_groups = len(test_groups)
        # FA-026: per-fold checks that are meaningless by
        # construction for a pooled scheme are INFO, not FAIL/WARNING:
        # a 1-row test fold, or a test fold that is exactly one
        # group absent from train (LeaveOneGroupOut).
        is_pooled = len(test_idx) == 1 or (
            group_list is not None
            and len(test_groups) == 1
            and test_groups.isdisjoint(train_groups)
        )

        # --- Validity flags ---
        if not train_idx:
            add("FAIL", "empty_train", "Train set is empty.")
        if not test_idx:
            add("FAIL", "empty_test", "Test set is empty.")
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            examples = sorted(overlap)[:5]
            add(
                "FAIL",
                "train_test_overlap",
                f"{len(overlap)} rows appear in both the train and "
                f"test sets of this fold (e.g. {examples}); "
                "the fold's score would be meaningless.",
            )
        for side, indices in (("train", train_idx), ("test", test_idx)):
            duplicates = len(indices) - len(set(indices))
            if duplicates:
                add(
                    "WARNING",
                    "duplicate_indices",
                    f"{duplicates} duplicate index "
                    f"{'entry' if duplicates == 1 else 'entries'} "
                    f"in the {side} set; the row is counted twice.",
                )
        if test_idx and len(test_idx) < thresholds.min_test_size:
            add(
                "INFO" if is_pooled else "WARNING",
                "small_test",
                f"Test set has only {len(test_idx)} rows "
                f"(minimum {thresholds.min_test_size}).",
            )
        if labels and not skip_class_checks:
            single_class_train = bool(train_idx) and len(train_counts) == 1
            if single_class_train:
                add(
                    "FAIL",
                    "single_class_train",
                    f"Train set has a single class: {next(iter(train_counts))}.",
                )
            single_class_test = bool(test_idx) and len(test_counts) == 1
            if single_class_test:
                add(
                    "INFO" if is_pooled else "FAIL",
                    "single_class_test",
                    f"Test set has a single class: {next(iter(test_counts))}.",
                )
            # P3 suppression, mirrored for train (FA-019): when a
            # side has a single class, the missing classes on that
            # side are the same fact, not separate findings.
            if not single_class_train:
                for cls in overall_counts:
                    if cls not in train_counts and train_idx:
                        add(
                            "WARNING",
                            "missing_class_train",
                            f"Class {cls!r} is missing from the train set.",
                        )
            if not single_class_test:
                for cls in overall_counts:
                    if cls not in test_counts and test_idx:
                        add(
                            "WARNING",
                            "missing_class_test",
                            f"Class {cls!r} is missing from the test set.",
                        )
                dev_threshold = _frac(thresholds.class_rate_deviation)
                ratio_threshold = _frac(thresholds.class_rate_ratio)
                floor = Fraction(5, 100)
                for cls, overall_rate in overall_rates.items():
                    rate = test_rates.get(cls, 0.0)
                    if rate == 0.0:
                        continue
                    rate_frac = Fraction(test_counts[cls], len(test_idx))
                    overall_frac = Fraction(overall_counts[cls], n_samples)
                    if abs(rate_frac - overall_frac) > dev_threshold:
                        add(
                            "WARNING",
                            "class_rate_deviation",
                            f"Class {cls!r} rate in test is "
                            f"{_fmt_percent(rate)} "
                            f"vs {_fmt_percent(overall_rate)} overall.",
                        )
                    elif (
                        overall_frac > 0
                        and abs(rate_frac - overall_frac) >= floor
                        and (
                            rate_frac / overall_frac > ratio_threshold
                            or rate_frac / overall_frac < 1 / ratio_threshold
                        )
                    ):
                        ratio = rate_frac / overall_frac
                        add(
                            "WARNING",
                            "class_rate_ratio_deviation",
                            f"Class {cls!r} rate in test is "
                            f"{_fmt_percent(rate)} "
                            f"vs {_fmt_percent(overall_rate)} overall "
                            f"({_fmt_ratio(float(ratio))}x).",
                        )

        # --- Groups (counts and overlap) ---
        if group_list is not None:
            # FA-005: missing group ids are one missing group, not
            # one group per row, and are warned about with a count.
            missing_in_fold = sum(
                1 for i in train_idx + test_idx if _is_missing(group_list[i])
            )
            if missing_in_fold:
                add(
                    "WARNING",
                    "missing_group_ids",
                    f"{missing_in_fold} rows in this fold have a "
                    "missing group id; group leakage cannot be "
                    "assessed for them.",
                )
            overlap_groups = train_groups & test_groups
            if overlap_groups:
                # FA-021: deduplicate before slicing, so the
                # examples are distinct groups.
                examples = []
                seen_groups = set()
                for i in train_idx:
                    g = _MISSING_GROUP if _is_missing(group_list[i]) else group_list[i]
                    if g in overlap_groups and g not in seen_groups:
                        seen_groups.add(g)
                        examples.append(_display_group(g))
                    if len(examples) == 3:
                        break
                n_overlap = len(overlap_groups)
                add(
                    "FAIL",
                    "group_overlap",
                    f"{n_overlap} group{'s' if n_overlap != 1 else ''} "
                    f"{'appear' if n_overlap != 1 else 'appears'} in both "
                    "the train and test sets of this fold "
                    f"(e.g. {examples}); group leakage makes the "
                    "fold's score meaningless.",
                )

        # --- Time order ---
        train_tmin = train_tmax = test_tmin = test_tmax = None
        if time_list is not None and train_idx and test_idx:
            # FA-006: missing times are excluded from min/max and
            # warned about, so the verdict cannot depend on the
            # order of indices inside a fold.
            train_times = [
                time_list[i] for i in train_idx if not _is_missing(time_list[i])
            ]
            test_times = [
                time_list[i] for i in test_idx if not _is_missing(time_list[i])
            ]
            missing_in_fold = (len(train_idx) - len(train_times)) + (
                len(test_idx) - len(test_times)
            )
            if missing_in_fold:
                add(
                    "WARNING",
                    "missing_time",
                    f"{missing_in_fold} rows in this fold have a "
                    "missing timestamp; time order is assessed "
                    "from the remaining timestamps.",
                )
            try:
                if train_times:
                    train_tmin, train_tmax = min(train_times), max(train_times)
                if test_times:
                    test_tmin, test_tmax = min(test_times), max(test_times)
                if (
                    train_tmax is not None
                    and test_tmin is not None
                    and train_tmax > test_tmin
                ):
                    add(
                        "FAIL",
                        "time_order",
                        "Train times extend past the start of test times "
                        f"(train max {train_tmax} > test min {test_tmin}); "
                        "the model would learn from the future.",
                    )
                elif (
                    train_tmax is not None
                    and test_tmin is not None
                    and train_tmax == test_tmin
                ):
                    # FA-018 decision: a boundary tie is a WARNING,
                    # not a FAIL and not a silent pass. Timestamps
                    # at the same instant may be one batch, but
                    # they may also hide leakage, so the report
                    # must say so and the rule is documented here
                    # and in the README.
                    add(
                        "WARNING",
                        "time_tie_at_boundary",
                        "Train max time equals test min time "
                        f"({train_tmax}); a boundary tie may hide "
                        "temporal leakage.",
                    )
            except TypeError:
                raise ValueError(
                    "time values must be comparable (all numeric or "
                    "all datetime, with the same timezone awareness); "
                    "mixed or incompatible time types cannot be ordered"
                ) from None

        fold_results.append(
            FoldResult(
                fold=fold_no,
                n_train=len(train_idx),
                n_test=len(test_idx),
                train_class_counts=train_counts,
                test_class_counts=test_counts,
                train_class_rates=train_rates,
                test_class_rates=test_rates,
                n_train_groups=n_train_groups,
                n_test_groups=n_test_groups,
                train_time_min=train_tmin,
                train_time_max=train_tmax,
                test_time_min=test_tmin,
                test_time_max=test_tmax,
                flags=flags,
            )
        )
        if use_bitmask:
            mask = 0
            for i in set(test_idx):
                mask |= 1 << i
            test_masks.append(mask)
        else:
            test_sets.append(set(test_idx))
        test_seen.update(test_idx)

    if fold_no < 0:
        raise ValueError("cv produced no folds")

    # --- Fold-size imbalance (needs all folds first) ---
    test_sizes = sorted(f.n_test for f in fold_results)
    # FA-017: the true median, not the upper middle element.
    median_size = statistics.median(test_sizes) if test_sizes else 0
    if median_size:
        for fold in fold_results:
            deviation = abs(fold.n_test - median_size) / median_size
            if deviation > thresholds.fold_size_deviation:
                flag = Flag(
                    "WARNING",
                    "fold_size_imbalance",
                    f"Test size {fold.n_test} differs from the median "
                    f"test size {_fmt_num(median_size)} by {deviation:.0%}.",
                    fold=fold.fold,
                )
                fold.flags.append(flag)
                all_flags.append(flag)

    # --- Coverage across folds ---
    never_tested = [i for i in range(n_samples) if test_seen[i] == 0]
    if never_tested:
        # P5: with a time column, never-tested rows are structural.
        # FA-020: a single hold-out is structural too: its
        # train-only rows are not a warning.
        if time_list is not None or len(fold_results) == 1:
            severity = "INFO"
            note = (
                " This is expected for time-series splits, where "
                "the earliest rows are train-only by design."
                if time_list is not None
                else " This is expected for a single hold-out split."
            )
        else:
            severity = "WARNING"
            note = ""
        all_flags.append(
            Flag(
                severity,
                "never_tested",
                f"{len(never_tested)} rows are never tested.{note}",
                fold=None,
            )
        )
    max_overlap = 0
    overlapping_pairs = 0
    if use_bitmask:
        for a in range(len(test_masks)):
            for b in range(a + 1, len(test_masks)):
                overlap = (test_masks[a] & test_masks[b]).bit_count()
                max_overlap = max(max_overlap, overlap)
                if overlap:
                    overlapping_pairs += 1
    else:
        for a in range(len(test_sets)):
            for b in range(a + 1, len(test_sets)):
                overlap = len(test_sets[a] & test_sets[b])
                max_overlap = max(max_overlap, overlap)
                if overlap:
                    overlapping_pairs += 1
    if overlapping_pairs:
        all_flags.append(
            Flag(
                "INFO",
                "overlapping_test_folds",
                f"{overlapping_pairs} pairs of test folds overlap "
                f"(max overlap {max_overlap} rows). This is expected for "
                "repeated or shuffled splitters.",
                fold=None,
            )
        )

    verdict = "OK"
    if any(f.severity == "FAIL" for f in all_flags):
        verdict = "FAIL"
    elif any(f.severity == "WARNING" for f in all_flags):
        verdict = "WARNING"

    return Report(
        folds=fold_results,
        flags=all_flags,
        verdict=verdict,
        n_samples=n_samples,
        never_tested=never_tested,
        max_test_overlap=max_overlap,
        n_overlapping_test_pairs=overlapping_pairs,
        test_counts={i: test_seen[i] for i in range(n_samples)},
    )
