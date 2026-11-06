"""Data containers and report rendering for foldaudit.

These classes are deliberately plain: a fold result is a dict-friendly
record, a flag is a severity + code + message, and a report can render
itself as text or JSON. Nothing here trains a model or draws a plot.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field


@dataclass
class Flag:
    """One finding for one fold (or for the split as a whole).

    severity is one of "FAIL", "WARNING" or "INFO".
    """

    severity: str
    code: str
    message: str
    fold: int | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "fold": self.fold,
        }


@dataclass
class FoldResult:
    """Per-fold measurements for one fold of a split."""

    fold: int
    n_train: int
    n_test: int
    train_class_counts: dict = field(default_factory=dict)
    test_class_counts: dict = field(default_factory=dict)
    train_class_rates: dict = field(default_factory=dict)
    test_class_rates: dict = field(default_factory=dict)
    n_train_groups: int | None = None
    n_test_groups: int | None = None
    train_time_min: object = None
    train_time_max: object = None
    test_time_min: object = None
    test_time_max: object = None
    flags: list[Flag] = field(default_factory=list)

    def to_dict(self) -> dict:
        # FA-013: class tables are records, not JSON objects, so
        # labels 1 and "1", or None and "null", cannot collide and
        # lose their type. FA-012: labels are stringified here, so
        # numpy scalars, tuples, datetimes, bytes and Decimal
        # labels cannot crash to_json().
        def counts_records(counts):
            return [
                {"label": str(k), "repr": repr(k), "count": int(v)}
                for k, v in counts.items()
            ]

        def rates_records(rates):
            return [
                {"label": str(k), "repr": repr(k), "rate": float(v)}
                for k, v in rates.items()
            ]

        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_class_counts": counts_records(self.train_class_counts),
            "test_class_counts": counts_records(self.test_class_counts),
            "train_class_rates": rates_records(self.train_class_rates),
            "test_class_rates": rates_records(self.test_class_rates),
            "n_train_groups": self.n_train_groups,
            "n_test_groups": self.n_test_groups,
            "train_time_min": _jsonable(self.train_time_min),
            "train_time_max": _jsonable(self.train_time_max),
            "test_time_min": _jsonable(self.test_time_min),
            "test_time_max": _jsonable(self.test_time_max),
            "flags": [f.to_dict() for f in self.flags],
        }


def _jsonable(value: object) -> object:
    # FA-012/FA-013/FA-014: numpy scalars become plain ints/floats,
    # datetimes become ISO strings, and non-finite floats become
    # strings, so to_json() never emits bare NaN/Infinity tokens.
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


@dataclass
class Report:
    """The result of :func:`foldaudit.audit`.

    verdict is "OK", "WARNING" or "FAIL": FAIL if any FAIL flag exists,
    otherwise WARNING if any WARNING flag exists, otherwise OK.
    """

    folds: list[FoldResult]
    flags: list[Flag] = field(default_factory=list)
    verdict: str = "OK"
    # Coverage summary across all folds (see core.audit for definitions).
    n_samples: int = 0
    never_tested: list[int] = field(default_factory=list)
    max_test_overlap: int = 0
    n_overlapping_test_pairs: int = 0
    # FA-025: per-row test counts, including 0 for never-tested rows.
    test_counts: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """A pasteable plain-text report."""
        lines = [f"foldaudit verdict: {self.verdict}", ""]

        # F5: the rates column is as wide as the longest rates
        # string in this report, with a minimum of 40, plus one
        # extra character so the longest fold still has a gap
        # before its verdict. Few-class reports keep width 40,
        # so their output is byte-identical to 0.2.1.
        # P-003: use the adaptive percentage formatter here, as
        # flag messages already do, so a 0.5% rate prints as
        # "0.50%", never as "0%".
        def _fmt_percent(rate: float) -> str:
            if 0 < rate < 0.01:
                return f"{rate:.2%}"
            return f"{rate:.0%}"

        rates_strings = [
            ", ".join(
                f"{k}={_fmt_percent(v)}" for k, v in fold.test_class_rates.items()
            )
            for fold in self.folds
        ]
        rates_width = max([40] + [len(s) + 1 for s in rates_strings])
        # FA-016: every column width is computed from the data,
        # with at least one gap character, so 8-digit sizes and
        # 6-digit fold numbers cannot run together.
        fold_width = max([6] + [len(str(f.fold)) + 1 for f in self.folds])
        train_width = max([8] + [len(str(f.n_train)) + 1 for f in self.folds])
        test_width = max([8] + [len(str(f.n_test)) + 1 for f in self.folds])
        header = (
            f"{'Fold':<{fold_width}}{'Train':<{train_width}}"
            f"{'Test':<{test_width}}"
            f"{'Test class rates':<{rates_width}}{'Verdict':<8}"
        )
        lines.append(header)
        for fold, rates in zip(self.folds, rates_strings):
            # P-004: derive the per-fold display verdict from
            # severity, FAIL > WARNING > INFO > OK, so an
            # INFO-only fold (e.g. a pooled leave-one-out fold)
            # displays INFO, not WARNING.
            if any(f.severity == "FAIL" for f in fold.flags):
                fold_verdict = "FAIL"
            elif any(f.severity == "WARNING" for f in fold.flags):
                fold_verdict = "WARNING"
            elif fold.flags:
                fold_verdict = "INFO"
            else:
                fold_verdict = "OK"
            lines.append(
                f"{fold.fold:<{fold_width}}{fold.n_train:<{train_width}}"
                f"{fold.n_test:<{test_width}}"
                f"{rates:<{rates_width}}{fold_verdict:<8}"
            )
        if self.never_tested:
            lines.append("")
            lines.append(
                f"Rows never tested ({len(self.never_tested)}): "
                f"{self.never_tested[:20]}"
                + (" ..." if len(self.never_tested) > 20 else "")
            )
        if self.flags:
            lines.append("")
            lines.append("Flags:")
            for flag in self.flags:
                where = f"fold {flag.fold}: " if flag.fold is not None else ""
                lines.append(f"  [{flag.severity}] {where}{flag.message}")
        # FA-029: an OK verdict must say what was not checked.
        lines.append("")
        lines.append(
            "Scope: Checks performed: fold validity, class balance, "
            "group overlap, time order, fold-size balance and "
            "coverage. Not checked: duplicate rows, feature/target "
            "leakage, preprocessing leakage, and semantic leakage "
            "beyond group labels."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "n_samples": self.n_samples,
            "never_tested": self.never_tested,
            "max_test_overlap": self.max_test_overlap,
            "n_overlapping_test_pairs": self.n_overlapping_test_pairs,
            "test_counts": self.test_counts,
            "scope": (
                "Checks performed: fold validity, class balance, "
                "group overlap, time order, fold-size balance and "
                "coverage. Not checked: duplicate rows, feature/target "
                "leakage, preprocessing leakage, and semantic leakage "
                "beyond group labels."
            ),
            "folds": [f.to_dict() for f in self.folds],
            "flags": [f.to_dict() for f in self.flags],
        }

    def to_json(self, indent: int = 2) -> str:
        """The report as JSON, for saving or for CI.

        Strict RFC 8259: allow_nan=False, so a non-finite value
        that slipped through _jsonable raises instead of emitting
        a bare NaN/Infinity token (FA-013).
        """
        return json.dumps(self.to_dict(), indent=indent, default=str, allow_nan=False)

    def raise_if_fail(self, fail_on_warning: bool = False) -> None:
        """Raise ValueError unless the verdict is acceptable.

        With ``fail_on_warning=False`` (the default) only FAIL raises.
        With ``fail_on_warning=True`` a WARNING verdict raises too.
        This is the hook for using foldaudit in CI.
        """
        if self.verdict == "FAIL" or (fail_on_warning and self.verdict == "WARNING"):
            raise ValueError(
                f"foldaudit verdict is {self.verdict}; see report.flags for details."
            )
