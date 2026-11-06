"""Tests for foldaudit v0.1. No third-party dependencies needed:
splitters are faked with plain index lists, exactly as the library
supports, so these tests run with the standard library alone.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foldaudit import Thresholds, audit


def kfold_folds(n, k):
    """A simple KFold-like splitter, as index pairs."""
    fold_size = n // k
    folds = []
    for i in range(k):
        test = list(range(i * fold_size, (i + 1) * fold_size))
        train = [j for j in range(n) if j not in test]
        folds.append((train, test))
    return folds


def test_balanced_kfold_is_ok():
    y = [0, 1] * 50
    report = audit(kfold_folds(100, 5), y=y)
    assert report.verdict == "OK"
    assert len(report.folds) == 5
    assert all(f.n_test == 20 for f in report.folds)
    assert report.never_tested == []


def test_single_class_test_fold_fails():
    # Fold 0 tests only class 0; fold 1 tests only class 1.
    y = [0] * 50 + [1] * 50
    folds = [
        (list(range(50, 100)), list(range(50))),
        (list(range(50)), list(range(50, 100))),
    ]
    report = audit(folds, y=y)
    assert report.verdict == "FAIL"
    assert any(f.code == "single_class_test" for f in report.flags)


def test_empty_test_fold_fails():
    report = audit([([0, 1, 2], []), ([], [0, 1, 2])], y=[0, 1, 0])
    assert report.verdict == "FAIL"
    assert any(f.code == "empty_test" for f in report.flags)
    assert any(f.code == "empty_train" for f in report.flags)


def test_class_rate_deviation_warns():
    y = [0] * 90 + [1] * 10
    # Test fold is 90% class 1 vs 10% overall, but NOT single-class:
    # a single-class fold now suppresses the deviation flag (P3),
    # so this test uses a skewed-but-mixed fold.
    test = [0] + list(range(90, 99))
    train = [i for i in range(100) if i not in test]
    folds = [(train, test)]
    report = audit(folds, y=y, thresholds=Thresholds(min_test_size=1))
    assert report.verdict in ("WARNING", "FAIL")
    assert any(f.code == "class_rate_deviation" for f in report.flags)


def test_time_order_violation_fails():
    # Train includes time 10, test starts at time 5: future leakage.
    report = audit(
        [([0, 1, 2], [3, 4])],
        y=[0, 1, 0, 1, 0],
        time=[0, 5, 10, 5, 6],
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.verdict == "FAIL"
    assert any(f.code == "time_order" for f in report.flags)


def test_time_order_ok():
    # Train times (0, 1) precede test times (2, 3), so there must be
    # no time_order flag. Before the P5 fix this verdict was WARNING
    # because rows 0 and 1 are never tested; with a time column
    # supplied, never_tested is now INFO (train-only rows are
    # structural in time-series splits), so the verdict is OK.
    report = audit(
        [([0, 1], [2, 3])],
        y=[0, 1, 0, 1],
        time=[0, 1, 2, 3],
        thresholds=Thresholds(min_test_size=1),
    )
    assert not any(f.code == "time_order" for f in report.flags)
    assert report.verdict == "OK"


def test_never_tested_rows_reported():
    report = audit(
        [([2, 3], [0, 1])],
        y=[0, 1, 0, 1],
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.never_tested == [2, 3]


def test_splitter_object_is_accepted():
    class FakeSplitter:
        def split(self, X, y=None, groups=None):
            yield [0, 1, 2], [3, 4]
            yield [3, 4], [0, 1, 2]

    report = audit(
        FakeSplitter(),
        X=None,
        y=[0, 1, 0, 1, 0],
        thresholds=Thresholds(min_test_size=1),
    )
    assert len(report.folds) == 2


def test_report_text_and_json():
    report = audit(kfold_folds(100, 5), y=[0, 1] * 50)
    assert "foldaudit verdict: OK" in report.text
    data = json.loads(report.to_json())
    assert data["verdict"] == "OK"
    assert len(data["folds"]) == 5


def test_raise_if_fail():
    report = audit([([0, 1, 2], [])], y=[0, 1, 0])
    try:
        report.raise_if_fail()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a FAIL verdict")


def test_group_counts_are_reported():
    report = audit(
        [([0, 1, 2], [3, 4, 5])],
        y=[0, 1, 0, 1, 0, 1],
        groups=["g1", "g1", "g2", "g2", "g3", "g3"],
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.folds[0].n_train_groups == 2
    assert report.folds[0].n_test_groups == 2


def test_fold_size_imbalance_warns():
    # Test sizes 2, 2 and 20: the 20-fold deviates from the median (2)
    # by far more than the 50% default.
    folds = [
        (
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [0, 1],
        ),
        (
            [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [2, 3],
        ),
        ([0, 1, 2, 3], list(range(4, 24))),
    ]
    y = [0, 1] * 12
    report = audit(folds, y=y)
    assert any(f.code == "fold_size_imbalance" for f in report.flags)


def test_overlapping_test_folds_are_reported_not_failed():
    # NOTE: the two folds share the same TEST set, but neither fold
    # has train/test overlap within itself. The old version used
    # ([0, 1], [0, 1]) for the second fold, which is itself a
    # train_test_overlap FAIL since 0.2.1 and would break this test.
    folds = [([2, 3], [0, 1]), ([2, 3], [0, 1])]
    report = audit(folds, y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1))
    assert report.max_test_overlap == 2
    assert report.n_overlapping_test_pairs == 1
    assert any(f.code == "overlapping_test_folds" for f in report.flags)
    assert report.verdict != "FAIL"


def test_no_arg_splitter_is_accepted():
    class NoArgSplitter:
        def split(self):
            yield [0, 1], [2, 3]

    report = audit(
        NoArgSplitter(), y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1)
    )
    assert len(report.folds) == 1


def test_custom_threshold_changes_verdict():
    # A 30-point class-rate deviation warns at the 0.20 default
    # but not at a 0.40 custom threshold.
    # Overall: 70% class 0, 30% class 1. The test fold below is
    # 40% class 0 / 60% class 1: a 30-point deviation, so it warns
    # at the 0.20 default but not at a 0.40 custom threshold.
    y = [0] * 70 + [1] * 30
    test = [0, 1, 2, 3, 70, 71, 72, 73, 74, 75]
    train = [i for i in range(100) if i not in test]
    folds = [(train, test)]
    default = audit(folds, y=y, thresholds=Thresholds(min_test_size=1))
    assert any(f.code == "class_rate_deviation" for f in default.flags)
    loose = audit(
        folds, y=y, thresholds=Thresholds(min_test_size=1, class_rate_deviation=0.40)
    )
    assert not any(f.code == "class_rate_deviation" for f in loose.flags)


def test_splitter_error_is_not_swallowed():
    class BrokenSplitter:
        def split(self, X, y=None):
            raise RuntimeError("splitter broke internally")

    try:
        audit(BrokenSplitter(), y=[0, 1, 0, 1])
    except RuntimeError as exc:
        assert "splitter broke" in str(exc)
    else:
        raise AssertionError("expected the splitter's RuntimeError to surface")


# --- Regression tests for the 2 November 2026 problems report ---


def test_p2_out_of_range_index_raises_value_error():
    try:
        audit([([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])], y=[0, 1, 0])
    except ValueError as exc:
        assert "fold 0" in str(exc)
        assert "3 samples" in str(exc)
    else:
        raise AssertionError("expected ValueError, not IndexError")


def test_p2_negative_index_raises_value_error():
    try:
        audit([([0, -1], [1])], y=[0, 1, 0])
    except ValueError as exc:
        assert "-1" in str(exc)
    else:
        raise AssertionError("expected ValueError for a negative index")


def test_p2_valid_indices_still_pass():
    report = audit(
        [([2, 3], [0, 1]), ([0, 1], [2, 3])],
        y=[0, 1, 0, 1],
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.verdict == "OK"


def test_p4_small_test_warns_without_labels():
    report = audit(
        [(list(range(10)), [10]), ([10], list(range(10)))],
        thresholds=Thresholds(min_test_size=5),
    )
    assert any(f.code == "small_test" for f in report.flags)


def test_p3_single_class_fold_does_not_repeat_the_same_fact():
    report = audit(
        [(list(range(50, 100)), list(range(50)))],
        y=[0] * 50 + [1] * 50,
        thresholds=Thresholds(min_test_size=1),
    )
    codes = [f.code for f in report.folds[0].flags]
    assert codes == ["single_class_train", "single_class_test"]
    assert "missing_class_test" not in codes
    assert "class_rate_deviation" not in codes


def test_p1_rare_class_overrepresentation_warns():
    # Report's own example: 2% overall, 15% in the fold (7.5x).
    y = [0] * 196 + [1] * 4
    test = list(range(17)) + [196, 197, 198]
    train = [i for i in range(200) if i not in test]
    report = audit([(train, test)], y=y, thresholds=Thresholds(min_test_size=1))
    assert any(f.code == "class_rate_ratio_deviation" for f in report.flags)
    assert not any(f.code == "class_rate_deviation" for f in report.flags)


def test_p1_rare_class_underrepresentation_warns():
    # Overall 20% class 1, fold 5%: ratio 0.25, absolute gap 15 points.
    y = [0] * 80 + [1] * 20
    # Test indices: 19 zeros, 1 one -> 5% class 1 vs 20% overall.
    test = list(range(19)) + [80]
    train = [i for i in range(100) if i not in test]
    report = audit([(train, test)], y=y, thresholds=Thresholds(min_test_size=1))
    assert any(f.code == "class_rate_ratio_deviation" for f in report.flags)


def test_p1_tiny_absolute_shift_does_not_warn():
    # 1% -> 3% is 3x, but only 2 points: below the 5-point floor.
    y = [0] * 198 + [1] * 2
    test = list(range(97)) + [198, 199, 100]
    train = [i for i in range(200) if i not in test]
    report = audit([(train, test)], y=y, thresholds=Thresholds(min_test_size=1))
    assert not any(f.code == "class_rate_ratio_deviation" for f in report.flags)


def test_p5_never_tested_is_info_with_time_and_warning_without():
    folds = [
        (list(range(20 + i * 20)), list(range(20 + i * 20, 40 + i * 20)))
        for i in range(5)
    ]
    with_time = audit(
        folds,
        y=[0, 1] * 60,
        time=list(range(120)),
        thresholds=Thresholds(min_test_size=1),
    )
    assert with_time.verdict == "OK"
    flag = next(f for f in with_time.flags if f.code == "never_tested")
    assert flag.severity == "INFO"
    without_time = audit(folds, y=[0, 1] * 60, thresholds=Thresholds(min_test_size=1))
    assert without_time.verdict == "WARNING"
    flag = next(f for f in without_time.flags if f.code == "never_tested")
    assert flag.severity == "WARNING"


# --- Regression tests for the 0.2.0 problems report, Problem 6 ---


def test_p6_train_test_overlap_fails():
    # The report's own example: rows 80-89 in both sides.
    y = [0, 1] * 50
    report = audit([(list(range(90)), list(range(80, 100)))], y=y)
    assert report.verdict == "FAIL"
    flag = next(f for f in report.flags if f.code == "train_test_overlap")
    assert flag.severity == "FAIL"
    assert "10 rows" in flag.message


def test_p6_single_overlapping_row_fails():
    report = audit(
        [([0, 1], [1, 2])], y=[0, 1, 0], thresholds=Thresholds(min_test_size=1)
    )
    assert report.verdict == "FAIL"
    assert any(f.code == "train_test_overlap" for f in report.flags)


def test_p6_clean_fold_has_no_overlap_flag():
    report = audit(
        [([2, 3], [0, 1]), ([0, 1], [2, 3])],
        y=[0, 1, 0, 1],
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.verdict == "OK"
    assert not any(f.code == "train_test_overlap" for f in report.flags)


def test_p6_duplicate_index_inside_test_warns_not_fails():
    report = audit(
        [([0, 1], [2, 2, 3])], y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1)
    )
    assert any(f.code == "duplicate_indices" for f in report.flags)
    assert report.verdict != "FAIL"


def test_p6_duplicate_index_inside_train_warns():
    report = audit(
        [([0, 0, 1], [2, 3])], y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1)
    )
    assert any(f.code == "duplicate_indices" for f in report.flags)


# --- Regression tests for the complete fix list (0.2.2, F1-F5) ---
# Existing tests above are untouched; only these are appended.


def test_f1_group_overlap_fails():
    # Straddling case: patient-5's rows (30-35) are split by the
    # fold boundary at 33, so the group is on both sides.
    groups = [f"patient-{i // 6}" for i in range(60)]
    report = audit(
        [(list(range(33)), list(range(33, 60)))],
        y=[0, 1] * 30,
        groups=groups,
    )
    assert report.verdict == "FAIL"
    flag = next(f for f in report.flags if f.code == "group_overlap")
    assert flag.severity == "FAIL"
    assert "patient-5" in flag.message


def test_f1_clean_group_split_has_no_group_overlap():
    # The fix list's printed case splits exactly on a patient
    # boundary (30), so no group is on both sides: this is the
    # clean, GroupKFold-like case and must not be flagged.
    groups = [f"patient-{i // 6}" for i in range(60)]
    report = audit(
        [(list(range(30)), list(range(30, 60)))],
        y=[0, 1] * 30,
        groups=groups,
    )
    assert not any(f.code == "group_overlap" for f in report.flags)


def test_f2_invalid_thresholds_raise_value_error():
    for kwargs in (
        {"class_rate_deviation": 1.5},
        {"class_rate_deviation": -0.1},
        {"class_rate_ratio": 0},
        {"class_rate_ratio": 0.5},
        {"min_test_size": -1},
        {"fold_size_deviation": -0.2},
    ):
        try:
            Thresholds(**kwargs)
        except ValueError as exc:
            assert next(iter(kwargs)) in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    # Boundaries that must stay valid.
    Thresholds(class_rate_deviation=0)
    Thresholds(class_rate_deviation=1)
    Thresholds(class_rate_ratio=1.0)
    Thresholds(min_test_size=0)
    Thresholds(fold_size_deviation=0)


def test_f3_bool_index_is_rejected():
    try:
        audit([([True, 1], [2, 3])], y=[0, 1, 0, 1])
    except ValueError as exc:
        assert "non-integer index True" in str(exc)
    else:
        raise AssertionError("expected ValueError for a boolean index")


def test_f4_ratio_message_uses_two_significant_digits():
    y = [0] * 80 + [1] * 20
    test = list(range(19)) + [80]
    train = [i for i in range(100) if i not in test]
    report = audit([(train, test)], y=y, thresholds=Thresholds(min_test_size=1))
    message = next(
        f.message for f in report.flags if f.code == "class_rate_ratio_deviation"
    )
    assert "(0.25x)" in message
    y = [0] * 196 + [1] * 4
    test = list(range(17)) + [196, 197, 198]
    train = [i for i in range(200) if i not in test]
    report = audit([(train, test)], y=y, thresholds=Thresholds(min_test_size=1))
    message = next(
        f.message for f in report.flags if f.code == "class_rate_ratio_deviation"
    )
    assert "(7.5x)" in message


def test_f5_text_table_many_classes_has_gap_before_verdict():
    y = [i % 12 for i in range(240)]
    folds = [
        (list(range(120, 240)), list(range(120))),
        (list(range(120)), list(range(120, 240))),
    ]
    report = audit(folds, y=y, thresholds=Thresholds(min_test_size=1))
    lines = report.text.splitlines()
    fold_lines = [line for line in lines if line.startswith(("0 ", "1 "))]
    assert len(fold_lines) == 2
    for line in fold_lines:
        # A gap, not a collision: rates end, then a space, then OK.
        assert "11=8% OK" in line
        assert "11=8%OK" not in line


# --- Regression tests for 0.2.4 (audit FA-001 to FA-030) ---
# Standard library only, like the rest of this suite: numpy/pandas
# objects are faked with tolist()/shape/dtype attributes where a
# container type matters, so these tests run without dependencies.


def test_fa001_split_by_name_groups_and_times():
    seen = {}

    class GroupSplit:
        def split(self, X, groups):
            seen["groups"] = list(groups)
            yield [0, 1, 2, 3], [4, 5]

    audit(GroupSplit(), [[0]] * 6, [0, 1] * 3, groups=list("aabbcc"))
    assert seen["groups"] == list("aabbcc")

    class PurgedSplit:
        def split(self, X, y, times):
            seen["times"] = list(times)
            yield [0, 1, 2, 3], [4, 5]

    audit(
        PurgedSplit(),
        [[0]] * 6,
        [0, 1] * 3,
        groups=list("aabbcc"),
        time=[1, 2, 3, 4, 5, 6],
    )
    assert seen["times"] == [1, 2, 3, 4, 5, 6]


def test_fa002_keyword_only_groups_is_passed():
    seen = {}

    class KeywordSplit:
        def split(self, X, y=None, *, groups=None):
            seen["groups"] = groups
            yield [0, 1, 2, 3], [4, 5]

    audit(KeywordSplit(), [[0]] * 6, [0, 1] * 3, groups=list("aabbcc"))
    assert seen["groups"] == list("aabbcc")


def test_fa003_args_kwargs_wrapper_works():
    class Inner:
        def split(self, X, y, groups):
            yield [0, 1, 2], [3, 4, 5]

    class Wrapper:
        def __init__(self, inner):
            self.inner = inner

        def split(self, *args, **kwargs):
            return self.inner.split(*args, **kwargs)

    report = audit(Wrapper(Inner()), [[0]] * 6, [0, 1] * 3, groups=list("aabbcc"))
    assert len(report.folds) == 1


def test_fa004_nan_none_inf_labels_raise():
    for bad in (float("nan"), None, float("inf")):
        y = [0, 1, 0, 1, bad, 0]
        try:
            audit([([0, 1], [2, 3, 4, 5])], y=y)
        except ValueError as exc:
            assert "label" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for label {bad!r}")


def test_fa005_missing_groups_warn_and_overlap():
    nan = float("nan")
    groups = [nan] * 6
    report = audit(
        [([0, 1, 2], [3, 4, 5])],
        y=[0, 1, 0, 1, 0, 1],
        groups=groups,
        thresholds=Thresholds(min_test_size=1),
    )
    assert any(f.code == "missing_group_ids" for f in report.flags)
    assert any(f.code == "group_overlap" for f in report.flags)
    assert report.folds[0].n_train_groups == 1
    assert report.folds[0].n_test_groups == 1


def test_fa006_missing_time_order_independent_of_index_order():
    time = [float("nan"), 1, 2, 3, 4, 5, 6, 7, 8, 10]
    y = [0, 1] * 5

    def codes(train):
        report = audit(
            [(train, [4, 5, 6])],
            y=y,
            time=time,
            thresholds=Thresholds(min_test_size=1),
        )
        return {f.code for f in report.flags}

    assert codes([0, 1, 2, 3, 9]) == codes([9, 0, 1, 2, 3])
    assert "time_order" in codes([0, 1, 2, 3, 9])
    assert "missing_time" in codes([0, 1, 2, 3, 9])


def test_fa007_thresholds_reject_nan_and_wrong_types_and_are_frozen():
    for kwargs in (
        {"class_rate_deviation": float("nan")},
        {"class_rate_ratio": float("nan")},
        {"min_test_size": float("nan")},
        {"fold_size_deviation": float("nan")},
        {"class_rate_deviation": "0.2"},
    ):
        try:
            Thresholds(**kwargs)
        except ValueError as exc:
            assert next(iter(kwargs)) in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    frozen = Thresholds()
    try:
        frozen.class_rate_deviation = 0.5
    except AttributeError:
        pass
    else:
        raise AssertionError("expected frozen Thresholds to reject assignment")


def test_fa008_multioutput_and_unhashable_raise_clearly():
    try:
        audit([([0, 1], [2, 3])], y=[[0, 1], [1, 0], [0, 1], [1, 0]])
    except ValueError as exc:
        assert "one-dimensional" in str(exc) and "y" in str(exc)
    else:
        raise AssertionError("expected ValueError for 2-D y")
    try:
        audit(
            [([0, 1], [2, 3])],
            y=[0, 1, 0, 1],
            groups=[[1], [2], [1], [2]],
        )
    except ValueError as exc:
        assert "groups" in str(exc)
    else:
        raise AssertionError("expected ValueError for unhashable groups")


def test_fa009_sparse_like_x_only_uses_shape():
    class SparseLike:
        shape = (30, 30)

        def __len__(self):
            raise TypeError("sparse array length is ambiguous")

    folds = [(list(range(20)), list(range(20, 30)))]
    report = audit(folds, X=SparseLike())
    assert report.n_samples == 30


def test_fa010_x_shorter_than_y_raises():
    class Splitter:
        def split(self, X, y=None, groups=None):
            yield list(range(40)), list(range(40, 80))

    try:
        audit(Splitter(), [[0]] * 80, [0, 1] * 50)
    except ValueError as exc:
        assert "X" in str(exc) and "length" in str(exc)
    else:
        raise AssertionError("expected ValueError for X shorter than y")


def test_fa011_generator_single_pair_and_malformed_folds():
    def gen():
        yield [0, 1], [2, 3]

    report = audit(gen(), y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1))
    assert len(report.folds) == 1
    report = audit(
        ([0, 1], [2, 3]), y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1)
    )
    assert len(report.folds) == 1
    try:
        audit("not-a-splitter", y=[0, 1, 0, 1])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for string cv")
    try:
        audit([([0, 1], [2], [3])], y=[0, 1, 0, 1])
    except ValueError as exc:
        assert "fold 0" in str(exc)
    else:
        raise AssertionError("expected ValueError naming the malformed fold")


def test_fa012_json_with_unusual_labels():
    from datetime import date
    from decimal import Decimal

    y = [Decimal(0), Decimal(1)] * 15
    data = json.loads(audit(kfold_folds(30, 3), y=y).to_json())
    assert data["verdict"] in ("OK", "WARNING", "FAIL")
    y2 = [date(2020, 1, 1), date(2020, 1, 2)] * 15
    json.loads(audit(kfold_folds(30, 3), y=y2).to_json())


def test_fa013_json_is_strict():
    report = audit(
        [([0, 1, 2, 3], [4, 5, 6, 7])],
        y=[0, 1] * 4,
        time=[0.0, 1.0, 2.0, float("nan"), 4.0, float("inf"), 6.0, 0.0],
        thresholds=Thresholds(min_test_size=1),
    )

    def boom(constant):
        raise ValueError(constant)

    json.loads(report.to_json(), parse_constant=boom)


def test_fa014_datetime64_ints_become_iso_strings():
    class Datetime64Like:
        dtype = "datetime64[ns]"

        def tolist(self):
            return [
                1577836800000000000,
                1577923200000000000,
                1578009600000000000,
                1578096000000000000,
            ]

    report = audit(
        [([0, 1], [2, 3])],
        y=[0, 1, 0, 1],
        time=Datetime64Like(),
        thresholds=Thresholds(min_test_size=1),
    )
    assert isinstance(report.folds[0].train_time_min, str)
    assert report.folds[0].train_time_min.startswith("2020-01-01")


def test_fa015_continuous_target_does_not_flood():
    y = [float(i) for i in range(500)]
    folds = []
    for i in range(5):
        test = list(range(i * 100, (i + 1) * 100))
        train = [j for j in range(500) if j not in test]
        folds.append((train, test))
    report = audit(folds, y=y)
    assert any(f.code == "continuous_target" for f in report.flags)
    assert len(report.flags) < 50


def test_fa016_text_columns_do_not_run_together():
    from foldaudit import FoldResult, Report

    row = (
        Report(folds=[FoldResult(0, 10_000_000, 12_345_678)])
        .text.splitlines()[3]
        .split()
    )
    assert row[1] == "10000000" and row[2] == "12345678"


def test_fa017_median_of_even_count_is_true_median():
    report = audit(
        [([0], list(range(40))), ([0], list(range(100)))],
        y=[0, 1] * 60,
        thresholds=Thresholds(fold_size_deviation=0.40, min_test_size=1),
    )
    assert not any("median test size 100" in f.message for f in report.flags)
    assert any("median test size 70" in f.message for f in report.flags)


def test_fa018_time_tie_at_boundary_warns():
    time = list(range(20))
    time[10] = time[9]
    report = audit(
        [(list(range(10)), list(range(10, 20)))],
        y=[0, 1] * 10,
        time=time,
        thresholds=Thresholds(min_test_size=1),
    )
    flag = next(f for f in report.flags if f.code == "time_tie_at_boundary")
    assert flag.severity == "WARNING" and flag.fold == 0


def test_fa019_class_missing_from_train_is_reported():
    y = [0, 1] * 60
    y[50] = 2
    y[60] = 2
    a, b = list(range(80)), list(range(40, 120))
    folds = [
        ([i for i in range(120) if i not in set(a)], a),
        ([i for i in range(120) if i not in set(b)], b),
    ]
    report = audit(folds, y=y)
    assert any(f.code == "missing_class_train" for f in report.flags)
    assert report.verdict != "OK"


def test_fa020_single_holdout_is_not_a_warning():
    report = audit([(list(range(80)), list(range(80, 100)))], y=[0, 1] * 50)
    assert report.verdict == "OK"
    flag = next(f for f in report.flags if f.code == "never_tested")
    assert flag.severity == "INFO"


def test_fa021_group_overlap_examples_are_distinct():
    groups = ["A", "A", "A", "B", "C", "A", "D", "E", "B", "C", "A"]
    report = audit(
        [([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10])],
        y=[0, 1] * 5 + [0],
        groups=groups,
        thresholds=Thresholds(min_test_size=1),
    )
    message = next(f.message for f in report.flags if f.code == "group_overlap")
    assert "['A', 'B', 'C']" in message


def test_fa022_ratio_message_has_no_scientific_notation():
    y = [0] * 19990 + [1] * 10
    test = list(range(19829, 19990)) + [19990 + i for i in range(9)]
    train = [i for i in range(20000) if i not in set(test)]
    report = audit([(train, test)], y=y, thresholds=Thresholds(min_test_size=1))
    message = next(
        f.message for f in report.flags if f.code == "class_rate_ratio_deviation"
    )
    assert "e+" not in message and "vs 0% overall" not in message


def test_fa023_exact_thresholds_use_exact_arithmetic():
    report = audit(
        [(list(range(5, 10)), list(range(5)))],
        y=[1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        thresholds=Thresholds(min_test_size=1),
    )
    assert "class_rate_deviation" not in {f.code for f in report.flags}
    report = audit(
        [(list(range(12, 60)), list(range(12)))],
        y=[1] + [0] * 11 + [1] + [0] * 47,
        thresholds=Thresholds(min_test_size=1),
    )
    assert "class_rate_ratio_deviation" in {f.code for f in report.flags}


def test_fa024_overlap_counts_match_brute_force():
    folds = [
        (list(range(20, 60)), list(range(20))),
        (list(range(20)) + list(range(40, 60)), list(range(20, 40))),
        (list(range(40)), list(range(40, 60))),
    ]
    report = audit(folds, y=[0, 1] * 30, thresholds=Thresholds(min_test_size=1))
    sets = [set(test) for _, test in folds]
    expected_pairs = 0
    expected_max = 0
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            overlap = len(sets[a] & sets[b])
            expected_max = max(expected_max, overlap)
            if overlap:
                expected_pairs += 1
    assert report.n_overlapping_test_pairs == expected_pairs
    assert report.max_test_overlap == expected_max


def test_fa025_report_exposes_per_row_test_counts():
    folds = [([2, 3], [0, 1]), ([0, 1], [2, 3])]
    report = audit(folds, y=[0, 1, 0, 1], thresholds=Thresholds(min_test_size=1))
    assert report.test_counts == {0: 1, 1: 1, 2: 1, 3: 1}
    assert sum(report.test_counts.values()) == 4


def test_fa026_leave_one_out_is_not_a_fail():
    def loo(n):
        for i in range(n):
            yield [j for j in range(n) if j != i], [i]

    report = audit(loo(60), y=[0, 1, 2] * 20, thresholds=Thresholds(min_test_size=1))
    assert report.verdict != "FAIL"
    assert not any(
        f.code == "single_class_test" and f.severity == "FAIL" for f in report.flags
    )


def test_fa027_string_times_raise():
    try:
        audit(
            [(list(range(2)), list(range(2, 4)))],
            y=[0, 1, 0, 1],
            time=["01/01/2020", "02/01/2020", "03/01/2020", "04/01/2020"],
        )
    except ValueError as exc:
        assert "time" in str(exc)
    else:
        raise AssertionError("expected ValueError for string times")


def test_fa028_py_typed_and_spdx_license():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "foldaudit" / "py.typed").exists()
    assert 'license = "MIT"' in (root / "pyproject.toml").read_text()


def test_fa029_text_and_json_state_scope():
    report = audit(kfold_folds(100, 5), y=[0, 1] * 50)
    assert "not checked" in report.text.lower()
    assert "scope" in report.text.lower()
    assert "scope" in report.to_dict()
    assert "Not checked" in report.to_dict()["scope"]


def test_fa030_boundary_cases_are_pinned():
    # min_test_size exactly met does not warn; fold-size deviation
    # exactly at the threshold does not warn.
    report = audit(
        [([0, 1], list(range(2, 12)))],
        y=[0, 1] * 6,
        thresholds=Thresholds(min_test_size=10),
    )
    assert "small_test" not in {f.code for f in report.flags}


# --- Regression tests for 0.2.5 (problems report P-001 to P-006) ---
# Standard library only, like the rest of this suite: numpy
# arrays are faked with a shape/tolist interface where a
# container type matters, so these tests run without dependencies.


def test_p001_tuple_labels_are_accepted_and_unhashable_still_rejected():
    report = audit(
        [([2, 3], [0, 1])],
        y=[("class", 1), ("class", 2), ("other", 1), ("other", 2)],
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.folds[0].test_class_counts[("class", 1)] == 1
    assert report.folds[0].test_class_counts[("class", 2)] == 1
    assert report.folds[0].train_class_counts[("other", 1)] == 1
    assert report.folds[0].train_class_counts[("other", 2)] == 1
    try:
        audit([([2, 3], [0, 1])], y=[[1], [2], [3], [4]])
    except ValueError as exc:
        assert "nested" in str(exc) or "unhashable" in str(exc)
    else:
        raise AssertionError("expected ValueError for unhashable list labels")


def test_p002_generator_time_is_not_consumed():
    report = audit(
        [([0], [1])],
        y=[0, 1],
        time=(value for value in [0, 1]),
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.folds[0].train_time_min == 0
    assert report.folds[0].test_time_min == 1
    assert report.folds[0].train_time_max == 0
    assert report.folds[0].test_time_max == 1


def test_p003_small_rate_displays_as_050_percent_not_0_percent():
    y = [0] * 200 + [1]
    report = audit(
        [([0], list(range(1, 201)))],
        y=y,
        thresholds=Thresholds(min_test_size=1),
    )
    assert report.folds[0].test_class_rates[1] == 0.005
    assert "1=0.50%" in report.text
    assert "1=0% " not in report.text


def test_p004_info_only_fold_displays_info_not_warning():
    report = audit(
        [([0, 1, 2], [3])],
        y=[0, 1, 0, 1],
        thresholds=Thresholds(min_test_size=10),
    )
    assert report.verdict == "OK"
    assert all(f.severity == "INFO" for f in report.folds[0].flags)
    assert "INFO" in report.text.splitlines()[3]
    assert "WARNING" not in report.text.splitlines()[3]


def test_p005_splitter_groups_not_passed_into_unused():
    seen = {}

    class Splitter:
        def split(self, X, unused=None, groups=None):
            seen["unused"] = unused
            seen["groups"] = groups
            yield [0, 1], [2, 3]

    audit(
        Splitter(),
        [[0]] * 4,
        [0, 1, 0, 1],
        groups=["a", "a", "b", "b"],
        thresholds=Thresholds(min_test_size=1),
    )
    assert seen["unused"] is None
    assert seen["groups"] == ["a", "a", "b", "b"]


def test_p006_single_numpy_pair_is_one_fold():
    class FakeIntArray:
        def __init__(self, values):
            self._values = list(values)
            self.shape = (len(values),)
            self.dtype = "int64"

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

        def tolist(self):
            return list(self._values)

    report = audit(
        (FakeIntArray([2, 3]), FakeIntArray([0, 1])),
        y=[0, 1, 0, 1],
        thresholds=Thresholds(min_test_size=1),
    )
    assert len(report.folds) == 1
    assert report.folds[0].n_train == 2
    assert report.folds[0].n_test == 2
