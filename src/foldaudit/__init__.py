"""foldaudit: check that cross-validation folds are fair and usable
before you trust the score.

The public API is intentionally small:

    from foldaudit import audit

    report = audit(cv, X, y, groups=groups, time=dates)
    print(report.text)
    print(report.to_json())

``audit`` accepts either a splitter object with a ``.split()`` method
(like scikit-learn's KFold, StratifiedKFold, GroupKFold or
TimeSeriesSplit) or a plain list of ``(train_idx, test_idx)`` pairs,
so it does not depend on scikit-learn at all. It trains no model,
makes no plots, and needs no GPU or paid API.
"""

from .core import Thresholds, audit
from .report import Flag, FoldResult, Report

__all__ = ["Flag", "FoldResult", "Report", "Thresholds", "audit"]
__version__ = "0.2.6"
