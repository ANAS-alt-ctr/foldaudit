# foldaudit

**Name:** foldaudit
**One sentence:** It checks that your cross-validation folds are fair
and usable before you trust the score.

## The problem

When you run cross-validation, you get scores. You do not get any
information about the folds those scores came from. A fold can be
broken in ways that make its score meaningless: it contains only one
class, it is far too small, its class balance is nothing like your
full dataset, or, in a time-series split, training data comes from
after the test data. Scikit-learn's own issue tracker documents
degenerate folds of exactly this kind, for example StratifiedGroupKFold
producing empty or unusable splits, while `cross_val_score` only
returns scores and never inspects the folds themselves. Almost nobody
inspects every fold's indices by hand. foldaudit does that inspection
for you, before you train anything.

## What it is

A small, pure-Python library, version 0.2.5. No third-party
dependencies, no model training, no plots, no GPU, no paid APIs.
It works with scikit-learn splitters and without them: give it a
splitter with a `.split()` method, or a plain list of
`(train_idx, test_idx)` pairs.

## Installation (once published)

```bash
pip install foldaudit
```


Local use from this folder, without installing:

```bash
PYTHONPATH=src python -c "from foldaudit import audit; print('ok')"
```

## Usage

```python
from foldaudit import audit

# With a splitter (KFold, StratifiedKFold, GroupKFold,
# TimeSeriesSplit, or any object with .split())
report = audit(cv, X, y, groups=groups, time=dates)

# Without scikit-learn: plain index pairs
report = audit([(train_idx, test_idx), ...], y=y)

print(report.text)      # pasteable plain-text report
print(report.to_json()) # JSON, for saving or for CI
report.raise_if_fail()  # raises ValueError on a FAIL verdict
# report.raise_if_fail(fail_on_warning=True)  # also fails on WARNING
```

## What it checks (version 0.2.5)

1. **Per-fold table.** Train/test sizes, class counts and rates,
   group counts, and time min/max per fold.
2. **Validity flags.** Empty train/test, single-class train/test,
   a class missing from a test fold, a test fold smaller than
   the minimum size, a row appearing in both train and test of the
   same fold (FAIL, since 0.2.1), a duplicated index inside
   one side of a fold (WARNING, since 0.2.1), and a group
   appearing on both sides of a fold (FAIL, since 0.2.2).
3. **Stratification and balance.** A fold's class rate compared
   with the overall rate, and a fold's test size compared with the
   median test size. Thresholds are configurable; the report always
   quotes the actual numbers.
4. **Coverage.** How many times each row is tested, rows never
   tested, and overlapping test folds (reported, not failed, since
   repeated or shuffled splitters overlap by design).
5. **Report export.** Plain text and JSON, with an overall verdict
   of OK, WARNING, or FAIL, and an optional fail-on-warning hook
   for CI.

Additional checks: time order (train times must precede test
times, otherwise FAIL), group counts per fold, and group
overlap (a group on both sides of a fold is group leakage, FAIL,
since 0.2.2). Version 0.2.0
adds the rare-class ratio check: a fold's class rate more than 2x
(or less than 1/2x) the overall rate warns, even when the absolute
gap is under 20 points, provided the gap is at least 5 points and
the class appears in the test fold.

## Thresholds (defaults, all configurable)

| Setting | Default | Meaning |
| --- | --- | --- |
| `class_rate_deviation` | 0.20 | Warn when a fold's class rate differs from the overall rate by more than 20 percentage points. |
| `class_rate_ratio` | 2.0 | Warn when a fold's class rate is more than this many times the overall rate, or less than 1/this of it (rare-class check; 5-point absolute floor applies). |
| `min_test_size` | 10 | Warn when a test fold has fewer rows than this. |
| `fold_size_deviation` | 0.50 | Warn when a test fold's size differs from the median test size by more than 50%. |

```python
from foldaudit import Thresholds, audit

report = audit(cv, X, y, thresholds=Thresholds(
    class_rate_deviation=0.10,
    min_test_size=20,
))
```

These are documented heuristics, not universal truths. Change them
to fit your problem.

## Example output

A degenerate split, where each test fold contains only one class:

```
foldaudit verdict: FAIL

Fold  Train   Test    Test class rates   Verdict
0     50      50      0=100%             FAIL
1     50      50      1=100%             FAIL

Flags:
  [FAIL] fold 0: Train set has a single class: 1.
  [FAIL] fold 0: Test set has a single class: 0.
  ...

Since 0.2.0 the missing-class and rate-deviation warnings for a
single-class fold are suppressed: they were the same fact reported
three times. One bad fold now reports two flags, not five.
```

A healthy balanced split reports `foldaudit verdict: OK` with no flags.

## What it deliberately will NOT do

- No model training or scores.
- No new splitters, and no purged or embargoed splits.
- No semantic train/test leakage detection beyond group labels
  (duplicated rows, feature or target leakage — that is a
  separate tool's job). A literal index appearing in both train
  and test of one fold IS flagged, since 0.2.1, and a group label
  appearing on both sides IS flagged, since 0.2.2: both are basic
  fold validity, not semantic leakage analysis.
- No plots, drift detection, slice or fairness metrics, and no
  experiment tracking.
- No automatic "fixing" of folds. It tells you what is wrong so
  you can choose a better splitter yourself.

## Project layout

```
fold-audit/
  pyproject.toml
  README.md
  src/
    foldaudit/
      __init__.py   # public API: audit, Thresholds, Report, ...
      core.py       # audit logic, flags, coverage, time, groups
      report.py     # FoldResult, Flag, Report, text/JSON rendering
  tests/
    test_foldaudit.py   # 72 tests, standard library only
```

## Tests and demo

```bash
python -m pytest tests -q
PYTHONPATH=src python examples/demo.py
```

72 tests pass. They use fake splitters made of plain index lists,
so they need no scikit-learn, pandas, or any other dependency.
Ruff check and ruff format check are clean. A wheel builds as
`foldaudit-0.2.5-py3-none-any.whl` and installs and imports as
version 0.2.5.

## What changed in 0.2.0 (fixes for the 2 November 2026 problems report)

All five reported problems are fixed, with a regression test each:

1. **Rare-class blind spot:** new `class_rate_ratio_deviation` check
   (see above). The report's own example — 2% overall, 15% in a fold —
   now warns at 7.5x.
2. **Out-of-range indices:** a bad index now raises
   `ValueError: fold 0 train set contains index 3, but only 3 samples
   were provided`, instead of a bare `IndexError` from inside the
   counting loop. Negative and non-integer indices are rejected too.
3. **Report noise:** a single-class test fold now reports only the
   single-class FAILs. The missing-class and rate-deviation warnings
   for that fold were consequences of the same fact, so they are
   suppressed. The degenerate demo fold went from 5 flags to 2.
4. **Small-fold check without labels:** `small_test` no longer lives
   inside the labels block, so auditing without `y` still warns on a
   1-row test fold. An empty test fold gets `empty_test` only.
5. **TimeSeriesSplit verdict:** with a `time` column supplied,
   `never_tested` is INFO, not WARNING, because the earliest rows
   are train-only by design. Without `time`, it stays WARNING.

## What changed in 0.2.1 (fix for the 0.2.0 problems report)

Problem 6, the only open item from that report, is fixed:

- **Train/test overlap within a fold (FAIL):** a row in both sides
  of one fold now raises `train_test_overlap`, quoting the count,
  e.g. 10 rows in the report's example. Previously that split
  reported only an unrelated `never_tested` warning.
- **Duplicate index inside one side (WARNING):** a test fold like
  `[2, 2, 3]` is flagged as `duplicate_indices` instead of being
  silently counted twice. Indices are counted as given, not
  silently deduplicated.
- **Polish:** the sdist now includes `examples/` via
  `MANIFEST.in`. No `[project.urls]` Homepage/Repository was
  added, because no GitHub repo exists yet — adding a guessed
  URL would be worse than none.

## What changed in 0.2.2 (complete fix list, F1-F8)

Only the items in that list were changed:

- **F1 group overlap (FAIL):** a group label on both sides of a
  fold now raises `group_overlap`. Counts are unchanged. No other
  leakage detection was added.
- **F2 Thresholds validation:** invalid values now raise
  `ValueError` naming the field (`class_rate_deviation` outside
  [0, 1], `class_rate_ratio` below 1.0, `min_test_size` below 0,
  `fold_size_deviation` below 0). Defaults and field names are
  unchanged.
- **F3 boolean index:** a boolean index now raises the same
  non-integer `ValueError` as 2.5. Ordinary ints still pass.
- **F4 ratio formatting:** ratios now print with two significant
  digits, so 0.25 prints as "(0.25x)" and 7.5 still as "(7.5x)".
- **F5 text table:** the rates column is now as wide as the
  longest rates string (minimum 40, plus one gap character), so a
  12-class report no longer collides with its verdict. The demo
  output is byte-identical to 0.2.1.
- **F6 README:** the PyPI-name block was removed; the dated check
  is recorded once, in Blunt status below.
- **F7 sdist:** verified to list `examples/demo.py`; the
  `MANIFEST.in` that already provided it is unchanged.
- **F8 [project.urls]:** skipped, because no GitHub repo exists
  yet. No URL was invented or guessed.


## What changed in 0.2.4 (audit FA-001 to FA-030)

All 30 issues from the 0.2.2 audit are fixed in this version, with
a regression test each (30 new tests, 66 in total). This is a fresh
build on 0.2.2; it is not 0.2.3, and no 0.2.3 code was reused.

- **FA-001/002/003 splitter routing:** splitter arguments are now
  dispatched by parameter name. A parameter named `groups` gets
  groups, `time`/`times` gets time, keyword-only parameters are
  passed by keyword, and a `*args` wrapper or an unavailable
  signature falls back to `split(X, y, groups)`. An unknown
  required parameter raises a clear error instead of receiving
  the wrong object.
- **FA-004 missing/infinite labels:** rejected with ValueError
  naming the position, as scikit-learn does. NaN labels no longer
  become one phantom class per row.
- **FA-005 missing groups:** normalised to one missing group and
  warned about as `missing_group_ids`, with the count. All-missing
  groups therefore also trigger `group_overlap`, instead of
  silently passing.
- **FA-006 missing times:** warned about as `missing_time` and
  excluded from min/max, so the time-order verdict no longer
  depends on index order.
- **FA-007 Thresholds:** frozen dataclass; NaN, infinite and
  non-numeric values are rejected naming the field. `audit()`
  also rejects a non-Thresholds thresholds object.
- **FA-008 input validation:** 2-D, mapping, string and unhashable
  y/groups/time values raise ValueError naming the argument and
  the required 1-D hashable shape.
- **FA-009/010 sample sizes:** sample count uses `shape[0]`
  first, so sparse X works, and X length is checked against
  y/groups/time and raises ValueError on mismatch.
- **FA-011 fold iterables:** any iterable of pairs is accepted,
  a single `(train, test)` pair is one fold, strings raise
  TypeError, and malformed folds raise ValueError naming the fold.
- **FA-012/013 JSON:** class tables in JSON are now records
  (`label`, `repr`, `count`/`rate`), so labels `1` and `"1"` or
  `None` and `"null"` cannot collide, numpy scalars, tuples,
  datetimes, bytes and Decimal labels cannot crash export, and
  `to_json()` uses `allow_nan=False` so no bare NaN/Infinity
  token is ever emitted. This changes `to_dict()`/`to_json()`
  class-table shape from dicts to lists; the Python
  `FoldResult` attributes are unchanged dicts.
- **FA-014 datetime64 times:** numpy datetime64 ints are
  converted to ISO strings (NaT becomes missing). Month/year
  datetime64 units are left as-is.
- **FA-015 continuous targets:** a float target with more than
  half its values unique warns as `continuous_target` and skips
  class checks, instead of flooding the report. This detection
  is float-only, as specified in the registry.
- **FA-016 text table:** every column width is computed from
  the data, so large sizes and fold numbers cannot run together.
- **FA-017 median:** fold-size imbalance now uses the true
  median (`statistics.median`), not the upper middle element.
- **FA-018 boundary ties (author decision):** equal train-max
  and test-min timestamps warn as `time_tie_at_boundary`
  (WARNING), not FAIL and not a silent pass. A tie may be one
  batch, but it may also hide leakage, so it must be visible.
- **FA-019 missing train classes:** mirrored as
  `missing_class_train`, suppressed only when the train side
  itself is single-class (the same P3 rule as for test).
- **FA-020 single hold-out:** `never_tested` is INFO, not
  WARNING, for a single fold, as it already was with time.
- **FA-021 group examples:** distinct overlapping groups are
  shown, deduplicated before slicing to three.
- **FA-022 message formatting:** rates under 1% print with
  two decimal places (never "0%"), and ratios never print in
  scientific notation.
- **FA-023 exact thresholds:** class-rate decisions use exact
  Fraction arithmetic from the decimal threshold, so exactly
  20 points does not warn and exactly 5 points at 2.5x does.
- **FA-024 performance:** folds are streamed when sample data
  is given (materialisation only remains when the sample count
  must itself be inferred from the folds), and pairwise test
  overlap uses bitmasks for up to 250,000 rows instead of
  Python set intersections. Pairwise checking is still O(k^2)
  in the number of folds, in C bit operations rather than
  Python set operations; it is not claimed to be linear.
- **FA-025 per-row test counts:** `Report.test_counts` now
  exposes the count for every row, including 0 for never-tested
  rows, matching what the README coverage section promised.
- **FA-026 pooled schemes:** 1-row test folds, and test folds
  that are exactly one group absent from train, downgrade
  `single_class_test` and `small_test` to INFO instead of FAIL.
- **FA-027 string times:** rejected with ValueError; string
  times are never compared lexicographically.
- **FA-028 packaging:** SPDX `license = "MIT"` with
  license-files, and a `py.typed` marker is shipped. The old
  `License :: OSI Approved :: MIT License` classifier was
  removed because modern setuptools (PEP 639) forbids it
  alongside an SPDX license expression; this is metadata
  only, no code change.
- **FA-029 scope:** every text report ends with a Scope
  footer stating what is checked and what is not checked,
  and `to_dict()`/`to_json()` carry the same `scope` field.
- **FA-030 test suite:** 30 new regression tests, one per
  bug, all standard-library only. The auditor's 28
  sklearn/numpy/pandas regression tests are not copied into
  the shipped suite (it must run without dependencies), but
  equivalent stdlib tests cover the same behaviours; the
  mutation score was not re-measured in this build.

Not fixed / still limited: no project is claimed
production-ready or leakage-free. Semantic leakage (duplicate
rows, feature/target leakage, preprocessing leakage) is still
out of scope by design and is now stated in every report.

## What changed in 0.2.5 (problems report P-001 to P-006)

All six problems from the 0.2.4 problems report are fixed, with a
regression test each (6 new tests, 72 in total). No new
dependencies, no API or parameter renames, no removed features.

- **P-001 tuple labels:** hashable tuple labels are now accepted
  as atomic class labels. Unhashable nested values (lists, dicts,
  sets, or tuples containing unhashable values) are still rejected.
- **P-002 generator time:** a generator `time` input is
  materialised once at the start of `audit()` and reused for
  validation and auditing, instead of being consumed by the
  string-time validation pass.
- **P-003 small rates:** the text table now uses the adaptive
  percentage formatter, so a 0.5% test class rate prints as
  "0.50%", never as "0%".
- **P-004 fold verdict:** the per-fold table verdict is derived
  from severity, FAIL > WARNING > INFO > OK, so an INFO-only
  fold (e.g. a pooled leave-one-out fold) displays INFO, not
  WARNING. The overall verdict rule is unchanged.
- **P-005 splitter routing:** positional slots already consumed
  are tracked, and recognised named arguments after a skipped
  defaulted slot are passed by keyword, so e.g.
  `split(X, unused=None, groups=None)` no longer receives
  groups in `unused`.
- **P-006 numpy pair:** a single `(train, test)` pair given as
  two 1-D numpy integer arrays is now recognised as one fold,
  like the list/tuple pair form. The shipped tests use a
  stdlib fake array with the same shape/tolist interface, so
  the suite still runs without numpy.

## Blunt status: done vs not done

Done: the core audit, the version 0.1 checks plus the 0.2.0
rare-class ratio check, the 0.2.1 overlap/duplicate checks, the
0.2.2 group-overlap check, and the 0.2.4 FA-001 to FA-030 fixes, the 0.2.5 P-001 to P-006 fixes,
text and JSON reports, configurable thresholds, 72 tests, a
runnable demo, a clean ruff check and ruff format check, and a
buildable wheel and sdist that pass twine check.
Not done: not published on PyPI, not proven to be wanted by
anyone, and no GitHub repo exists yet, so no Homepage link is
set. The PyPI name "foldaudit" was checked free on PyPI on
2 November 2026 and remains unpublished. Real scikit-learn
integration has now been tested successfully: by the auditor's
regression suite (StratifiedKFold OK, degenerate KFold FAIL,
GroupKFold counts correct), and in a real-data test on NYC
Yellow Taxi, January 2024 (first 1,500,000 rows of 2,964,624
trips; label payment_type, groups 252 pickup zones, time
pickup timestamp) — StratifiedKFold(5) OK in 8.9s, KFold(5)
with zones as groups FAIL in 12.6s (233–237 of 252 zones
leaking into train and test in every fold), GroupKFold(5)
by zone OK in 12.2s with no group or class-balance false
alarms, and KFold (no shuffle) on time-sorted data with
time FAIL in 12.2s, including a 2002-12-31 timestamp flagged
in January 2024 data. That real-data test was reported by
Anas on 2 November 2026 and was not run in this environment,
where scikit-learn is still not installed; the shipped test
suite itself remains standard-library only by design.

## Risks and honest limits

- Demand is indirect: scikit-learn issues document degenerate
  folds, but nobody was found asking for a "fold audit library"
  by name, so popularity is unproven.
- The library must earn its existence through the full checklist,
  severity levels, and report, not just a counts loop.
- Scikit-learn could one day add fold warnings itself; foldaudit
  stays splitter-agnostic to survive that.

## License

MIT.
