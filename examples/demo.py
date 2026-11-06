"""Demo for foldaudit. Run with:  PYTHONPATH=src python examples/demo.py

No third-party libraries are used. The splitters below are plain
index lists, which is exactly what foldaudit accepts in place of a
scikit-learn splitter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foldaudit import audit

print("== Healthy balanced split ==")
y = [0, 1] * 50
folds = []
for i in range(5):
    test = list(range(i * 20, (i + 1) * 20))
    train = [j for j in range(100) if j not in test]
    folds.append((train, test))
print(audit(folds, y=y).text)

print()
print("== Degenerate split: each test fold has a single class ==")
y = [0] * 50 + [1] * 50
folds = [
    (list(range(50, 100)), list(range(50))),
    (list(range(50)), list(range(50, 100))),
]
print(audit(folds, y=y).text)
