"""Purged walk-forward cross-validation with embargo.

Because labels look `horizon` bars into the future, a naive split lets the
training set peek at information that overlaps the test labels. We therefore:

* walk forward (expanding train window, always earlier than the test fold), and
* purge the `horizon + embargo` bars immediately before each test fold, plus an
  embargo of bars after it, from the training indices.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def purged_walk_forward(n: int, n_splits: int, horizon: int,
                        embargo: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional arrays for each fold."""
    indices = np.arange(n)
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        raise ValueError("Not enough samples for the requested number of splits.")

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits else n
        test_idx = indices[test_start:test_end]

        # Train only on the past, purging the horizon+embargo gap before the fold.
        train_end = max(0, test_start - horizon - embargo)
        train_idx = indices[:train_end]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx
