"""Tests for ``numcompute_stream.io``.

Cover CSV loading and splitting a dataset into chunks for streaming,
including edge cases (e.g., missing values, malformed input, uneven chunks).
"""

import inspect

import numpy as np
import pytest

from numcompute_stream.io import iter_chunks, load_csv


def test_iter_chunks_is_lazy_and_keeps_short_last_chunk():
    """10 rows in chunks of 4 -> sizes 4, 4, 2, produced one at a time."""
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10)
    gen = iter_chunks(X, y, chunk_size=4)
    assert inspect.isgenerator(gen)

    chunks = list(gen)
    assert [len(Xc) for Xc, _ in chunks] == [4, 4, 2]
    # Order preserved: stitching the chunks back together gives the original data.
    np.testing.assert_array_equal(np.vstack([Xc for Xc, _ in chunks]), X)
    np.testing.assert_array_equal(np.concatenate([yc for _, yc in chunks]), y)


def test_iter_chunks_without_y_yields_arrays():
    X = np.ones((5, 3))
    chunks = list(iter_chunks(X, chunk_size=2))
    assert all(isinstance(c, np.ndarray) for c in chunks)
    assert [c.shape for c in chunks] == [(2, 3), (2, 3), (1, 3)]


def test_iter_chunks_chunk_size_larger_than_data_gives_one_chunk():
    X = np.ones((3, 2))
    chunks = list(iter_chunks(X, chunk_size=100))
    assert len(chunks) == 1 and chunks[0].shape == (3, 2)


def test_iter_chunks_shuffle_is_seeded_and_keeps_rows_aligned():
    X = np.arange(12.0).reshape(6, 2)
    y = X[:, 0] * 10  # label encodes its row, so misalignment is detectable
    first = list(iter_chunks(X, y, chunk_size=4, shuffle=True, seed=1))
    again = list(iter_chunks(X, y, chunk_size=4, shuffle=True, seed=1))

    X_all = np.vstack([Xc for Xc, _ in first])
    y_all = np.concatenate([yc for _, yc in first])
    assert not np.array_equal(X_all, X)                       # order changed
    np.testing.assert_array_equal(np.sort(X_all, axis=0), X)  # same rows
    np.testing.assert_array_equal(y_all, X_all[:, 0] * 10)    # still aligned
    for (a, _), (b, _) in zip(first, again):
        np.testing.assert_array_equal(a, b)                   # reproducible


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, None])
def test_iter_chunks_invalid_chunk_size_raises(bad):
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        next(iter_chunks(np.ones((4, 2)), chunk_size=bad))


@pytest.mark.parametrize("X, y, match", [
    (np.ones((4, 2)), np.ones(3), "same number of samples"),
    (np.empty((0, 0)), None, "no rows"),
])
def test_iter_chunks_bad_data_raises(X, y, match):
    with pytest.raises(ValueError, match=match):
        next(iter_chunks(X, y, chunk_size=2))


def test_load_csv_then_iter_chunks_keeps_missing_values(tmp_path):
    """Missing cells load as NaN and pass through chunking untouched (no rows dropped)."""
    f = tmp_path / "data.csv"
    f.write_text("1,2\n,4\n5,\n7,8\n9,10\n")
    data = load_csv(f)

    chunks = list(iter_chunks(data, chunk_size=2))
    assert [len(c) for c in chunks] == [2, 2, 1]
    stitched = np.vstack(chunks)
    assert np.isnan(stitched).sum() == 2
    np.testing.assert_array_equal(np.isnan(stitched), np.isnan(data))
