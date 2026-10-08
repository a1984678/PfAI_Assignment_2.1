"""
benchmarking.py

Micro-benchmark harness so every benchmark in the project measures the same way.

- time_it(func, *args, repeats=5, warmup=1) — median run time in seconds
- compare(candidates, *args, repeats=5) — time several functions on the same input
- report(results, baseline=None) — text table with speed-up against a baseline
- environment_info() — Python / NumPy / OS / CPU details, so timings can be reproduced
- compare_loop_vs_vectorised(loop_func, vec_func, *args) — loop vs NumPy timing, checked equal
"""

import platform
import sys
import time

import numpy as np


def time_it(func, *args, repeats=5, warmup=1, **kwargs):
    """
    Median wall-clock time of func(*args, **kwargs).

    Parameters:
    ----------
    - func: callable
    - args, kwargs: passed to func on every call
    - repeats: int >= 1, number of timed runs
    - warmup: int >= 0, untimed runs first (fills caches, avoids first-call cost)

    Returns:
    -------
    - float, median seconds over the timed runs

    Raises:
    ------
    - ValueError: if func is not callable, repeats < 1 or warmup < 0

    Complexity:
    ----------
    - Time complexity: (warmup + repeats) calls of func
    - Space complexity: O(repeats)

    Note:
    ----
    - The median is used instead of the mean so one slow run (the OS doing something
    else) does not distort the result.
    """
    if not callable(func):
        raise ValueError("func must be callable.")
    if repeats < 1 or int(repeats) != repeats:
        raise ValueError("repeats must be an integer >= 1. Got: {!r}".format(repeats))
    if warmup < 0 or int(warmup) != warmup:
        raise ValueError("warmup must be an integer >= 0. Got: {!r}".format(warmup))
    for _ in range(int(warmup)):
        func(*args, **kwargs)
    times = np.empty(int(repeats))
    for i in range(int(repeats)):
        start = time.perf_counter()
        func(*args, **kwargs)
        times[i] = time.perf_counter() - start
    return float(np.median(times))


def compare(candidates, *args, repeats=5, warmup=1):
    """
    Time several functions on the same input.

    Parameters:
    ----------
    - candidates: dict {label: callable}
    - args: passed to every callable
    - repeats, warmup: see time_it

    Returns:
    -------
    - dict {label: median seconds}, in the same order as candidates

    Raises:
    ------
    - ValueError: if candidates is empty or not a dict

    Complexity:
    ----------
    - Time complexity: the sum of the candidates' costs
    - Space complexity: O(number of candidates)
    """
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("candidates must be a non-empty dict of {label: callable}.")
    return {label: time_it(func, *args, repeats=repeats, warmup=warmup)
            for label, func in candidates.items()}


def report(results, baseline=None):
    """
    Format timings as a plain-text table with speed-up against a baseline.

    Parameters:
    ----------
    - results: dict {label: seconds}, e.g. the output of compare
    - baseline: label to compare against, or None (default) to use the first entry

    Returns:
    -------
    - str, one line per label: time in milliseconds and "x faster" / "x slower"

    Raises:
    ------
    - ValueError: if results is empty, a time is not positive, or baseline is unknown

    Complexity:
    ----------
    - Time and space complexity: O(number of labels)
    """
    if not isinstance(results, dict) or not results:
        raise ValueError("results must be a non-empty dict of {label: seconds}.")
    if any(t <= 0 for t in results.values()):
        raise ValueError("All times must be positive.")
    if baseline is None:
        baseline = next(iter(results))
    if baseline not in results:
        raise ValueError("baseline {!r} is not in results.".format(baseline))
    base = results[baseline]
    width = max(len(str(k)) for k in results)
    lines = []
    for label, t in results.items():
        if label == baseline:
            note = "(baseline)"
        elif t <= base:
            note = "{:.1f}x faster".format(base / t)
        else:
            note = "{:.1f}x slower".format(t / base)
        lines.append("{:<{w}}  {:>10.3f} ms  {}".format(str(label), t * 1e3, note, w=width))
    return "\n".join(lines)


def environment_info():
    """
    Describe the machine and software the timings were taken on.

    Parameters:
    ----------
    - None

    Returns:
    -------
    - dict with keys 'python', 'numpy', 'platform', 'processor' (all str)

    Complexity:
    ----------
    - Time and space complexity: O(1)

    Note:
    ----
    - Print this next to any benchmark table: timings from another machine or
    NumPy version are not directly comparable.
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def compare_loop_vs_vectorised(loop_func, vec_func, *args, repeats=5, warmup=1,
                               rtol=1e-9, atol=1e-12):
    """
    Time a plain-Python-loop version against a vectorised NumPy version.

    Parameters:
    ----------
    - loop_func: callable, the loop implementation
    - vec_func: callable, the NumPy implementation (same arguments and result)
    - args: passed to both functions
    - repeats, warmup: see time_it
    - rtol, atol: tolerance used to check the two results agree

    Returns:
    -------
    - dict with 'loop' and 'vectorised' (median seconds) and 'speedup' (loop / vectorised)

    Raises:
    ------
    - ValueError: if the two functions give different results (a fast wrong answer
    is not a speed-up), or repeats/warmup are invalid

    Complexity:
    ----------
    - Time complexity: one correctness run of each function plus the timed runs
    - Space complexity: O(repeats)
    """
    expected = np.asarray(loop_func(*args))
    got = np.asarray(vec_func(*args))
    if expected.shape != got.shape or not np.allclose(expected, got, rtol=rtol, atol=atol,
                                                      equal_nan=True):
        raise ValueError("loop_func and vec_func return different results.")
    loop_t = time_it(loop_func, *args, repeats=repeats, warmup=warmup)
    vec_t = time_it(vec_func, *args, repeats=repeats, warmup=warmup)
    return {"loop": loop_t, "vectorised": vec_t,
            "speedup": loop_t / vec_t if vec_t > 0 else float("inf")}
