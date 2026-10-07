"""
run_all.py

Runs every benchmark script in this folder one after the other.

    python benchmark/run_all.py            (full sizes)
    python benchmark/run_all.py --quick    (smaller sizes, a few seconds)

Covers metrics, optim, stats, rank, sort_search, utils and benchmarking.
Results from a full run are saved in results.md and results_rank_sort_utils.md.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["bench_metrics_optim_stats.py", "bench_rank_sort_utils.py"]


def main():
    extra = sys.argv[1:]
    for name in SCRIPTS:
        print("=" * 72)
        print(name)
        print("=" * 72)
        code = subprocess.call([sys.executable, os.path.join(HERE, name)] + extra)
        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    main()
