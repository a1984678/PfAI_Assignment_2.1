"""Generate ``sensor_stream.csv``, the synthetic stream used by ``stream_demo.ipynb``.

12,000 readings from 4 sensors (scaled to [-1, 1]) with a label "ok" or "fault":

- Rows 0-5,999: fault when the reading leaves a circle, x0^2 + x1^2 >= 0.5.
- Rows 6,000-: the rule changes (concept drift, e.g. after maintenance): fault when
  x0 + x2 <= 0.
- 5% of labels flipped (noise) and 3% of readings missing (blank cells).

Run from the project root: ``python demo/data/make_sensor_stream.py`` (seeded, so the file
is always the same).
"""

from pathlib import Path

import numpy as np

N, DRIFT_ROW, SEED = 12000, 6000, 0
PATH = Path(__file__).with_name("sensor_stream.csv")


def main():
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1, 1, size=(N, 4))
    ok = X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5
    ok[DRIFT_ROW:] = X[DRIFT_ROW:, 0] + X[DRIFT_ROW:, 2] > 0
    ok ^= rng.random(N) < 0.05                                   # label noise
    cells = np.char.mod("%.4f", X)
    cells[rng.random(X.shape) < 0.03] = ""                       # missing readings
    rows = [",".join(row) for row in np.column_stack([cells, np.where(ok, "ok", "fault")])]
    PATH.write_text("sensor_1,sensor_2,sensor_3,sensor_4,status\n" + "\n".join(rows) + "\n")
    print(f"Wrote {N} rows to {PATH}")


if __name__ == "__main__":
    main()
