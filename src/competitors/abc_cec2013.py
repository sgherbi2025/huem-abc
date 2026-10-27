import csv
import numpy as np
from cec2013 import CEC2013

DIMS = [30, 50]
N_RUNS = 30
POP = 100
MAX_ITER = 1500
LIMIT = 150
LB = -100.0
UB = 100.0


def abc(func, dim, pop=POP, max_iter=MAX_ITER, limit=LIMIT, seed=0):
    rng = np.random.default_rng(seed)
    half = pop // 2
    X = rng.uniform(LB, UB, (half, dim))
    F = np.array([func(x) for x in X])
    trial = np.zeros(half, dtype=int)
    best = F.min()
    for _ in range(max_iter):
        for i in range(half):
            k = rng.integers(half)
            while k == i:
                k = rng.integers(half)
            V = np.clip(X[i] + rng.uniform(-1.0, 1.0, dim) * (X[i] - X[k]), LB, UB)
            fv = func(V)
            if fv <= F[i]:
                X[i] = V
                F[i] = fv
                trial[i] = 0
            else:
                trial[i] += 1
        fit = np.where(F >= 0.0, 1.0 / (1.0 + F), 1.0 + np.abs(F))
        prob = fit / fit.sum()
        for _ in range(half):
            i = rng.choice(half, p=prob)
            k = rng.integers(half)
            while k == i:
                k = rng.integers(half)
            V = np.clip(X[i] + rng.uniform(-1.0, 1.0, dim) * (X[i] - X[k]), LB, UB)
            fv = func(V)
            if fv <= F[i]:
                X[i] = V
                F[i] = fv
                trial[i] = 0
            else:
                trial[i] += 1
        for i in range(half):
            if trial[i] > limit:
                X[i] = rng.uniform(LB, UB, dim)
                F[i] = func(X[i])
                trial[i] = 0
        best = min(best, F.min())
    return best


def get_function(bench, fid):
    if hasattr(bench, "get_function"):
        return bench.get_function(fid)
    return bench.funcs[fid - 1]


def run_dimension(dim):
    bench = CEC2013(dim)
    rows = []
    for fid in range(1, 29):
        vals = [abc(get_function(bench, fid), dim, seed=1000 * fid + r)
                for r in range(N_RUNS)]
        v = np.array(vals)
        rows.append([dim, "F%02d" % fid, v.min(), v.max(), np.median(v),
                     v.mean(), v.std(), N_RUNS])
        print("D=%d F%02d mean=%.6e" % (dim, fid, v.mean()), flush=True)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "Func", "Best", "Worst", "Median", "Mean", "Std", "N_runs"])
        w.writerows(rows)


if __name__ == "__main__":
    for dim in DIMS:
        write_csv("abc_ant_cec2013_D%d.csv" % dim, run_dimension(dim))
