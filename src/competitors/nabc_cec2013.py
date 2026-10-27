import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from opfunu.cec_based import cec2013

RNG_MASTER_SEED = 42
SN = 50
LIMIT = 100
NEIGH_N = 5
N_RUNS = 30
MAX_NFE_FACTOR = 10000

def fitness_transform(f):
    return np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))

def nabc(evaluate, lb, ub, D, SN, limit, N, max_nfe, rng):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    X = lb + rng.random((SN, D)) * (ub - lb)
    fit = np.empty(SN)
    for i in range(SN):
        fit[i] = evaluate(X[i])
    nfe = SN
    trial = np.zeros(SN, dtype=int)
    all_idx = np.arange(SN)

    def best_neighbor_move(i, X, fit):
        candidates = rng.choice(SN - 1, size=min(N, SN - 1), replace=False)
        candidates = np.where(candidates >= i, candidates + 1, candidates)
        nbest = candidates[np.argmin(fit[candidates])]
        j = rng.integers(0, D)
        phi = rng.uniform(-1.0, 1.0)
        V = X[i].copy()
        V[j] = X[nbest, j] + phi * (X[nbest, j] - X[i, j])
        np.clip(V, lb, ub, out=V)
        return V

    while nfe < max_nfe:
        for i in range(SN):
            if nfe >= max_nfe:
                break
            V = best_neighbor_move(i, X, fit)
            fV = evaluate(V)
            nfe += 1
            if fit[i] > fV:
                X[i] = V
                fit[i] = fV
                trial[i] = 0
            else:
                trial[i] += 1
        if nfe >= max_nfe:
            break
        fitv = fitness_transform(fit)
        p = fitv / fitv.sum()
        count = 0
        while count < SN and nfe < max_nfe:
            j = rng.choice(SN, p=p)
            V = best_neighbor_move(j, X, fit)
            fV = evaluate(V)
            nfe += 1
            if fit[j] > fV:
                X[j] = V
                fit[j] = fV
                trial[j] = 0
            else:
                trial[j] += 1
            count += 1
        if nfe >= max_nfe:
            break
        gbest_idx = np.argmin(fit)
        for i in range(SN):
            if nfe >= max_nfe:
                break
            if trial[i] > limit:
                trial[i] = 0
                others = all_idx[all_idx != i]
                a, b = rng.choice(others, size=2, replace=False)
                r = rng.random(3)
                r = r / r.sum()
                V = r[0] * X[i] + r[1] * X[gbest_idx] + r[2] * (X[a] - X[b])
                np.clip(V, lb, ub, out=V)
                fV = evaluate(V)
                nfe += 1
                X[i] = V
                fit[i] = fV
    return float(np.min(fit))

def run_single(func_num, run_idx, D):
    f_inst = getattr(cec2013, f"F{func_num}2013")(ndim=D)
    seed = RNG_MASTER_SEED * 10000 + func_num * 100 + run_idx
    rng = np.random.default_rng(seed)
    best = nabc(f_inst.evaluate, f_inst.lb, f_inst.ub, D, SN, LIMIT, NEIGH_N, MAX_NFE_FACTOR * D, rng)
    error = best - f_inst.f_global
    return max(error, 0.0)

def run_dimension(D, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in tqdm(range(1, 29), desc=f"CEC2013 D={D}"):
        f_inst = getattr(cec2013, f"F{i}2013")(ndim=D)
        errors = np.array([run_single(i, r, D) for r in range(N_RUNS)])
        rows.append({
            "Function": f"f{i}",
            "Name": f_inst.name,
            "D": D,
            "Runs": len(errors),
            "Mean Error": errors.mean(),
            "Std Dev": errors.std(ddof=1) if len(errors) > 1 else 0.0,
            "Best": errors.min(),
            "Worst": errors.max(),
        })
    df = pd.DataFrame(rows)
    for col in ["Mean Error", "Std Dev", "Best", "Worst"]:
        df[col] = df[col].map(lambda x: f"{x:.6E}")
    csv_path = output_dir / f"nabc_cec2013_D{D}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV enregistre : {csv_path.resolve()}")

if __name__ == "__main__":
    for D in [30, 50]:
        run_dimension(D, "results")
