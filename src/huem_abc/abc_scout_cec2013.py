
import sys, subprocess, os, time, signal, warnings
for pkg in ["opfunu", "tqdm", "pandas"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", pkg])

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import opfunu.cec_based.cec2013 as c13

warnings.filterwarnings("ignore")

def _sigterm_handler(signum, frame):
    raise KeyboardInterrupt("received SIGTERM")
signal.signal(signal.SIGTERM, _sigterm_handler)

VARIANT_NAME = "ABC_Scout"
SN = 50
LIMIT = 100
RUNS = 30
DIMS = [30, 50]
FUNC_IDS = list(range(1, 29))
ALPHA = 0.7
HUFS_RATIO = 0.2
USE_GROUND_TRUTH_FOPT = True
OUTPUT_DIR = "./mnabc_ablation_results/ABC_Scout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def MaxFEs_for(D):
    return 10000 * D

def build_cec2013_funcs(D):
    funcs = {}
    for i in range(1, 29):
        cls = getattr(c13, f"F{i}2013")
        inst = cls(ndim=D)
        funcs[i] = {
            "fn": inst.evaluate,
            "f_opt": float(inst.f_global),
            "bounds": (float(inst.lb[0]), float(inst.ub[0])),
        }
    return funcs

_CEC_CACHE = {}
def get_cec_funcs(D):
    if D not in _CEC_CACHE:
        _CEC_CACHE[D] = build_cec2013_funcs(D)
    return _CEC_CACHE[D]

class ABC_Scout:
    def __init__(self, func_id, D, cec_funcs, SN=50, MaxFEs=None, limit=100,
                 alpha=0.7, hufs_ratio=0.2, use_ground_truth_fopt=True):
        info = cec_funcs[func_id]
        self.fn, self.f_opt = info["fn"], info["f_opt"]
        self.lb, self.ub = info["bounds"]
        self.func_id, self.D, self.SN = func_id, D, SN
        self.MaxFEs = MaxFEs if MaxFEs is not None else 10000 * D
        self.limit = limit
        self.alpha = alpha
        self.hufs_ratio = hufs_ratio
        self.use_gt = use_ground_truth_fopt

    def _eval(self, x):
        return float(self.fn(x))

    @staticmethod
    def _classic_fitness(fvals):
        return np.where(fvals >= 0, 1.0 / (1.0 + fvals), 1.0 + np.abs(fvals))

    @staticmethod
    def _safe_probs(w):
        s = np.sum(w)
        if not np.isfinite(s) or s < 1e-12:
            n = len(w)
            return np.full(n, 1.0 / n)
        pr = np.clip(w, 0, None)
        pr = pr / (pr.sum() + 1e-12)
        return pr / pr.sum()

    def _other_index(self, i):
        k = np.random.randint(self.SN - 1)
        return k if k < i else k + 1

    def _scout_memory(self, x_best, x_r):
        r1, r2, r3, r4 = np.random.uniform(0, 1, 4)
        return np.clip(r1 * x_best + r2 * x_r + r3 * (self.lb + r4 * (self.ub - self.lb)),
                       self.lb, self.ub)

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        SN = self.SN
        pop = np.random.uniform(self.lb, self.ub, (SN, self.D))
        fvals = np.array([self._eval(x) for x in pop])
        trial = np.zeros(SN, int)
        NFE = SN
        best_val = fvals.min()

        while NFE < self.MaxFEs:
            for i in range(SN):
                if NFE >= self.MaxFEs:
                    break
                k = self._other_index(i)
                j = np.random.randint(self.D)
                v = pop[i].copy()
                v[j] = np.clip(pop[i, j] + np.random.uniform(-1, 1) * (pop[i, j] - pop[k, j]),
                               self.lb, self.ub)
                fv = self._eval(v); NFE += 1
                if fv <= fvals[i]:
                    pop[i], fvals[i], trial[i] = v, fv, 0
                else:
                    trial[i] += 1
                best_val = min(best_val, fv)

            fit = self._classic_fitness(fvals)
            pr = self._safe_probs(fit)
            for _ in range(SN):
                if NFE >= self.MaxFEs:
                    break
                i = np.random.choice(SN, p=pr)
                k = self._other_index(i)
                j = np.random.randint(self.D)
                v = pop[i].copy()
                v[j] = np.clip(pop[i, j] + np.random.uniform(-1, 1) * (pop[i, j] - pop[k, j]),
                               self.lb, self.ub)
                fv = self._eval(v); NFE += 1
                if fv <= fvals[i]:
                    pop[i], fvals[i], trial[i] = v, fv, 0
                else:
                    trial[i] += 1
                best_val = min(best_val, fv)

            x_best = pop[np.argmin(fvals)]
            for i in range(SN):
                if NFE >= self.MaxFEs:
                    break
                if trial[i] > self.limit:
                    x_r = pop[self._other_index(i)]
                    pop[i] = self._scout_memory(x_best, x_r)
                    fvals[i] = self._eval(pop[i]); NFE += 1; trial[i] = 0
                    best_val = min(best_val, fvals[i])

        return max(0.0, best_val - self.f_opt)

def results_csv_for(D):
    return os.path.join(OUTPUT_DIR, f"abc_scout_cec2013_D{D}.csv")

def aggregate(D):
    path = results_csv_for(D)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    summary = (df.groupby(["variant", "D", "func_id"])["final_error"]
               .agg(mean="mean", std="std", best="min", worst="max", median="median", n_runs="count")
               .reset_index().sort_values(["func_id"]))
    summary["std"] = summary["std"].fillna(0.0)
    out_path = os.path.join(OUTPUT_DIR, f"summary_abc_scout_cec2013_D{D}.csv")
    summary.to_csv(out_path, index=False)
    return summary

def run_experiment(dims=DIMS, func_ids=FUNC_IDS, runs=RUNS):
    total = len(dims) * len(func_ids) * runs
    pbar = tqdm(total=total, desc=VARIANT_NAME)

    for D in dims:
        csv_path = results_csv_for(D)
        if os.path.exists(csv_path):
            done_df = pd.read_csv(csv_path)
        else:
            done_df = pd.DataFrame(columns=["variant", "D", "func_id", "run_idx", "seed",
                                            "final_error", "runtime_sec", "timestamp"])
        done = set(zip(done_df.func_id, done_df.run_idx))
        cec_funcs = get_cec_funcs(D)
        maxfe = MaxFEs_for(D)
        buffer_rows = []
        pbar.reset(total=len(dims) * len(func_ids) * runs)
        pbar.initial = sum(len(pd.read_csv(results_csv_for(d))) if os.path.exists(results_csv_for(d)) else 0
                           for d in dims if os.path.exists(results_csv_for(d)))
        try:
            for func_id in func_ids:
                for run_idx in range(runs):
                    if (func_id, run_idx) in done:
                        continue
                    seed = 100000 * func_id + 1000 * D + run_idx
                    t0 = time.time()
                    algo = ABC_Scout(func_id=func_id, D=D, cec_funcs=cec_funcs, SN=SN,
                                     MaxFEs=maxfe, limit=LIMIT, alpha=ALPHA,
                                     hufs_ratio=HUFS_RATIO,
                                     use_ground_truth_fopt=USE_GROUND_TRUTH_FOPT)
                    err = algo.run(seed=seed)
                    rt = time.time() - t0
                    buffer_rows.append({
                        "variant": VARIANT_NAME, "D": D, "func_id": func_id,
                        "run_idx": run_idx, "seed": seed,
                        "final_error": err, "runtime_sec": rt,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    done.add((func_id, run_idx))
                    pbar.update(1)
                    if len(buffer_rows) >= 10:
                        done_df = pd.concat([done_df, pd.DataFrame(buffer_rows)], ignore_index=True)
                        done_df.to_csv(csv_path, index=False)
                        buffer_rows = []
        finally:
            if buffer_rows:
                done_df = pd.concat([done_df, pd.DataFrame(buffer_rows)], ignore_index=True)
            done_df.to_csv(csv_path, index=False)

    pbar.close()
    for D in dims:
        aggregate(D)

if __name__ == "__main__":
    run_experiment()
