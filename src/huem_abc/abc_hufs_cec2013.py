import sys, subprocess
for pkg in ["opfunu", "tqdm", "scipy", "pandas"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", pkg])

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
import opfunu.cec_based.cec2013 as c13

warnings.filterwarnings("ignore")

VARIANT_NAME = "ABC_HUFS"

SN          = 50
LIMIT       = 100
RUNS        = 30
DIMS        = [30, 50]
FUNC_IDS    = list(range(1, 29))
ALPHA       = 0.7
HUFS_RATIO  = 0.2
USE_GROUND_TRUTH_FOPT = True
OUTPUT_DIR  = "./mnabc_ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def MaxFEs_for(D):
    return 10000 * D

def build_cec2013_funcs(D):
    funcs = {}
    for i in range(1, 29):
        cls = getattr(c13, f"F{i}2013")
        inst = cls(ndim=D)
        funcs[i] = {
            'fn': inst.evaluate,
            'f_opt': float(inst.f_global),
            'bounds': (float(inst.lb[0]), float(inst.ub[0])),
        }
    return funcs

_CEC_CACHE = {}
def get_cec_funcs(D):
    if D not in _CEC_CACHE:
        _CEC_CACHE[D] = build_cec2013_funcs(D)
    return _CEC_CACHE[D]

class _ABCBase:
    def __init__(self, func_id, D, cec_funcs, SN=50, MaxFEs=None, limit=100,
                 alpha=0.7, hufs_ratio=0.2, use_ground_truth_fopt=True):
        info = cec_funcs[func_id]
        self.fn, self.f_opt = info['fn'], info['f_opt']
        self.lb, self.ub = info['bounds']
        self.func_id, self.D, self.SN = func_id, D, SN
        self.MaxFEs = MaxFEs if MaxFEs is not None else 10000 * D
        self.limit = limit
        self.alpha = alpha
        self.hufs_ratio = hufs_ratio
        self.use_gt = use_ground_truth_fopt

    def _eval(self, x):
        return float(self.fn(x))

    def _quality(self, fvals, f_star):
        target = self.f_opt if self.use_gt else f_star
        return 1.0 / (1.0 + np.abs(fvals - target))

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

    def _diversity(self, pop):
        Dm = cdist(pop, pop)
        np.fill_diagonal(Dm, np.inf)
        return Dm.min(axis=1)

    def _utility(self, pop, fvals, f_star, alpha):
        q = self._quality(fvals, f_star)
        qn = (q - q.min()) / (q.max() - q.min() + 1e-10)
        d = self._diversity(pop)
        dn = (d - d.min()) / (d.max() - d.min() + 1e-10)
        return alpha * qn + (1 - alpha) * dn

    def _hufs_utility(self, pop, fvals, f_star, alpha):
        u = self._utility(pop, fvals, f_star, alpha)
        k = max(1, int(self.SN * self.hufs_ratio))
        idx = np.argsort(u)[::-1][:k]
        return pop[idx]

    def _anchor_shift(self, x_i, anchor, x_k):
        j = np.random.randint(self.D)
        v = x_i.copy()
        v[j] = np.clip(anchor[j] + np.random.uniform(-1, 1) * (x_i[j] - x_k[j]),
                        self.lb, self.ub)
        return v

    def _scout_random(self):
        return np.random.uniform(self.lb, self.ub, self.D)

    def _other_index(self, i):
        k = np.random.randint(self.SN - 1)
        return k if k < i else k + 1

    def _init_pop(self):
        pop = np.random.uniform(self.lb, self.ub, (self.SN, self.D))
        fvals = np.array([self._eval(x) for x in pop])
        return pop, fvals

class ABC_HUFS(_ABCBase):
    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        SN = self.SN
        pop, fvals = self._init_pop()
        trial = np.zeros(SN, int)
        NFE = SN
        best_val = fvals.min()
        history = [max(0.0, best_val - self.f_opt)]

        while NFE < self.MaxFEs:
            hufs = self._hufs_utility(pop, fvals, best_val, alpha=1.0)

            for i in range(SN):
                if NFE >= self.MaxFEs: break
                anc = hufs[np.random.randint(len(hufs))]
                k = self._other_index(i)
                v = self._anchor_shift(pop[i], anc, pop[k])
                fv = self._eval(v); NFE += 1
                if fv <= fvals[i]: pop[i], fvals[i], trial[i] = v, fv, 0
                else: trial[i] += 1
                best_val = min(best_val, fv)

            fit = self._classic_fitness(fvals)
            pr = self._safe_probs(fit)
            for _ in range(SN):
                if NFE >= self.MaxFEs: break
                i = np.random.choice(SN, p=pr)
                anc = hufs[np.random.randint(len(hufs))]
                k = self._other_index(i)
                v = self._anchor_shift(pop[i], anc, pop[k])
                fv = self._eval(v); NFE += 1
                if fv <= fvals[i]: pop[i], fvals[i], trial[i] = v, fv, 0
                else: trial[i] += 1
                best_val = min(best_val, fv)

            for i in range(SN):
                if NFE >= self.MaxFEs: break
                if trial[i] > self.limit:
                    pop[i] = self._scout_random()
                    fvals[i] = self._eval(pop[i]); NFE += 1; trial[i] = 0
                    best_val = min(best_val, fvals[i])

            history.append(max(0.0, best_val - self.f_opt))

        return max(0.0, best_val - self.f_opt), np.array(history)

def run_experiment():
    for D in DIMS:
        csv_path = os.path.join(OUTPUT_DIR, f"abc_hufs_cec2013_D{D}.csv")
        done = set()
        if os.path.exists(csv_path):
            done = set(zip(pd.read_csv(csv_path).func_id, pd.read_csv(csv_path).run_idx))

        cec_funcs = get_cec_funcs(D)
        maxfe = MaxFEs_for(D)
        total = len(FUNC_IDS) * RUNS
        pbar = tqdm(total=total, initial=len(done), desc=f"ABC_HUFS D={D}")

        buffer_rows = []
        for func_id in FUNC_IDS:
            for run_idx in range(RUNS):
                if (func_id, run_idx) in done:
                    continue
                seed = 100000 * func_id + 1000 * D + run_idx
                t0 = time.time()
                algo = ABC_HUFS(func_id=func_id, D=D, cec_funcs=cec_funcs, SN=SN,
                              MaxFEs=maxfe, limit=LIMIT, alpha=ALPHA,
                              hufs_ratio=HUFS_RATIO,
                              use_ground_truth_fopt=USE_GROUND_TRUTH_FOPT)
                err, hist = algo.run(seed=seed)
                rt = time.time() - t0
                buffer_rows.append({
                    "variant": VARIANT_NAME, "D": D, "func_id": func_id,
                    "run_idx": run_idx, "seed": seed,
                    "final_error": err, "runtime_sec": rt,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                if os.path.exists(csv_path):
                    pd.DataFrame(buffer_rows).to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    pd.DataFrame(buffer_rows).to_csv(csv_path, index=False)
                buffer_rows = []
                pbar.update(1)
        pbar.close()

        df = pd.read_csv(csv_path)
        summary = (df.groupby(["variant","D","func_id"])["final_error"]
                   .agg(mean="mean", std="std", best="min", worst="max", median="median", n_runs="count")
                   .reset_index().sort_values(["D","func_id"]))
        summary["std"] = summary["std"].fillna(0.0)
        summary_path = os.path.join(OUTPUT_DIR, f"summary_mean_std_D{D}.csv")
        summary.to_csv(summary_path, index=False)
        print(f"[done] D={D}: wrote {csv_path} ({len(df)} runs) and {summary_path}")

if __name__ == "__main__":
    run_experiment()
