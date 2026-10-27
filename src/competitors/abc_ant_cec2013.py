import os
import time
import numpy as np
import pandas as pd
from opfunu.cec_based import cec2013

np.set_printoptions(suppress=False, precision=17)
pd.set_option("display.float_format", lambda v: f"{v!r}")

RESULTS_DIR = "abcant_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class ABCANT:
    def __init__(self, func, ndim, lb, ub, SN=50, limit=100, CR=0.5,
                 max_nfe=None, rng=None):
        self.func = func
        self.D = ndim
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.SN = SN
        self.limit = limit
        self.CR = CR
        self.max_nfe = max_nfe if max_nfe is not None else 10000 * ndim
        self.rng = rng if rng is not None else np.random.default_rng()

        self.n_random = max(2, int(round(0.3 * SN)))
        self.n_ring_radius = max(1, int(round(0.1 * SN)))
        self.n_cellular = 4

        self.elite_size = max(2, int(round(0.1 * SN)))

        self.nfe = 0
        self.history = []

    def _evaluate(self, x):
        x = np.clip(x, self.lb, self.ub)
        val = self.func.evaluate(x)
        self.nfe += 1
        return val

    def _init_population(self):
        X = self.lb + self.rng.random((self.SN, self.D)) * (self.ub - self.lb)
        f = np.empty(self.SN)
        for i in range(self.SN):
            f[i] = self._evaluate(X[i])
        return X, f

    def _elite_indices(self, f):
        order = np.argsort(f)
        return order[:self.elite_size]

    def _compute_fdc(self, X, f, gb_idx):
        dists = np.linalg.norm(X - X[gb_idx], axis=1)
        fits = f
        if np.std(dists) < 1e-300 or np.std(fits) < 1e-300:
            return 1.0
        r = np.corrcoef(fits, dists)[0, 1]
        if np.isnan(r):
            r = 1.0
        return r

    def _select_topology(self, r):
        if r > 0.75:
            return "random"
        elif r < 0.15:
            return "cellular"
        else:
            return "ring"

    def _neighbors(self, i, topology):
        SN = self.SN
        if topology == "random":
            pool = [k for k in range(SN) if k != i]
            size = min(self.n_random, len(pool))
            return self.rng.choice(pool, size=size, replace=False)
        elif topology == "ring":
            r = self.n_ring_radius
            idx = [(i + off) % SN for off in range(-r, r + 1) if off != 0]
            return np.array(idx)
        else:
            idx = [(i - 2) % SN, (i - 1) % SN, (i + 1) % SN, (i + 2) % SN]
            idx = list(dict.fromkeys(idx))
            idx = [k for k in idx if k != i]
            return np.array(idx)

    def _repair(self, v, elite_idx, X):
        out = (v < self.lb) | (v > self.ub)
        if np.any(out):
            e = X[self.rng.choice(elite_idx)]
            v = v.copy()
            v[out] = e[out]
        return np.clip(v, self.lb, self.ub)

    def optimize(self):
        X, f = self._init_population()
        trial = np.zeros(self.SN, dtype=int)
        gb_idx = int(np.argmin(f))
        gb_val = f[gb_idx]
        elite_idx = self._elite_indices(f)

        trigger_fdc = True
        topology = "random"

        while self.nfe < self.max_nfe:
            gb_updated = False

            if trigger_fdc:
                r = self._compute_fdc(X, f, gb_idx)
                topology = self._select_topology(r)

            for i in range(self.SN):
                if self.nfe >= self.max_nfe:
                    break
                nb = self._neighbors(i, topology)
                nb_best = nb[np.argmin(f[nb])]
                cand_pool = [k for k in range(self.SN) if k not in (i, nb_best)]
                r_idx = self.rng.choice(cand_pool)

                j = self.rng.integers(0, self.D)
                phi = self.rng.uniform(-1, 1)
                w = self.rng.uniform(0, 1.5)

                V = X[i].copy()
                V[j] = (X[nb_best, j] + phi * (X[nb_best, j] - X[r_idx, j])
                        + w * (X[gb_idx, j] - X[nb_best, j]))
                V = self._repair(V, elite_idx, X)

                fv = self._evaluate(V)
                if fv < f[i]:
                    X[i] = V
                    f[i] = fv
                    trial[i] = 0
                else:
                    trial[i] += 1

                if f[i] < gb_val:
                    gb_val = f[i]
                    gb_idx = i
                    gb_updated = True

            for i in range(self.SN):
                if self.nfe >= self.max_nfe:
                    break
                nb = self._neighbors(i, topology)
                nb_best = nb[np.argmin(f[nb])]
                cand_pool = [k for k in range(self.SN) if k != nb_best]
                r_idx = self.rng.choice(cand_pool)

                j = self.rng.integers(0, self.D)
                phi = self.rng.uniform(-1, 1)

                V = X[nb_best].copy()
                V[j] = X[nb_best, j] + phi * (X[nb_best, j] - X[r_idx, j])
                V = self._repair(V, elite_idx, X)

                fv = self._evaluate(V)
                if fv < f[nb_best]:
                    X[nb_best] = V
                    f[nb_best] = fv
                    trial[nb_best] = 0
                else:
                    trial[nb_best] += 1

                if f[nb_best] < gb_val:
                    gb_val = f[nb_best]
                    gb_idx = nb_best
                    gb_updated = True

            elite_idx = self._elite_indices(f)
            for i in range(self.SN):
                if self.nfe >= self.max_nfe:
                    break
                if trial[i] > self.limit:
                    e_pool = [k for k in elite_idx if k != i]
                    if len(e_pool) < 2:
                        e_pool = list(elite_idx)
                    e1, e2 = self.rng.choice(e_pool, size=2, replace=False)

                    abc_w = self.rng.random(3)
                    abc_w = abc_w / abc_w.sum()
                    a, b, c = abc_w

                    jrand = self.rng.integers(0, self.D)
                    mask = (self.rng.random(self.D) <= self.CR)
                    mask[jrand] = True

                    V = X[i].copy()
                    V[mask] = a * X[i, mask] + b * X[e1, mask] + c * X[e2, mask]
                    V = np.clip(V, self.lb, self.ub)

                    fv = self._evaluate(V)
                    X[i] = V
                    f[i] = fv
                    trial[i] = 0

                    if f[i] < gb_val:
                        gb_val = f[i]
                        gb_idx = i
                        gb_updated = True

            elite_idx = self._elite_indices(f)
            trigger_fdc = not gb_updated

        return gb_val, X[gb_idx].copy()


def build_cec2013_suite(ndim):
    suite = {}
    for fid in range(1, 29):
        cls = getattr(cec2013, f"F{fid}2013")
        suite[fid] = cls(ndim=ndim)
    return suite


SN = 50
LIMIT = 100
CR = 0.5
N_RUNS = 30
NFE_PER_D = 10000


def run_experiment(D):
    suite = build_cec2013_suite(D)
    max_nfe = NFE_PER_D * D
    rows = []

    for fid in range(1, 29):
        func = suite[fid]
        errors = []
        print(f"[D={D}] F{fid:02d}")
        for run_id in range(N_RUNS):
            seed = 100000 * D + 1000 * fid + run_id
            rng = np.random.default_rng(seed)
            opt = ABCANT(func, ndim=D, lb=func.lb, ub=func.ub,
                         SN=SN, limit=LIMIT, CR=CR, max_nfe=max_nfe, rng=rng)
            t0 = time.time()
            best_f, best_x = opt.optimize()
            elapsed = time.time() - t0
            error = best_f - func.f_global
            errors.append(error)
            print(f"    run {run_id + 1:2d}/{N_RUNS}  error={error!r}  "
                  f"nfe={opt.nfe}  time={elapsed:.1f}s")

        errors = np.array(errors, dtype=float)
        rows.append({
            "D": D,
            "Func": f"F{fid:02d}",
            "Best": np.min(errors),
            "Worst": np.max(errors),
            "Median": np.median(errors),
            "Mean": np.mean(errors),
            "Std": np.std(errors, ddof=1) if len(errors) > 1 else 0.0,
            "N_runs": len(errors),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"abc_ant_cec2013_D{D}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[D={D}] saved -> {csv_path}")
    return df


if __name__ == "__main__":
    run_experiment(30)
    run_experiment(50)
