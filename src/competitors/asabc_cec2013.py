import numpy as np
import time


def dynamic_radius(fes, max_fes, sn):
    k = (sn / 4.0 - 1.0) * (fes / max_fes) + 1.0
    return max(1, int(round(k)))


def compute_ph(population, fitness_target):
    sn = population.shape[0]
    fitness_ref = np.einsum('ij,ij->i', population, population)

    def info_matrix(fit):
        M = np.full((sn, sn), 0.5)
        M[fit[:, None] < fit[None, :]] = 1.0
        M[fit[:, None] > fit[None, :]] = 0.0
        return M

    Mt = info_matrix(fitness_target)
    Mr = info_matrix(fitness_ref)
    best_idx = int(np.argmin(fitness_target))

    keep = np.ones(sn, dtype=bool)
    keep[best_idx] = False
    Mt_sub = Mt[np.ix_(keep, keep)]
    Mr_sub = Mr[np.ix_(keep, keep)]
    iu = np.triu_indices(sn - 1, k=1)
    vt = Mt_sub[iu]
    vr = Mr_sub[iu]
    if vt.size == 0:
        return 0.0
    return float(np.mean(np.abs(vt - vr)))


class ASABC:

    def __init__(self, objective, lb, ub, dim, max_fes,
                 sn=50, limit=50, phbdy=0.3, cr_exl=0.1, cr_exp=0.7,
                 ph_period=50, seed=None):
        self.f = objective
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.D = dim
        self.max_fes = int(max_fes)
        self.SN = sn
        self.limit = limit
        self.phbdy = phbdy
        self.cr_exl = cr_exl
        self.cr_exp = cr_exp
        self.ph_period = ph_period
        self.rng = np.random.default_rng(seed)

        self.fes = 0
        self.X = None
        self.fit_raw = None
        self.trial = None
        self.mode = "exploit"

    def _eval(self, x):
        np.clip(x, self.lb, self.ub, out=x)
        self.fes += 1
        return float(self.f(x))

    def _abc_fitness(self, fraw):
        return np.where(fraw >= 0, 1.0 / (1.0 + fraw), 1.0 + np.abs(fraw))

    def _init_population(self):
        self.X = self.lb + self.rng.random((self.SN, self.D)) * (self.ub - self.lb)
        self.fit_raw = np.empty(self.SN)
        for i in range(self.SN):
            self.fit_raw[i] = self._eval(self.X[i])
        self.trial = np.zeros(self.SN, dtype=int)

    def _update_ph_mode(self):
        ph = compute_ph(self.X, self.fit_raw)
        self.mode = "exploit" if ph < self.phbdy else "explore"

    def _rand_others(self, i, n, exclude=None):
        exc0 = i
        exc1 = exclude[0] if exclude else -1
        exc2 = exclude[1] if exclude and len(exclude) > 1 else -1
        rng = self.rng
        sn = self.SN
        if n == 1:
            while True:
                c = int(rng.integers(0, sn))
                if c != exc0 and c != exc1 and c != exc2:
                    return c
        else:
            r1 = -1
            while True:
                c = int(rng.integers(0, sn))
                if c != exc0 and c != exc1 and c != exc2 and c != r1:
                    if r1 == -1:
                        r1 = c
                    else:
                        return r1, c

    def _offsets(self, k_radius):
        return np.concatenate((np.arange(-k_radius, 0), np.arange(1, k_radius + 1)))

    def _lbest_of(self, i, offsets, fit_raw):
        neigh = (i + offsets) % self.SN
        return int(neigh[np.argmin(fit_raw[neigh])])

    def _employed_phase(self, k_radius):
        offsets = self._offsets(k_radius)
        X, fit_raw, rng, D = self.X, self.fit_raw, self.rng, self.D
        exploit = self.mode == "exploit"
        for i in range(self.SN):
            v = X[i].copy()
            j = int(rng.integers(0, D))
            if exploit:
                lbest = self._lbest_of(i, offsets, fit_raw)
                r1, r2 = self._rand_others(i, 2, exclude=(lbest,))
                phi = rng.uniform(-1, 1)
                v[j] = X[lbest, j] + phi * (X[r1, j] - X[r2, j])
            else:
                k = self._rand_others(i, 1)
                phi = rng.uniform(-1, 1)
                v[j] = X[i, j] + phi * (X[i, j] - X[k, j])
            self._greedy_select(i, v)
            if self.fes >= self.max_fes:
                return

    def _onlooker_phase(self, k_radius):
        offsets = self._offsets(k_radius)
        X, fit_raw, rng, D = self.X, self.fit_raw, self.rng, self.D
        exploit = self.mode == "exploit"
        fitness = self._abc_fitness(fit_raw)
        probs = fitness / fitness.sum()
        best_idx = int(np.argmin(fit_raw))
        x_best = X[best_idx]

        t = 0
        i = 0
        while t < self.SN:
            if rng.random() < probs[i]:
                v = X[i].copy()
                j = int(rng.integers(0, D))
                phi = rng.uniform(-1, 1)
                r1, r2 = self._rand_others(i, 2)
                if exploit:
                    lbest = self._lbest_of(i, offsets, fit_raw)
                    v[j] = (X[lbest, j] + phi * (X[i, j] - X[r1, j])
                            + phi * (x_best[j] - X[r2, j]))
                else:
                    v[j] = (X[i, j] + phi * (X[i, j] - X[r1, j])
                            + phi * (x_best[j] - X[r2, j]))
                self._greedy_select(i, v)
                t += 1
                if self.fes >= self.max_fes:
                    return
            i = (i + 1) % self.SN

    def _greedy_select(self, i, v):
        fv = self._eval(v)
        if fv <= self.fit_raw[i]:
            self.X[i] = v
            self.fit_raw[i] = fv
            self.trial[i] = 0
        else:
            self.trial[i] += 1

    def _scout_phase(self, k_radius):
        best_idx = int(np.argmin(self.fit_raw))
        x_best = self.X[best_idx]
        offsets = None
        for i in range(self.SN):
            if self.trial[i] <= self.limit:
                continue
            if self.fes >= self.max_fes:
                return
            v = self.X[i].copy()
            if self.mode == "exploit":
                mask = self.rng.random(self.D) <= self.cr_exl
                if mask.any():
                    v[mask] = self.rng.normal(x_best[mask], 1.0)
            else:
                if offsets is None:
                    offsets = self._offsets(k_radius)
                r = self._rand_others(i, 1)
                neigh = (r + offsets) % self.SN
                x_rlbest = self.X[neigh[np.argmin(self.fit_raw[neigh])]]
                mask = self.rng.random(self.D) <= self.cr_exp
                if mask.any():
                    mean = (x_rlbest[mask] + x_best[mask]) / 2.0
                    std = np.abs(x_rlbest[mask] - x_best[mask]) + 1e-12
                    v[mask] = self.rng.normal(mean, std)
            fv = self._eval(v)
            self.X[i] = v
            self.fit_raw[i] = fv
            self.trial[i] = 0

    def run(self):
        self._init_population()
        gen = 0
        self._update_ph_mode()
        while self.fes < self.max_fes:
            k_radius = dynamic_radius(self.fes, self.max_fes, self.SN)
            self._employed_phase(k_radius)
            if self.fes >= self.max_fes:
                break
            self._onlooker_phase(k_radius)
            if self.fes >= self.max_fes:
                break
            self._scout_phase(k_radius)
            gen += 1
            if gen % self.ph_period == 0:
                self._update_ph_mode()
        best_idx = int(np.argmin(self.fit_raw))
        return float(self.fit_raw[best_idx]), self.fes


def run_trial(objective, lb, ub, dim, max_fes, seed, f_global=0.0, **kwargs):
    t0 = time.time()
    algo = ASABC(objective, lb, ub, dim, max_fes, seed=seed, **kwargs)
    best_raw, fes_used = algo.run()
    err = max(best_raw - f_global, 0.0)
    return {"best_error": err, "fes_used": fes_used, "time_sec": time.time() - t0}


import os
import json
import csv
from opfunu.cec_based import cec2013

DIMENSIONS = [30, 50]
N_RUNS = 30
SN = 50
LIMIT = 100
PHBDY = 0.3
CR_EXL = 0.1
CR_EXP = 0.7
PH_PERIOD = 50

MAXFES_PER_DIM = 10000

FUNCTION_NAMES = [f"F{i}" for i in range(1, 29)]

CHECKPOINT_EVERY_SEC = 30 * 60

OUTPUT_DIR = os.path.abspath("results")
RAW_CSV = os.path.join(OUTPUT_DIR, "asabc_cec2013_raw_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "asabc_cec2013_summary.csv")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_opfunu_class(fname, dim):
    idx = int(fname.replace("F", ""))
    cls = getattr(cec2013, f"F{idx}2013")
    return cls(ndim=dim)


print("Configuration loaded.")
print(f"  {len(FUNCTION_NAMES)} functions x {len(DIMENSIONS)} dimensions x {N_RUNS} runs "
      f"= {len(FUNCTION_NAMES)*len(DIMENSIONS)*N_RUNS} trials in total.")
for d in DIMENSIONS:
    print(f"  MaxFEs for D={d}: {MAXFES_PER_DIM * d:,} (expected: {10000*d:,})")
print(f"  Results -> {RAW_CSV}")
print(f"  Automatic backups every {CHECKPOINT_EVERY_SEC//60} min -> {CHECKPOINT_DIR}")

RAW_FIELDS = ["function", "dim", "run", "best_error", "fes_used", "time_sec", "timestamp"]


def load_done_trials():
    done = set()
    if os.path.exists(RAW_CSV):
        with open(RAW_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["function"], int(row["dim"]), int(row["run"])))
    else:
        with open(RAW_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RAW_FIELDS).writeheader()
    return done


def append_result(function, dim, run, result):
    with open(RAW_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writerow({
            "function": function, "dim": dim, "run": run,
            "best_error": result["best_error"], "fes_used": result["fes_used"],
            "time_sec": round(result["time_sec"], 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })


def save_checkpoint(n_done, n_total, t_start):
    ts = time.strftime("%Y%m%d_%H%M%S")
    snap_path = os.path.join(CHECKPOINT_DIR, f"raw_results_checkpoint_{ts}.csv")
    if os.path.exists(RAW_CSV):
        import shutil
        shutil.copyfile(RAW_CSV, snap_path)
    state = {
        "timestamp": ts,
        "trials_done": n_done,
        "trials_total": n_total,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(CHECKPOINT_DIR, f"state_{ts}.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"[checkpoint {ts}] {n_done}/{n_total} trials completed - snapshot saved.")


done_trials = load_done_trials()

tasks = [
    (fname, dim, run)
    for dim in DIMENSIONS
    for fname in FUNCTION_NAMES
    for run in range(1, N_RUNS + 1)
]
n_total = len(tasks)

t_start = time.time()
last_checkpoint = t_start
n_done = len(done_trials)

print(f"{n_done}/{n_total} trials already completed (resumed from {RAW_CSV}).")

for (fname, dim, run) in tasks:
    if (fname, dim, run) in done_trials:
        continue

    prob = get_opfunu_class(fname, dim)
    max_fes = MAXFES_PER_DIM * dim
    seed = hash((fname, dim, run)) % (2**32)

    result = run_trial(
        objective=lambda x, _p=prob: _p.evaluate(x),
        lb=prob.lb, ub=prob.ub, dim=dim, max_fes=max_fes, seed=seed,
        f_global=prob.f_global,
        sn=SN, limit=LIMIT, phbdy=PHBDY, cr_exl=CR_EXL, cr_exp=CR_EXP,
        ph_period=PH_PERIOD,
    )
    append_result(fname, dim, run, result)
    n_done += 1

    if time.time() - last_checkpoint >= CHECKPOINT_EVERY_SEC:
        save_checkpoint(n_done, n_total, t_start)
        last_checkpoint = time.time()

    if n_done % 10 == 0 or n_done == n_total:
        elapsed = time.time() - t_start
        print(f"[{n_done}/{n_total}] {fname} D={dim} run={run} "
              f"err={result['best_error']:.4e}  (elapsed {elapsed/60:.1f} min)")

save_checkpoint(n_done, n_total, t_start)
print("Experiment completed." if n_done == n_total else "Interrupted - rerun the script to resume.")

import pandas as pd

raw = pd.read_csv(RAW_CSV)
summary = (
    raw.groupby(["function", "dim"])
       .agg(mean_error=("best_error", "mean"),
            std_error=("best_error", "std"),
            n_runs=("best_error", "count"),
            mean_fes=("fes_used", "mean"))
       .reset_index()
)
summary["function_order"] = summary["function"].str.replace("F", "").astype(int)
summary = summary.sort_values(["dim", "function_order"]).drop(columns="function_order")
summary.to_csv(SUMMARY_CSV, index=False)

print(f"Summary written -> {SUMMARY_CSV}")
print(summary.to_string())

raw_all = pd.read_csv(RAW_CSV)
raw_all["function_order"] = raw_all["function"].str.replace("F", "").astype(int)

for d in DIMENSIONS:
    sub = raw_all[raw_all["dim"] == d]
    mean_df = (
        sub.groupby(["function_order", "function"], as_index=False)
           .agg(mean_error=("best_error", "mean"),
                mean_fes=("fes_used", "mean"),
                mean_time_sec=("time_sec", "mean"))
           .sort_values("function_order")
           .drop(columns="function_order")
    )
    out_path = os.path.join(OUTPUT_DIR, f"asabc_cec2013_D{d}.csv")
    mean_df.to_csv(out_path, index=False)
    print(f"D={d}: {len(mean_df)} functions -> {out_path}")
