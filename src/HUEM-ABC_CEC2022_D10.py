# %%
# Cell 0 — Imports & style publication
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, friedmanchisquare
from tabulate import tabulate
import warnings, time, os, glob, pickle, threading, zipfile
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})
C_HUEM = '#C0392B'
print('Imports OK.')

# %%
# Cell 1 — Paramètres expérimentaux (CEC 2022, D = 10)
D       = 10                     # CEC 2022 : D = 10 (MaxFEs = 200 000)
SN      = 50                     # taille de colonie (papier 4.1.1)
MaxFEs  = 200_000                # budget CEC 2022 a D=10
LIMIT   = 100                    # limite d'abandon (papier 4.1.1)
RUNS    = 30                     # runs indépendants
ALPHA0  = 1                    # config retenue
HR0     = 0.1                    # HUFS ratio retenu
N_FUNCS = 12

OUT_DIR  = f'results_HUEMABC_CEC2022_D{D}_{HR0:.2f}'
CKPT_DIR = f'{OUT_DIR}/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

print('Parametres — HUEM-ABC, CEC 2022 (D = 10) :')
for k, v in dict(D=D, SN=SN, MaxFEs=MaxFEs, limit=LIMIT, alpha=ALPHA0,
                 hufs_ratio=HR0, RUNS=RUNS, N_FUNCS=N_FUNCS).items():
    print(f'  {k:12s} = {v}')

# %%
# Cell 2 — Chargement des fonctions CEC 2022 officielles
from cec2022_official import CEC2022   # necessite input_data/ a cote du fichier

CEC2022_INFO = {
     1: ('F1',  'Zakharov',            'Unimodal'),
     2: ('F2',  'Rosenbrock',          'Basic Multimodal'),
     3: ('F3',  'Schaffer F7 (rot.)',  'Basic Multimodal'),
     4: ('F4',  'Step-Rastrigin',      'Basic Multimodal'),
     5: ('F5',  'Levy',                'Basic Multimodal'),
     6: ('F6',  'Hybrid 1 (HF2)',      'Hybrid'),
     7: ('F7',  'Hybrid 2 (HF10)',     'Hybrid'),
     8: ('F8',  'Hybrid 3 (HF6)',      'Hybrid'),
     9: ('F9',  'Composition 1 (CF1)', 'Composition'),
    10: ('F10', 'Composition 2 (CF2)', 'Composition'),
    11: ('F11', 'Composition 3 (CF6)', 'Composition'),
    12: ('F12', 'Composition 4 (CF7)', 'Composition'),
}

CEC2022_FUNCS = {}
for fid in range(1, 13):
    f = CEC2022(func_num=fid, dim=D)
    CEC2022_FUNCS[fid] = {'fn': f, 'f_opt': float(f.f_opt),
                          'lb': np.asarray(f.lb, float), 'ub': np.asarray(f.ub, float),
                          'name': CEC2022_INFO[fid][0], 'desc': CEC2022_INFO[fid][1],
                          'type': CEC2022_INFO[fid][2]}

# Validation rapide : f(x*) = f_opt, valeurs finies sur tirages uniformes
_rng = np.random.RandomState(0)
print('Validation CEC 2022 (D=10) :')
for fid, v in CEC2022_FUNCS.items():
    xs = _rng.uniform(v['lb'], v['ub'], (20, D))
    vals = np.array([v['fn'](x) for x in xs])
    ok = np.all(np.isfinite(vals))
    print(f'  F{fid:<2d} f_opt={v["f_opt"]:7.0f}  min(20 tirages)={vals.min():.3e}  max={vals.max():.3e}  {"OK" if ok else "PROBLEME"}')
assert ok
print(f'OK — {len(CEC2022_FUNCS)} fonctions CEC 2022 chargees (D={D}), bornes [-100,100].')

# %%
# Cell 3 — HUEM-ABC (optimiseur, perturbation 1 dimension)
class HUEMABCOptimizer:
    def __init__(self, func_id, SN=50, MaxFEs=200_000, limit=100, D=10,
                 alpha=0.7, hufs_ratio=0.2):
        info = CEC2022_FUNCS[func_id]
        self.fn, self.f_opt = info['fn'], info['f_opt']
        self.lb, self.ub = info['lb'], info['ub']
        self.func_id = func_id
        self.SN, self.MaxFEs, self.limit, self.D = SN, MaxFEs, limit, D
        self.alpha, self.hufs_ratio = alpha, hufs_ratio

    def _eval(self, x):
        return float(self.fn(x))

    def _utility(self, pop, fvals):
        fa = fvals.copy()
        fr = (fa.max() - fa) / ((fa.max() - fa.min()) + 1e-12)   # qualite (rang f)
        diff = pop[:, None, :] - pop[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        d = dist.min(axis=1)
        dn = (d - d.min()) / ((d.max() - d.min()) + 1e-12)       # diversite normalisee
        return self.alpha * fr + (1 - self.alpha) * dn           # Eq. (6)

    def _hufs(self, pop, fvals):
        utils = self._utility(pop, fvals)
        k = max(1, int(self.SN * self.hufs_ratio))
        idx = np.argsort(utils)[::-1][:k]
        return pop[idx], fvals[idx]

    def _anchor_shift(self, pop, i, anchor, k):
        jj = np.random.randint(self.D)                           # Eq. (8), 1 dimension
        v = pop[i].copy()
        x = anchor[jj] + np.random.uniform(-1, 1) * (pop[i, jj] - pop[k, jj])
        v[jj] = self.lb[jj] if x < self.lb[jj] else (self.ub[jj] if x > self.ub[jj] else x)
        return v

    def run(self, seed=None):
        if seed is not None: np.random.seed(seed)
        lb, ub, D, SN = self.lb, self.ub, self.D, self.SN
        MaxFEs, limit, f_opt = self.MaxFEs, self.limit, self.f_opt
        _eval, _shift = self._eval, self._anchor_shift
        randint, random_sample = np.random.randint, np.random.random_sample
        n_other = SN - 1

        pop = np.random.uniform(lb, ub, (SN, D))
        fvals = np.array([_eval(x) for x in pop])
        trial = np.zeros(SN, dtype=int)
        best_val = fvals.min(); NFE = SN
        history = [max(0.0, best_val - f_opt)]

        while NFE < MaxFEs:
            hufs, hufs_fv = self._hufs(pop, fvals)
            n_hufs = len(hufs)
            x_best = hufs[np.argmin(hufs_fv)]

            # Employed
            for i in range(SN):
                if NFE >= MaxFEs: break
                anc = hufs[randint(n_hufs)]
                k = randint(n_other);  k += (k >= i)
                v = _shift(pop, i, anc, k); fv = _eval(v); NFE += 1
                if fv <= fvals[i]: pop[i] = v; fvals[i] = fv; trial[i] = 0
                else: trial[i] += 1
                best_val = min(best_val, fv)

            # Onlooker (roulette sur U + delta = 0.01)
            utils = self._utility(pop, fvals) + 0.01
            pr = utils / utils.sum(); cdf = pr.cumsum(); cdf /= cdf[-1]
            for _ in range(SN):
                if NFE >= MaxFEs: break
                i = int(cdf.searchsorted(random_sample(), side='right'))
                anc = hufs[randint(n_hufs)]
                k = randint(n_other);  k += (k >= i)
                v = _shift(pop, i, anc, k); fv = _eval(v); NFE += 1
                if fv <= fvals[i]: pop[i] = v; fvals[i] = fv; trial[i] = 0
                else: trial[i] += 1
                best_val = min(best_val, fv)

            # Scout — restart memoire, Eq. (9)
            for i in range(SN):
                if NFE >= MaxFEs: break
                if trial[i] > limit:
                    r1, r2, r3 = np.random.uniform(0, 1, 3)
                    w = r1 + r2 + r3
                    x_new = (r1 * x_best + r2 * pop[randint(SN)] + r3 * np.random.uniform(lb, ub)) / w
                    pop[i] = x_new; fvals[i] = _eval(x_new)
                    NFE += 1; trial[i] = 0
                    best_val = min(best_val, fvals[i])

            history.append(max(0.0, best_val - f_opt))

        return max(0.0, best_val - f_opt), np.array(history)

print('OK — HUEMABCOptimizer pret (CEC 2022).')

# %%
# Cell 4 — Sanity check (MaxFEs reduit)
for fid in [1, 7, 12]:
    err, _ = HUEMABCOptimizer(fid, SN=10, MaxFEs=500, limit=10, D=D).run(seed=0)
    print(f'  F{fid:<2d} ({CEC2022_INFO[fid][2]:12s}): erreur = {err:.3e}')
print('OK — sanity check.')

# %%
# Cell 5 — Système de checkpoints (reprise sur interruption)
results = {}                       # results[func_id] = {'errors': [...], 'hist': array|None}
_sens_results = {}                 # _sens_results[func_id][(alpha, h)] = {'errors': [...]}
_stop_evt = threading.Event()

def _save(path_stub, data):
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = f'{path_stub}_{ts}.pkl'
    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + '.tmp', 'wb') as fh: fh.write(blob)
    os.replace(path + '.tmp', path)
    return path

def _load_latest(directory, pattern, target_dict):
    cands = sorted(glob.glob(os.path.join(directory, pattern)))
    for p in cands[::-1]:
        try:
            with open(p, 'rb') as fh: target_dict.update(pickle.load(fh))
            print(f'OK checkpoint charge : {p} ({len(target_dict)} fonctions)')
            return
        except Exception:
            continue
    print('OK — aucun checkpoint exploitable, demarrage a zero.')

def _autosave_loop():
    n = 0
    while not _stop_evt.wait(1800):     # toutes les 30 min
        if not results and not _sens_results: continue
        n += 1
        try:
            _save(f'{CKPT_DIR}/chk_main', results)
            _save(f'{CKPT_DIR}/chk_sens', _sens_results)
            print(f'  [Autosave #{n}] main={len(results)} fns, sens={len(_sens_results)} fns')
        except Exception as e:
            print(f'  [Autosave #{n}] ERREUR : {e}')

_load_latest(CKPT_DIR, 'chk_main_*.pkl', results)
_load_latest(CKPT_DIR, 'chk_sens_*.pkl', _sens_results)
_autosave_t = threading.Thread(target=_autosave_loop, daemon=True)
_autosave_t.start()
print('OK — autosave demarre (30 min).')

# %%
# Cell 6 — TEST PRINCIPAL : HUEM-ABC (alpha=0.7, h=0.2), 12 fonctions x 30 runs
def print_summary(fid, errs):
    e = np.asarray(errs, float)
    print(f'  >> {CEC2022_INFO[fid][0]:<4s} {CEC2022_INFO[fid][1]:22s} {len(e)} runs | '
          f'mean = {e.mean():.4e}   std = {e.std(ddof=0):.4e}', flush=True)

t0 = time.time()
for fid in range(1, 13):
    if fid not in results:
        results[fid] = {'errors': [], 'hist': None}
    if len(results[fid]['errors']) >= RUNS:
        print('[complet] ', end=''); print_summary(fid, results[fid]['errors']); continue
    t_f = time.time()
    for run in range(len(results[fid]['errors']), RUNS):
        seed = run * 137 + fid * 31
        err, hist = HUEMABCOptimizer(fid, SN=SN, MaxFEs=MaxFEs, limit=LIMIT, D=D,
                                     alpha=ALPHA0, hufs_ratio=HR0).run(seed=seed)
        results[fid]['errors'].append(float(err))
        if results[fid]['hist'] is None:
            results[fid]['hist'] = hist
    print(f'{CEC2022_INFO[fid][0]} TERMINE [{time.time()-t_f:.0f}s]')
    print_summary(fid, results[fid]['errors'])

_stop_evt.set()
_save(f'{CKPT_DIR}/chk_main_FINAL', results)

rows, raw = [], []
for fid in range(1, 13):
    e = np.array(results[fid]['errors'])
    rows.append({'FuncID': fid, 'Func': CEC2022_INFO[fid][0], 'Type': CEC2022_INFO[fid][2],
                 'Runs': len(e), 'Best': e.min(), 'Worst': e.max(),
                 'Median': np.median(e), 'Mean': e.mean(), 'Std': e.std(ddof=0)})
    for r, v in enumerate(e, 1):
        raw.append({'FuncID': fid, 'Func': CEC2022_INFO[fid][0], 'Run': r, 'Error': v})
p_sum = f'{OUT_DIR}/huem_abc_cec2022_D{D}_summary.csv'
p_raw = f'{OUT_DIR}/huem_abc_cec2022_D{D}_raw_errors.csv'
pd.DataFrame(rows).to_csv(p_sum, index=False); pd.DataFrame(raw).to_csv(p_raw, index=False)
print(f'\nOK — test principal termine en {(time.time()-t0)/3600:.2f} h')
print(f'OK {p_sum}\nOK {p_raw}')
show = pd.DataFrame(rows)[['Func','Type','Best','Worst','Median','Mean','Std']].copy()
for c in ['Best','Worst','Median','Mean','Std']: show[c] = show[c].map('{:.3e}'.format)
print(tabulate(show, headers='keys', tablefmt='grid', showindex=False))

# %%
# Cell 7 — Courbes de convergence (1er run, 6 fonctions representatives)
selected = [1, 3, 7, 9, 11, 12]
fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.flatten()
for ax, fid in zip(axes, selected):
    h = results[fid]['hist']
    ax.plot(np.linspace(0, MaxFEs, len(h)), np.log10(np.maximum(h, 1e-12)), color=C_HUEM, lw=1.4)
    ax.set_title(f'{CEC2022_INFO[fid][0]} — {CEC2022_INFO[fid][1]}', fontsize=9)
    ax.set_xlabel('NFE', fontsize=8); ax.set_ylabel('log10(Error)', fontsize=8)
fig.suptitle(f'Convergence HUEM-ABC — CEC 2022 (D={D})', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_convergence_huem_abc_cec2022.pdf', bbox_inches='tight')
plt.show(); print('OK fig_convergence_huem_abc_cec2022.pdf')

# %%
# Cell 8 — Exports publication : JSON complet, table LaTeX, archive ZIP (compatible ancienne version)
import json, sys, platform, datetime, zipfile, os
import numpy as np, pandas as pd
from tabulate import tabulate

def run_seed(fid, run):
    return run * 137 + fid * 31          # meme formule que la Cell 6

def _jsonable(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f'Non serialisable : {type(o)}')

# Verification : les 30 runs sont termines pour les 12 fonctions
for fid in range(1, 13):
    n = len(results[fid]['errors'])
    assert n == RUNS, f'F{fid} : {n}/{RUNS} runs -> relancer la Cell 6'

# Tableau de synthese (reconstruit depuis results, sans dependre d'autres cellules)
df_main = pd.DataFrame([{
    'FuncID': fid, 'Func': CEC2022_INFO[fid][0], 'Type': CEC2022_INFO[fid][2],
    'Runs': len(results[fid]['errors']),
    'Best': np.min(results[fid]['errors']), 'Worst': np.max(results[fid]['errors']),
    'Median': np.median(results[fid]['errors']), 'Mean': np.mean(results[fid]['errors']),
    'Std': np.std(results[fid]['errors'], ddof=0)} for fid in range(1, 13)])

metadata = {
    'algorithm': 'HUEM-ABC', 'benchmark': 'CEC 2022',
    'dimension': D, 'n_functions': 12, 'bounds': [-100, 100],
    'protocol': {'SN': SN, 'limit': LIMIT, 'MaxFEs': MaxFEs, 'runs': RUNS,
                 'seed_rule': 'seed = run*137 + func_id*31 (run = 0..RUNS-1)'},
    'parameters': {'alpha': ALPHA0, 'hufs_ratio': HR0},
    'error_definition': 'error = max(0, best_f - f_opt)',
    'environment': {'python': sys.version.split()[0], 'numpy': np.__version__,
                    'pandas': pd.__version__, 'platform': platform.platform()},
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
}

functions = {}
for fid in range(1, 13):
    e = np.asarray(results[fid]['errors'], float)
    functions[CEC2022_INFO[fid][0]] = {
        'name': CEC2022_INFO[fid][1], 'type': CEC2022_INFO[fid][2],
        'f_opt': CEC2022_FUNCS[fid]['f_opt'], 'runs': len(e),
        'best': e.min(), 'worst': e.max(), 'median': float(np.median(e)),
        'mean': e.mean(), 'std': e.std(ddof=0),
        'seeds': [run_seed(fid, r) for r in range(len(e))],
        'errors': e,
    }

p_json = f'{OUT_DIR}/huem_abc_cec2022_D{D}_results.json'
with open(p_json, 'w', encoding='utf-8') as fh:
    json.dump({'metadata': metadata, 'functions': functions}, fh,
              indent=2, ensure_ascii=False, default=_jsonable)
print(f'OK {p_json}')

# Table LaTeX (booktabs)
t_main = df_main[['Func', 'Type', 'Best', 'Worst', 'Median', 'Mean', 'Std']].copy()
for c in ['Best', 'Worst', 'Median', 'Mean', 'Std']:
    t_main[c] = t_main[c].map('{:.2e}'.format)
p_tex = f'{OUT_DIR}/table_main_cec2022_D{D}.tex'
with open(p_tex, 'w', encoding='utf-8') as fh:
    fh.write(tabulate(t_main, headers='keys', tablefmt='latex_booktabs', showindex=False))
print(f'OK {p_tex}')

# Archive de depot (sans les checkpoints)
p_zip = f'{OUT_DIR}_publication_bundle.zip'
with zipfile.ZipFile(p_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT_DIR):
        if 'checkpoints' in root: continue
        for fn in files:
            fp = os.path.join(root, fn)
            z.write(fp, os.path.relpath(fp, os.path.dirname(OUT_DIR) or '.'))
print(f'OK {p_zip}')


