import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon, rankdata, friedmanchisquare
from tabulate import tabulate
import warnings, time, os, pickle, glob, threading
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'    : 'serif',
    'font.serif'     : ['Times New Roman', 'DejaVu Serif'],
    'font.size'      : 11,
    'axes.titlesize' : 12,
    'axes.labelsize' : 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
    'axes.grid'      : True,
    'grid.alpha'     : 0.3,
    'grid.linestyle' : '--',
})

C_huem_abc = '#C0392B'

NB_CONFIGS = [
    (1, C_huem_abc, 'huem_abc (nb=1, baseline — test réel)'),
]

print(f'✓ Configuration : nb_size = {[nb for nb,_,_ in NB_CONFIGS]} (test réel unique, pas d\'ablation)')

DIM = 10

N_BEES        = 30
N_RUNS        = 30
MAX_FES       = 200_000 if DIM == 10 else 1_000_000
N_FUNCS       = 7
FUNC_IDS      = [1, 3, 5, 7, 9, 11, 12]
MAX_ITER      = MAX_FES // N_BEES

RESULTS_DIR   = f'results_huem_abc_cec2022_D{DIM}'
BASE_DIR      = rf'C:\huem_abc_CEC2022_alpha07\huem_abc_D{DIM}'
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMS = dict(
    D            = DIM,
    SN           = N_BEES,
    MaxFEs       = MAX_FES,
    MaxIter      = MAX_ITER,
    limit        = 50,
    alpha        = 0.7,
    HUFS_ratio   = 0.2,
    NB_SIZE      = 7,
    RUNS         = N_RUNS,
    N_FUNCS      = N_FUNCS,
    FUNC_IDS     = FUNC_IDS,
    BASE_DIR     = BASE_DIR,
    RESULTS_DIR  = RESULTS_DIR,
)

print('Paramètres expérimentaux — huem_abc (nb_size=1), test réel CEC 2022 :')
for k, v in PARAMS.items():
    print(f'  {k:14s} = {v}')
print()
print(f'  Budget total / run : {MAX_FES:,} évaluations')
print(f'  MaxIter            : {MAX_ITER:,}')
print(f'  Runs / fonction    : {N_RUNS}')
print(f'  Fonctions CEC2022  : F1,F3,F5,F7,F9,F11,F12 (D={DIM})')
print(f'  Répertoire résultats : {RESULTS_DIR}/')

try:
    from opfunu.cec_based import cec2022
except ImportError as e:
    raise ImportError(
        "opfunu est requis pour un TEST RÉEL sur CEC2022. "
        "Installez-le avec : pip install opfunu"
    ) from e

if DIM not in (10, 20):
    raise ValueError(
        f"opfunu ne fournit les fonctions CEC2022 officielles que pour D=10 ou D=20 "
        f"(D={DIM} demandé). Changez DIM dans le script."
    )

CEC2022_INFO = {
    1:  ('F1',  'Shifted and full Rotated Zakharov Function',          'Unimodal'),
    3:  ('F3',  'Shifted and full Rotated Expanded Schaffer F6',       'Basic'),
    5:  ('F5',  'Shifted and full Rotated Levy Function',              'Basic'),
    7:  ('F7',  'Hybrid Function 2 (N=6)',                             'Hybrid'),
    9:  ('F9',  'Composition Function 1 (N=5)',                        'Composition'),
    11: ('F11', 'Composition Function 3 (N=5)',                        'Composition'),
    12: ('F12', 'Composition Function 4 (N=6)',                        'Composition'),
}

def get_cec2022_function(func_id, D):
    cls = getattr(cec2022, f'F{func_id}2022')
    f   = cls(ndim=D)
    return f.evaluate, float(f.f_global), np.asarray(f.lb, dtype=float), np.asarray(f.ub, dtype=float)

CEC2022_FUNCS = {}
for fid in FUNC_IDS:
    fn, f_opt, lb, ub = get_cec2022_function(fid, D=DIM)
    CEC2022_FUNCS[fid] = dict(fn=fn, f_opt=f_opt, lb=lb, ub=ub)
    print(f'  {CEC2022_INFO[fid][0]:4s} ({CEC2022_INFO[fid][2]:11s}) : f_opt = {f_opt:8.1f}  — {CEC2022_INFO[fid][1]}')

print()
print(f'✓ {len(CEC2022_FUNCS)} fonctions CEC2022 OFFICIELLES chargées via opfunu (D={DIM}).')

class huem_abc:

    def __init__(self, func_id, SN, MaxFEs, limit, D,
                 alpha=0.6, hufs_ratio=0.2):
        info = CEC2022_FUNCS[func_id]
        self.fn        = info['fn']
        self.f_opt     = info['f_opt']
        self.lb, self.ub = info['lb'], info['ub']
        self.func_id   = func_id
        self.SN        = SN
        self.MaxFEs    = MaxFEs
        self.limit     = limit
        self.D         = D
        self.alpha     = alpha
        self.hufs_ratio= hufs_ratio

    def _eval(self, x):
        return float(self.fn(x))

    def _fitness_val(self, fval):
        return 1.0 / (1.0 + abs(fval - self.f_opt))

    def _utility(self, pop, fvals):
        fa = np.array([self._fitness_val(f) for f in fvals])
        fa_range = fa.max() - fa.min()
        fn = (fa - fa.min()) / (fa_range + 1e-10)
        d = np.zeros(self.SN)
        for i in range(self.SN):
            di = np.linalg.norm(pop - pop[i], axis=1); di[i] = np.inf
            d[i] = di.min()
        d_range = d.max() - d.min()
        dn = (d - d.min()) / (d_range + 1e-10)
        return self.alpha * fn + (1 - self.alpha) * dn

    @staticmethod
    def _safe_probs(utils):
        s = np.sum(utils)
        if not np.isfinite(s) or s < 1e-10:
            n = len(utils); return np.full(n, 1.0/n)
        pr = utils / s
        return pr / pr.sum()

    def _hufs(self, pop, fvals):
        utils = self._utility(pop, fvals)
        k     = max(1, int(self.SN * self.hufs_ratio))
        idx   = np.argsort(utils)[::-1][:k]
        return pop[idx], fvals[idx]

    def _anchor_shift(self, pop, i, anchor, k_idx, fvals):
        jj = np.random.randint(self.D)
        v  = pop[i].copy()
        v[jj] = np.clip(
            anchor[jj] + np.random.uniform(-1, 1) * (pop[i, jj] - pop[k_idx, jj]),
            self.lb[jj], self.ub[jj])
        return v

    def run(self, seed=None):
        if seed is not None: np.random.seed(seed)
        lb, ub, D, SN = self.lb, self.ub, self.D, self.SN

        pop   = np.random.uniform(lb, ub, (SN, D))
        fvals = np.array([self._eval(x) for x in pop])
        trial = np.zeros(SN, int)
        best_val = fvals.min()
        best_err = best_val - self.f_opt
        NFE = SN
        history = [best_err]

        while NFE < self.MaxFEs:
            hufs, hufs_fv = self._hufs(pop, fvals)
            x_best = hufs[np.argmin(hufs_fv)]

            for i in range(SN):
                if NFE >= self.MaxFEs: break
                anc = hufs[np.random.randint(len(hufs))]
                k   = np.random.choice([j for j in range(SN) if j != i])
                v   = self._anchor_shift(pop, i, anc, k, fvals)
                fv  = self._eval(v)
                NFE += 1
                if fv <= fvals[i]: pop[i]=v; fvals[i]=fv; trial[i]=0
                else: trial[i] += 1
                if fv < best_val: best_val = fv

            utils = self._utility(pop, fvals)
            pr    = self._safe_probs(utils)
            for _ in range(SN):
                if NFE >= self.MaxFEs: break
                i   = np.random.choice(SN, p=pr)
                anc = hufs[np.random.randint(len(hufs))]
                k   = np.random.choice([j for j in range(SN) if j != i])
                v   = self._anchor_shift(pop, i, anc, k, fvals)
                fv  = self._eval(v)
                NFE += 1
                if fv <= fvals[i]: pop[i]=v; fvals[i]=fv; trial[i]=0
                else: trial[i] += 1
                if fv < best_val: best_val = fv

            for i in range(SN):
                if trial[i] > self.limit:
                    if NFE >= self.MaxFEs: break
                    pop[i]   = np.random.uniform(lb, ub, D)
                    fvals[i] = self._eval(pop[i])
                    NFE += 1
                    trial[i] = 0
                    if fvals[i] < best_val: best_val = fvals[i]

            best_err = best_val - self.f_opt
            history.append(best_err)

        return best_err, np.array(history)

print('✓ huem_abc (baseline, nb_size=1) prêt — bornes réelles CEC2022 (lb/ub par fonction).')

results = {}

t0_global = time.time()
for fid in FUNC_IDS:
    fname = CEC2022_INFO[fid][0]
    if fid not in results:
        results[fid] = {'errors': [], 'hist': []}

    n_done = len(results[fid]['errors'])
    if n_done >= N_RUNS:
        print(f'{fname} : déjà {n_done}/{N_RUNS} runs — passage à la fonction suivante.')
        continue

    print(f'\n{fname} — démarrage ({n_done}/{N_RUNS} runs)...')
    for run_idx in range(n_done, N_RUNS):
        t0 = time.time()
        opt = huem_abc(
            func_id=fid, SN=PARAMS['SN'], MaxFEs=PARAMS['MaxFEs'],
            limit=PARAMS['limit'], D=PARAMS['D'],
            alpha=PARAMS['alpha'], hufs_ratio=PARAMS['HUFS_ratio'],
        )
        best_err, hist = opt.run(seed=run_idx)
        results[fid]['errors'].append(best_err)
        results[fid]['hist'].append(hist)
        dt = time.time() - t0
        print(f'  run {run_idx+1:2d}/{N_RUNS} — err={best_err:.6e} — {dt:.1f}s')

    print(f'✓ {fname} terminé.')

print(f'\n✓✓✓ TEST RÉEL TERMINÉ — {time.time()-t0_global:.1f}s au total.')

with open(os.path.join(RESULTS_DIR, f'huem_abc_cec2022_D{DIM}.pkl'), 'wb') as fh:
    pickle.dump(results, fh)
print(f'✓ Résultats pickle sauvegardés → {RESULTS_DIR}/huem_abc_cec2022_D{DIM}.pkl')

rows = []
for fid in FUNC_IDS:
    fname = CEC2022_INFO[fid][0]
    errs = np.array(results[fid]['errors'])
    rows.append(dict(
        Function=fname,
        Mean=errs.mean(), Std=errs.std(),
        Best=errs.min(), Worst=errs.max(),
        Median=np.median(errs), N_runs=len(errs),
    ))

df_summary = pd.DataFrame(rows)
pd.set_option('display.float_format', lambda x: f'{x:.4e}')
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(RESULTS_DIR, f'huem_abc_cec2022_D{DIM}.csv'), index=False)
print(f'\n✓ Résumé sauvegardé → {RESULTS_DIR}/huem_abc_cec2022_D{DIM}.csv')

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for idx, fid in enumerate(FUNC_IDS):
    ax = axes.flat[idx]
    hists = results[fid]['hist']
    max_len = max(len(h) for h in hists)
    padded = np.array([np.pad(h, (0, max_len - len(h)), mode='edge') for h in hists])
    mean_curve = padded.mean(axis=0)
    ax.plot(np.clip(mean_curve, 1e-12, None), color=C_huem_abc, lw=1.2)
    ax.set_yscale('log')
    ax.set_title(CEC2022_INFO[fid][0], fontsize=10)
    ax.set_xlabel('Itération'); ax.set_ylabel('Erreur')
for idx in range(len(FUNC_IDS), 12):
    axes.flat[idx].axis('off')
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'convergence_nb1.pdf')
plt.savefig(fig_path)
plt.show()
print(f'✓ Courbes de convergence sauvegardées → {fig_path}')

def to_latex_booktabs(df, caption, label):
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{{label}}}')
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append('Function & Mean & Std & Best & Worst & Median \\\\')
    lines.append(r'\midrule')
    for _, r in df.iterrows():
        lines.append(
            f"{r['Function']} & {r['Mean']:.2e} & {r['Std']:.2e} & "
            f"{r['Best']:.2e} & {r['Worst']:.2e} & {r['Median']:.2e} \\\\"
        )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

tex = to_latex_booktabs(
    df_summary,
    caption=f'huem_abc (nb\\_size=1) — CEC2022, D={DIM}, {N_RUNS} runs — test réel (opfunu)',
    label=f'tab:huem_abc_nb1_D{DIM}_real',
)
tex_path = os.path.join(RESULTS_DIR, f'results_huem_abc_nb1_D{DIM}_real.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(tex)
print(tex)
print(f'\n✓ Table LaTeX sauvegardée → {tex_path}')
