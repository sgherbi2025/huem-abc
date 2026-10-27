import argparse
import glob
import math
import os
import pickle
import signal
import time
import warnings
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare, rankdata
from tabulate import tabulate

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

try:
    from opfunu.cec_based import cec2013
    from opfunu.utils import operator as _opf_operator
except ImportError as e:
    raise RuntimeError("opfunu est requis (pip install opfunu) : aucun fallback proxy n'est autorisé.") from e

D = 30
RES_DIR = f'results_huem_abc_CEC2013_D{D}'
SENS_DIR = f'{RES_DIR}/sensitivity'
CKPT_INTERVAL = 3600
CKPT_MAIN = os.path.join(RES_DIR, 'checkpoint_main.pkl')
CKPT_SENS = os.path.join(RES_DIR, 'checkpoint_sens.pkl')

PARAMS = dict(
    D=D,
    SN=50,
    MaxFEs=10_000 * D,
    limit=100,
    alpha=0.7,
    HUFS_ratio=0.2,
    RUNS=30,
    N_FUNCS=28,
)

ALPHA_VALUES = [0.3, 0.5, 0.6, 0.7, 0.9]
HUFS_RATIOS = [0.1, 0.2, 0.3]
SENS_FUNC_IDS = list(range(1, 29))
SENS_RUNS = 30
COMBOS = [(a, h) for a in ALPHA_VALUES for h in HUFS_RATIOS]
DEFAULT_COMBO = (PARAMS['alpha'], PARAMS['HUFS_ratio'])

CEC2013_INFO = {
    1: ('F1', 'Sphere', 'Unimodal'),
    2: ('F2', 'Schwefel 1.2', 'Unimodal'),
    3: ('F3', 'Rotated Elliptic', 'Unimodal'),
    4: ('F4', 'Schwefel 1.2 Noise', 'Unimodal'),
    5: ('F5', 'Schwefel 2.6', 'Unimodal'),
    6: ('F6', 'Shifted Rosenbrock', 'Multimodal'),
    7: ('F7', 'Shifted Griewank', 'Multimodal'),
    8: ('F8', 'Shifted Ackley', 'Multimodal'),
    9: ('F9', 'Shifted Rastrigin', 'Multimodal'),
    10: ('F10', 'Shifted Rastrigin Rot.', 'Multimodal'),
    11: ('F11', 'Shifted Weierstrass', 'Multimodal'),
    12: ('F12', 'Schwefel 2.13', 'Multimodal'),
    13: ('F13', 'Expanded Griewank Rosenbrock', 'Multimodal'),
    14: ('F14', 'Expanded Scaffer F6', 'Multimodal'),
    15: ('F15', 'Hybrid Composition 1', 'Hybrid'),
    16: ('F16', 'Hybrid Composition 2', 'Hybrid'),
    17: ('F17', 'Hybrid Composition 3', 'Hybrid'),
    18: ('F18', 'Hybrid Composition 4', 'Hybrid'),
    19: ('F19', 'Rotated Hybrid Composition 1', 'Hybrid'),
    20: ('F20', 'Rotated Hybrid Composition 2', 'Hybrid'),
    21: ('F21', 'Rotated Hybrid Composition 3', 'Hybrid'),
    22: ('F22', 'Rotated Hybrid Composition 4', 'Hybrid'),
    23: ('F23', 'Non-Continuous Hybrid', 'Hybrid'),
    24: ('F24', 'Rotated Non-Continuous', 'Hybrid'),
    25: ('F25', 'Scaffer Hybrid', 'Hybrid'),
    26: ('F26', 'Rotated Scaffer Hybrid', 'Hybrid'),
    27: ('F27', 'Expanded Hybrid 1', 'Hybrid'),
    28: ('F28', 'Expanded Hybrid 2', 'Hybrid'),
}

PLATEAU_FUNCS = {8: 'Ackley tourné : erreur ≈ 2.0e1', 20: 'Expanded Schaffer F6 : erreur ≈ D/2'}

CEC2013_FUNCS = {}


def save_checkpoint(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as fh:
                obj = pickle.load(fh)
            print(f'✓ Checkpoint rechargé : {path}')
            return obj
        except Exception as e:
            print(f'⚠ Checkpoint illisible ({e}) — on repart de zéro.')
    return None


class HourlyCheckpointer:
    def __init__(self, path, interval=CKPT_INTERVAL):
        self.path = path
        self.interval = interval
        self.last = time.time()

    def tick(self, obj, force=False):
        if force or (time.time() - self.last) >= self.interval:
            save_checkpoint(obj, self.path)
            self.last = time.time()
            print(f'  💾 checkpoint sauvegardé ({time.strftime("%H:%M:%S")}) → {self.path}', flush=True)


def _modified_schwefel_cec2013(x):
    z = np.asarray(x, float).ravel() + 4.209687462275036e+002
    nx = z.size
    fx = np.zeros(nx)
    m1 = z > 500
    m2 = z < -500
    m3 = ~m1 & ~m2
    t = np.fmod(z[m1], 500.0)
    fx[m1] = -(500.0 - t) * np.sin(np.sqrt(np.abs(500.0 - t))) + ((z[m1] - 500.0) / 100.0) ** 2 / nx
    t = np.fmod(np.abs(z[m2]), 500.0)
    fx[m2] = -(-500.0 + t) * np.sin(np.sqrt(np.abs(500.0 - t))) + ((z[m2] + 500.0) / 100.0) ** 2 / nx
    fx[m3] = -z[m3] * np.sin(np.sqrt(np.abs(z[m3])))
    return float(np.sum(fx) + 4.189828872724338e+002 * nx)


def get_cec2013_function(func_id, dim=30):
    f = getattr(cec2013, f'F{func_id}2013')(ndim=dim)
    lb, ub = float(f.lb[0]), float(f.ub[0])
    return f.evaluate, float(f.f_global), (lb, ub)


def load_functions():
    _opf_operator.modified_schwefel_func = _modified_schwefel_cec2013
    print('✓ Correctif Schwefel modifié (F14, F15) appliqué.')
    for fid in range(1, 29):
        fn, fopt, bounds = get_cec2013_function(fid, D)
        CEC2013_FUNCS[fid] = {
            'fn': fn, 'f_opt': fopt, 'bounds': bounds,
            'name': CEC2013_INFO[fid][0],
            'desc': CEC2013_INFO[fid][1],
            'type': CEC2013_INFO[fid][2],
        }
    print(f'✓ {len(CEC2013_FUNCS)} fonctions CEC 2013 chargées (D={D}).')


def validate_functions():
    print('\nValidation des fonctions CEC 2013 :')
    rng = np.random.RandomState(0)
    bad = []
    for fid, v in CEC2013_FUNCS.items():
        fn, fopt = v['fn'], v['f_opt']
        lb, ub = v['bounds']
        f_at_opt = fn(getattr(cec2013, f'F{fid}2013')(ndim=D).x_global)
        rand_vals = [fn(rng.uniform(lb, ub, D)) for _ in range(20)]
        ok_opt = abs(f_at_opt - fopt) < 1e-6
        ok_fin = np.all(np.isfinite(rand_vals)) and min(rand_vals) >= fopt - 1e-6
        varies = np.ptp(rand_vals) > 1e-6
        tag = 'OK' if (ok_opt and ok_fin) else '⚠ PROBLÈME'
        if not (ok_opt and ok_fin):
            bad.append(fid)
        if not varies:
            if fid in PLATEAU_FUNCS:
                tag += ' (plateau attendu : ' + PLATEAU_FUNCS[fid] + ')'
            else:
                tag += ' ⚠ CONSTANTE INATTENDUE'
                bad.append(fid)
        print(f'  F{fid:<2d} f_global={fopt:8.1f}  f(x*)={f_at_opt:9.2f}  '
              f'min/max(20 pts aléatoires)=[{min(rand_vals):.6g}, {max(rand_vals):.6g}]  {tag}')
    assert not bad, f'Fonctions invalides : {sorted(set(bad))}'
    print('✓ Toutes les fonctions validées.')


class huem_abc:
    def __init__(self, func_id, SN, MaxFEs, limit, D, alpha=0.7, hufs_ratio=0.2):
        info = CEC2013_FUNCS[func_id]
        self.fn = info['fn']
        self.f_opt = info['f_opt']
        self.lb, self.ub = info['bounds']
        self.func_id, self.SN, self.MaxFEs = func_id, SN, MaxFEs
        self.limit, self.D = limit, D
        self.alpha, self.hufs_ratio = alpha, hufs_ratio

    def _eval(self, x):
        return float(self.fn(x))

    def _utility(self, pop, fvals, f_star):
        q = 1.0 / (1.0 + np.abs(fvals - f_star))
        qn = (q - q.min()) / (q.max() - q.min() + 1e-10)
        dm = cdist(pop, pop)
        np.fill_diagonal(dm, np.inf)
        d = dm.min(axis=1)
        dn = (d - d.min()) / (d.max() - d.min() + 1e-10)
        return self.alpha * qn + (1 - self.alpha) * dn

    @staticmethod
    def _safe_probs(utils):
        s = np.sum(utils)
        if not np.isfinite(s) or s < 1e-10:
            n = len(utils)
            return np.full(n, 1.0 / n)
        pr = utils / s
        return pr / pr.sum()

    def _hufs(self, pop, fvals, f_star):
        utils = self._utility(pop, fvals, f_star)
        k = max(1, math.ceil(round(self.SN * self.hufs_ratio, 9)))
        idx = np.argsort(-utils)[:k]
        return pop[idx].copy(), fvals[idx].copy()

    def _generate(self, pop, i, anchor):
        j = np.random.randint(self.D)
        k = np.random.randint(self.SN - 1)
        k += (k >= i)
        r = np.random.uniform(-1, 1)
        v = pop[i].copy()
        v[j] = np.clip(anchor[j] + r * (pop[i, j] - pop[k, j]), self.lb, self.ub)
        return v

    def _try(self, pop, fvals, trial, i, anchor):
        v = self._generate(pop, i, anchor)
        fv = self._eval(v)
        if fv <= fvals[i]:
            pop[i] = v
            fvals[i] = fv
            trial[i] = 0
        else:
            trial[i] += 1
        return fv

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        lb, ub, dim, SN = self.lb, self.ub, self.D, self.SN

        pop = np.random.uniform(lb, ub, (SN, dim))
        fvals = np.array([self._eval(x) for x in pop])
        trial = np.zeros(SN, int)
        best_val = fvals.min()
        f_star = best_val
        NFE = SN
        history = [max(0.0, best_val - self.f_opt)]

        while NFE < self.MaxFEs:
            hufs, hufs_fv = self._hufs(pop, fvals, f_star)
            ib = int(np.argmin(hufs_fv))
            x_best = hufs[ib].copy()
            f_star = float(hufs_fv[ib])

            for i in range(SN):
                if NFE >= self.MaxFEs:
                    break
                anc = hufs[np.random.randint(len(hufs))]
                fv = self._try(pop, fvals, trial, i, anc)
                NFE += 1
                if fv < best_val:
                    best_val = fv

            pr = self._safe_probs(self._utility(pop, fvals, f_star))
            for _ in range(SN):
                if NFE >= self.MaxFEs:
                    break
                i = np.random.choice(SN, p=pr)
                anc = hufs[np.random.randint(len(hufs))]
                fv = self._try(pop, fvals, trial, i, anc)
                NFE += 1
                if fv < best_val:
                    best_val = fv

            for i in range(SN):
                if NFE >= self.MaxFEs:
                    break
                if trial[i] > self.limit:
                    r1, r2, r3, r4 = np.random.uniform(0, 1, 4)
                    xr = pop[np.random.randint(SN)]
                    pop[i] = np.clip(r1 * x_best + r2 * xr + r3 * (lb + r4 * (ub - lb)), lb, ub)
                    fvals[i] = self._eval(pop[i])
                    NFE += 1
                    trial[i] = 0
                    if fvals[i] < best_val:
                        best_val = fvals[i]

            history.append(max(0.0, best_val - self.f_opt))

        self.raw_err = best_val - self.f_opt
        return max(0.0, self.raw_err), np.array(history)


def sanity_check():
    print('Sanity check — F1..F28 (MaxFEs=2000, SN=10, 1 run) :')
    viol = []
    for fid in range(1, 29):
        o = huem_abc(fid, SN=10, MaxFEs=2000, limit=10, D=D)
        e, _ = o.run(seed=0)
        flag = '' if o.raw_err >= -1e-6 else '  ⚠ ERREUR NÉGATIVE'
        if o.raw_err < -1e-6:
            viol.append(fid)
        print(f'  F{fid:2d} ({CEC2013_INFO[fid][2]:10s}) : err = {e:.2e}{flag}')
    assert not viol, f'Valeur < f_global : {viol}'
    print('✓ Sanity check OK.')


def export_main_csv(results, verbose=False):
    raw_rows, stat_rows = [], []
    for fid in sorted(results):
        errs = np.array(results[fid]['errors'], float)
        if len(errs) == 0:
            continue
        fname, ftype = CEC2013_INFO[fid][0], CEC2013_INFO[fid][2]
        for r, e in enumerate(errs):
            raw_rows.append({'FuncID': fid, 'Func': fname, 'Type': ftype, 'Run': r + 1, 'Error': e})
        stat_rows.append({
            'FuncID': fid, 'Func': fname, 'Desc': CEC2013_INFO[fid][1], 'Type': ftype,
            'Runs': len(errs), 'Best': errs.min(), 'Worst': errs.max(),
            'Median': np.median(errs), 'Mean': errs.mean(), 'Std': errs.std(ddof=0),
        })
    df_raw, df_stat = pd.DataFrame(raw_rows), pd.DataFrame(stat_rows)
    df_raw.to_csv(f'{RES_DIR}/huem_abc_cec2013_runs_errors.csv', index=False)
    df_stat.to_csv(f'{RES_DIR}/huem_abc_cec2013_stats.csv', index=False)
    if verbose:
        print(f'  📄 CSV mis à jour ({len(df_stat)} fonctions, {len(df_raw)} runs)')
    return df_raw, df_stat


def run_main_experiment(results, ckpt_main):
    RUNS, SN, limit_ = PARAMS['RUNS'], PARAMS['SN'], PARAMS['limit']
    alpha_, hr, MaxFEs_ = PARAMS['alpha'], PARAMS['HUFS_ratio'], PARAMS['MaxFEs']
    t0 = time.time()
    try:
        for fid in range(1, 29):
            fname, ftype = CEC2013_INFO[fid][0], CEC2013_INFO[fid][2]
            if fid not in results:
                results[fid] = {'errors': [], 'hist': None}

            if len(results[fid]['errors']) >= RUNS:
                e = results[fid]['errors']
                print(f'{fname} ({ftype:10s}) [complet] mean={np.mean(e):.2e}  std={np.std(e):.2e}')
                continue

            t_f = time.time()
            for run in range(RUNS):
                if len(results[fid]['errors']) > run:
                    continue
                seed = run * 137 + fid * 31
                opt = huem_abc(fid, SN=SN, MaxFEs=MaxFEs_, limit=limit_, D=D, alpha=alpha_, hufs_ratio=hr)
                err, hist = opt.run(seed=seed)
                if opt.raw_err < -1e-6:
                    print(f'  ⚠ F{fid}: erreur brute négative {opt.raw_err:.3e}')
                results[fid]['errors'].append(float(err))
                if results[fid]['hist'] is None:
                    results[fid]['hist'] = hist
                ckpt_main.tick(results)
                if (run + 1) % 10 == 0 or (run + 1) == RUNS:
                    print(f'  {fname} ({ftype:10s}) run {run + 1:2d}/{RUNS}  '
                          f'mean={np.mean(results[fid]["errors"]):.2e}', flush=True)

            e = results[fid]['errors']
            print(f'{fname} TERMINÉ [{time.time() - t_f:.0f}s]  mean={np.mean(e):.2e}  std={np.std(e):.2e}')
            ckpt_main.tick(results, force=True)
            export_main_csv(results, verbose=True)
    except KeyboardInterrupt:
        print('\n⏸ Interruption — sauvegarde checkpoint + CSV...')
        ckpt_main.tick(results, force=True)
        export_main_csv(results, verbose=True)
        raise
    finally:
        ckpt_main.tick(results, force=True)
    print(f'\n✓ Expérience principale terminée en {(time.time() - t0) / 3600:.2f} h')


def export_part1(results, ckpt_main):
    df_raw, df_stat = export_main_csv(results, verbose=True)
    ckpt_main.tick(results, force=True)
    if df_stat.empty:
        print('Aucun résultat à exporter.')
        return df_stat

    cols = ['Func', 'Type', 'Best', 'Worst', 'Median', 'Mean', 'Std']
    df_stat[cols].to_csv(f'{RES_DIR}/cec_format_errors.csv', index=False)

    show = df_stat[cols].copy()
    for c in ['Best', 'Worst', 'Median', 'Mean', 'Std']:
        show[c] = show[c].map('{:.3e}'.format)
    print(tabulate(show, headers='keys', tablefmt='grid', showindex=False))

    print('\nFichiers de la Partie 1 :')
    for f in [f'{RES_DIR}/huem_abc_cec2013_runs_errors.csv', f'{RES_DIR}/huem_abc_cec2013_stats.csv',
              f'{RES_DIR}/cec_format_errors.csv', CKPT_MAIN]:
        print(f'  {f:70s} {os.path.getsize(f) / 1024:8.1f} Ko')
    return df_stat


def plot_convergence(results):
    sel = [f for f in [1, 6, 9, 15, 21, 28] if f in results and results[f]['hist'] is not None]
    if not sel:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, fid in zip(axes, sel):
        h = results[fid]['hist']
        ax.plot(np.linspace(0, PARAMS['MaxFEs'], len(h)), np.log10(np.maximum(h, 1e-10)),
                color='#C0392B', lw=1.5)
        ax.set_title(f'{CEC2013_INFO[fid][0]} — {CEC2013_INFO[fid][1][:25]}', fontsize=9)
        ax.set_xlabel('NFE')
        ax.set_ylabel('log₁₀(Error)')
    fig.suptitle(f'Convergence HUEM-ABC · CEC 2013 (D={D})', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{RES_DIR}/fig_convergence.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ fig_convergence.pdf')


def fmt_e(v, prec=2):
    return f'{v:.{prec}e}'


def write_latex_table(df_stat):
    if df_stat.empty:
        return
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Results of HUEM-ABC on CEC 2013 (D=' + str(D) + r', ' + str(PARAMS['RUNS']) + r' runs)}',
             r'\label{tab:huem_abc_cec2013}', r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Func & Type & Best & Worst & Median & Mean & Std \\']
    prev = None
    for _, r in df_stat.iterrows():
        if r['Type'] != prev:
            lines += [r'\midrule', r'\multicolumn{7}{l}{\textit{' + r['Type'] + r'}} \\']
            prev = r['Type']
        lines.append(f"{r['Func']} & {r['Type']} & {fmt_e(r['Best'])} & {fmt_e(r['Worst'])} & "
                     f"{fmt_e(r['Median'])} & {fmt_e(r['Mean'])} & {fmt_e(r['Std'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(f'{RES_DIR}/latex_table_cec2013.tex', 'w') as fh:
        fh.write('\n'.join(lines))
    print('✓ latex_table_cec2013.tex')


def run_sensitivity(results, sens_results, ckpt_sens):
    SN, limit_ = PARAMS['SN'], PARAMS['limit']
    MaxFEs_ = PARAMS['MaxFEs']
    print('Analyse de sensibilité — HUEM-ABC :')
    print(f'  α ∈ {ALPHA_VALUES}   hufs_ratio ∈ {HUFS_RATIOS}   → {len(COMBOS)} combinaisons')
    t0 = time.time()
    try:
        for fid in SENS_FUNC_IDS:
            fname, ftype = CEC2013_INFO[fid][0], CEC2013_INFO[fid][2]
            if fid not in sens_results:
                sens_results[fid] = {c: {'errors': []} for c in COMBOS}
                if fid in results and len(results[fid]['errors']) >= SENS_RUNS:
                    sens_results[fid][DEFAULT_COMBO]['errors'] = list(results[fid]['errors'][:SENS_RUNS])

            if all(len(sens_results[fid][c]['errors']) >= SENS_RUNS for c in COMBOS):
                print(f'{fname} ({ftype:10s}) [complet]')
                continue

            t_f = time.time()
            for run in range(SENS_RUNS):
                seed = run * 137 + fid * 31
                for (a, h) in COMBOS:
                    if len(sens_results[fid][(a, h)]['errors']) > run:
                        continue
                    opt = huem_abc(fid, SN=SN, MaxFEs=MaxFEs_, limit=limit_, D=D, alpha=a, hufs_ratio=h)
                    err, _ = opt.run(seed=seed)
                    if opt.raw_err < -1e-6:
                        print(f'  ⚠ F{fid} (α={a}, hufs={h}) : erreur brute négative {opt.raw_err:.3e}')
                    sens_results[fid][(a, h)]['errors'].append(float(err))
                ckpt_sens.tick(sens_results)
                if (run + 1) % 5 == 0 or (run + 1) == SENS_RUNS:
                    print(f'  {fname} ({ftype:10s}) run {run + 1:2d}/{SENS_RUNS}', flush=True)

            means = {c: np.mean(sens_results[fid][c]['errors']) for c in COMBOS}
            best_c = min(means, key=means.get)
            print(f'{fname} TERMINÉ [{time.time() - t_f:.0f}s]  '
                  f'best=(α={best_c[0]}, hufs={best_c[1]}) → {means[best_c]:.2e}')
            ckpt_sens.tick(sens_results, force=True)
    except KeyboardInterrupt:
        print('\n⏸ Interruption — sauvegarde du checkpoint...')
        ckpt_sens.tick(sens_results, force=True)
        raise
    finally:
        ckpt_sens.tick(sens_results, force=True)
    print(f'\n✓ Sensibilité terminée en {(time.time() - t0) / 3600:.2f} h')


def analyze_sensitivity(sens_results):
    rows = []
    for fid in SENS_FUNC_IDS:
        if fid not in sens_results:
            continue
        if any(len(sens_results[fid][c]['errors']) == 0 for c in COMBOS):
            continue
        means = [np.mean(sens_results[fid][c]['errors']) for c in COMBOS]
        ranks = rankdata(means)
        row = {'FuncID': fid, 'Name': CEC2013_INFO[fid][0], 'Type': CEC2013_INFO[fid][2]}
        for (a, h), m, r in zip(COMBOS, means, ranks):
            row[f'mean_a{a}_h{h}'] = m
            row[f'rank_a{a}_h{h}'] = r
        rows.append(row)
    if not rows:
        print('Aucun résultat de sensibilité exploitable.')
        return

    df_sens = pd.DataFrame(rows)
    avg_rank_grid = np.array([[df_sens[f'rank_a{a}_h{h}'].mean() for h in HUFS_RATIOS] for a in ALPHA_VALUES])
    best_i, best_j = np.unravel_index(np.argmin(avg_rank_grid), avg_rank_grid.shape)

    print('RANG MOYEN DE FRIEDMAN — grille α × hufs_ratio')
    print('  α \\ hufs | ' + ' | '.join(f'{h:>6.1f}' for h in HUFS_RATIOS))
    for i, a in enumerate(ALPHA_VALUES):
        print(f'  α = {a:<4.1f}  | ' + ' | '.join(f'{avg_rank_grid[i, j]:6.2f}' for j in range(len(HUFS_RATIOS))))
    print(f'\n✓ Meilleure combinaison : α={ALPHA_VALUES[best_i]}, hufs_ratio={HUFS_RATIOS[best_j]} '
          f'(rang moyen={avg_rank_grid[best_i, best_j]:.3f})')

    try:
        samples = [[df_sens.loc[df_sens['FuncID'] == f, f'mean_a{a}_h{h}'].values[0] for f in df_sens['FuncID']]
                   for (a, h) in COMBOS]
        stat, p = friedmanchisquare(*samples)
        print(f'Friedman χ² = {stat:.3f}, p = {p:.4e} ({"significatif" if p < 0.05 else "non significatif"} à 0.05)')
    except Exception as e:
        print(f'Friedman non calculable : {e}')

    df_sens.to_csv(f'{SENS_DIR}/sensitivity_full_stats.csv', index=False)
    pd.DataFrame(avg_rank_grid, index=[f'alpha={a}' for a in ALPHA_VALUES],
                 columns=[f'hufs={h}' for h in HUFS_RATIOS]).to_csv(f'{SENS_DIR}/sensitivity_avg_rank_grid.csv')
    print(f'✓ CSV → {SENS_DIR}/')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    im = ax.imshow(avg_rank_grid, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(COMBOS))
    ax.set_xticks(range(len(HUFS_RATIOS)))
    ax.set_xticklabels([f'{h:.1f}' for h in HUFS_RATIOS])
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels([f'{a:.1f}' for a in ALPHA_VALUES])
    ax.set_xlabel('hufs_ratio')
    ax.set_ylabel('α')
    ax.set_title(f'Rang moyen de Friedman (n={len(df_sens)} fonctions)')
    for i in range(len(ALPHA_VALUES)):
        for j in range(len(HUFS_RATIOS)):
            ax.text(j, i, f'{avg_rank_grid[i, j]:.2f}', ha='center', va='center')
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor='black', lw=2.5))
    plt.colorbar(im, ax=ax)

    ax2 = axes[1]
    log_grid = np.array([[np.mean(np.log10(np.maximum(df_sens[f'mean_a{a}_h{h}'], 1e-12)))
                          for h in HUFS_RATIOS] for a in ALPHA_VALUES])
    im2 = ax2.imshow(log_grid, cmap='RdYlGn_r', aspect='auto')
    ax2.set_xticks(range(len(HUFS_RATIOS)))
    ax2.set_xticklabels([f'{h:.1f}' for h in HUFS_RATIOS])
    ax2.set_yticks(range(len(ALPHA_VALUES)))
    ax2.set_yticklabels([f'{a:.1f}' for a in ALPHA_VALUES])
    ax2.set_xlabel('hufs_ratio')
    ax2.set_ylabel('α')
    ax2.set_title('Moyenne de log₁₀(erreur)')
    for i in range(len(ALPHA_VALUES)):
        for j in range(len(HUFS_RATIOS)):
            ax2.text(j, i, f'{log_grid[i, j]:.2f}', ha='center', va='center')
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(f'{SENS_DIR}/fig_sensitivity_heatmaps.pdf', bbox_inches='tight')
    plt.close(fig)
    print('✓ fig_sensitivity_heatmaps.pdf')


def make_archive():
    zip_path = 'huem_abc_CEC2013_results.zip'
    files = [f for f in sorted(glob.glob(f'{RES_DIR}/**/*.*', recursive=True)) if os.path.isfile(f)]
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f)
    print(f'✓ Archive → {zip_path}')
    for f in files:
        print(f'  {f:75s} ({os.path.getsize(f) / 1024:.1f} Ko)')


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='HUEM-ABC — CEC 2013 (D=30)')
    parser.add_argument('--part', choices=['1', '2', 'all'], default='all')
    parser.add_argument('--skip-checks', action='store_true')
    parser.add_argument('--no-zip', action='store_true')
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(SENS_DIR, exist_ok=True)

    print('Paramètres expérimentaux — HUEM-ABC, CEC 2013 :')
    for k, v in PARAMS.items():
        print(f'  {k:12s} = {v}')
    print(f'  Budget total / run : {PARAMS["MaxFEs"]:,} évaluations')
    print(f'✓ Checkpoint toutes les {CKPT_INTERVAL // 60} min — fichiers dans {RES_DIR}/')

    load_functions()
    if not args.skip_checks:
        validate_functions()
        sanity_check()

    results = load_checkpoint(CKPT_MAIN) or {}
    ckpt_main = HourlyCheckpointer(CKPT_MAIN)

    if args.part in ('1', 'all'):
        run_main_experiment(results, ckpt_main)
        df_stat = export_part1(results, ckpt_main)
        plot_convergence(results)
        write_latex_table(df_stat)

    if args.part in ('2', 'all'):
        sens_results = load_checkpoint(CKPT_SENS) or {}
        ckpt_sens = HourlyCheckpointer(CKPT_SENS)
        run_sensitivity(results, sens_results, ckpt_sens)
        analyze_sensitivity(sens_results)

    if not args.no_zip:
        make_archive()


if __name__ == '__main__':
    main()
