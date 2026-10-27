#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the CEC2013 tables (docx) and Friedman figures (png) comparing
HUEM-ABC with its variants (ABC, ABC-Scout, ABC-HUFS).

Requirements:
    pip install numpy scipy pandas matplotlib python-docx

Usage (from anywhere):
    python generate_experiments_tables_cec2013_huem_abc_variants.py

Expected layout (the script directory is the repository root, or a
'results' / 'results/cec2013' sub-directory of it):
    results/cec2013/<algo>_cec2013_D<dim>.csv
Outputs are written to:
    experiments/
"""
import os
import re
import glob
from math import floor, log10

import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = "Times New Roman"
SIZE = 11
NBEST = 5
ALPHA = 0.05          # significance level for Wilcoxon / Friedman
FIG_DPI = 300

# --- Comparison with other algorithms ---
COMP_ALGOS = ["ABC", "ABC-Scout", "ABC-HUFS", "HUEM-ABC"]   # column order
REF_ALGO = "HUEM-ABC"     # reference algorithm for the Wilcoxon test
CFG_TABLES_TXT = ""         # e.g. "Tables 8, 9 and 10" (set in main)
HUEM_FROM_CONFIGS = False   # set in main: True if huem_abc comes from mean_a*_h* configurations
# cec2013 benchmark dimensions: a dimension without CSV files is simply skipped.
DIMS = [30, 50]
FIRST_TABLE = 4      # tables are numbered from here (after the CEC2013 tables 4-7)
FIRST_FIGURE = 3      # figures are numbered from here (after CEC2013 figures 1-2)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # e.g. executed from an interactive session
    BASE_DIR = os.getcwd()

if BASE_DIR.endswith(os.path.join("results", "cec2013")):
    ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
elif BASE_DIR.endswith("results"):
    ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
else:
    ROOT = BASE_DIR

RESULTS = os.path.join(ROOT, "results", "cec2013")
OUTDIR = os.path.join(ROOT, "experiments")

CFG_RE = re.compile(r"^(mean|std|rank)_a([\d.]+)_h([\d.]+)$")

MINUS = "\u2212"   # −
APPROX = "\u2248"  # ≈


def cfg_label(cfg):
    a, h = cfg
    return f"α={a:g}, h={h:g}"


# --------------------------------------------------------------------------- #
# Formatting helpers (docx)
# --------------------------------------------------------------------------- #
def style_run(run, bold=None, italic=None, superscript=None):
    run.font.name = FONT
    run.font.size = Pt(SIZE)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if superscript is not None:
        run.font.superscript = superscript


# --------------------------------------------------------------------------- #
# Number formatting: (m.mm±s.ss)×10^e  /  2.03E+01 ± 2.57E-02
# --------------------------------------------------------------------------- #
def fmt_mantissa(v):
    """2 decimals if |v| < 10, otherwise 1 decimal (20.8 -> '20.8', 4.42 -> '4.42')."""
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def mantissa_exp(x):
    exp = floor(log10(abs(x)))
    mant = round(x / 10 ** exp, 2)
    if abs(mant) >= 10:
        mant /= 10
        exp += 1
    return mant, exp


SUP_DIGITS = {"-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2",
              "3": "\u00b3", "4": "\u2074", "5": "\u2075", "6": "\u2076",
              "7": "\u2077", "8": "\u2078", "9": "\u2079"}


def sup_exp(e):
    """Exponent as unicode superscript string: -7 -> '⁻⁷', 1 -> '¹'."""
    return "".join(SUP_DIGITS[c] for c in str(int(e)))


def fmt_sig2(v):
    """2 significant digits for the std mantissa (52.32, 0.016, 0.00000017)."""
    if v == 0:
        return "0.00"
    s = f"{v:.2g}"                    # 2 significant digits, may give '1.7e-07'
    if "e" in s or "E" in s:          # expand small values to plain decimal
        s = f"{float(s):.12f}".rstrip("0").rstrip(".")
    return s


E_FALLBACK = 0.01   # |std mantissa| below this -> notation 2.03E+01 instead of (m±s)×10^e


def e_notation_segments(mean, std, bold):
    segs = [(f"{mean:.2E}", bold, False)]
    if pd.notna(std):
        segs.append((" ± " + f"{std:.2E}", bold, False))
    return segs


def cell_segments(mean, std, bold):
    if pd.isna(mean):
        return [("--", bold, False)]

    if mean == 0:
        m, e = 0.0, 0
    else:
        m, e = mantissa_exp(mean)

    if pd.notna(std):
        s = std / 10 ** e
        return [(f"({m:.2f}±{s:.2f})×10{sup_exp(e)}", bold, False)]

    return [(f"{m:.2f}×10{sup_exp(e)}", bold, False)]


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
def _id_column(df):
    """Name of the column holding the function number (Function, func_id, F, ...)."""
    return next((c for c in df.columns
                 if c.strip().lower() in ("f", "fid", "id") or c.strip().lower().startswith("func")), None)


def function_ids(df):
    """Function numbers (1..12 for cec2013) from the id column, else 1..n."""
    col = _id_column(df)
    if col is not None:
        ids = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        if ids.notna().all():
            return ids.astype(int).to_numpy()
    return np.arange(1, len(df) + 1)


def parse_config_columns(df):
    configs = {}
    for col in df.columns:
        m = CFG_RE.match(col)
        if m:
            kind, a, h = m.group(1), float(m.group(2)), float(m.group(3))
            configs.setdefault((a, h), {})[kind] = col
    return configs


def build_data(csv_path):
    df = pd.read_csv(csv_path)
    df["_fid"] = function_ids(df)
    configs = parse_config_columns(df)
    if not configs:
        raise ValueError(f"No mean_a*_h* columns found in {csv_path}")

    n_std = sum(1 for c in configs.values() if "std" in c)
    if n_std == 0:
        print(f"[WARNING] Aucune colonne 'std_a*_h*' detectee dans {csv_path}.")
    else:
        bad = [configs[c]["std"] for c in configs if "std" in configs[c]
               and df[configs[c]["std"]].isna().all()]
        if bad:
            print(f"[WARNING] Colonnes std presentes mais toutes NaN: {bad}")

    cfgs = list(configs)
    mean_mat = df[[configs[c]["mean"] for c in cfgs]].to_numpy(dtype=float)
    rank_mat = pd.DataFrame(mean_mat).rank(axis=1, method="average").to_numpy()

    for i, cfg in enumerate(cfgs):
        a, h = cfg
        rcol = f"rank_a{a:g}_h{h:g}"
        df[rcol] = rank_mat[:, i]
        configs[cfg]["rank"] = rcol

    avg_rank_cfg = {cfg: float(rank_mat[:, i].mean()) for i, cfg in enumerate(cfgs)}

    sel_cfg = sorted(cfgs, key=lambda c: avg_rank_cfg[c])[:NBEST]

    mean_cols = [configs[c]["mean"] for c in sel_cfg]
    df["_best"] = df[mean_cols].min(axis=1)

    return df, configs, sel_cfg, avg_rank_cfg


# --------------------------------------------------------------------------- #
# Statistical tests
# --------------------------------------------------------------------------- #
def friedman_test(df, configs):
    """Friedman test on the per-function rank matrix (all configurations)."""
    if not SCIPY_OK:
        return None
    rank_cols = [configs[c]["rank"] for c in configs if "rank" in configs[c]]
    if len(rank_cols) < 2:
        return None
    data = [df[c].to_numpy(dtype=float) for c in rank_cols]
    stat, p = stats.friedmanchisquare(*data)
    return stat, p, len(rank_cols), len(df)


def wilcoxon_counts(df, configs, sel_cfg):
    """Win/tie/loss counts of each selected config against the best one.

    Returns a dict: cfg -> (n_better, n_worse, n_equal) where counts refer to
    the other config vs the best (minimization: better = lower mean).
    """
    best = sel_cfg[0]
    x_best = df[configs[best]["mean"]].to_numpy(dtype=float)
    out = {}
    for cfg in sel_cfg[1:]:
        x = df[configs[cfg]["mean"]].to_numpy(dtype=float)
        n_better = int(np.sum(x < x_best))
        n_worse = int(np.sum(x > x_best))
        n_equal = int(np.sum(x == x_best))
        out[cfg] = (n_better, n_worse, n_equal)
    return out


# --------------------------------------------------------------------------- #
# Figure: average Friedman ranks only
# --------------------------------------------------------------------------- #
def plot_avg_ranks(avg_ranks_by_dim, out_png):
    dims = sorted(avg_ranks_by_dim)
    all_cfgs = sorted({c for d in dims for c in avg_ranks_by_dim[d]},
                      key=lambda c: avg_ranks_by_dim[dims[0]].get(c, 999))

    fig, axes = plt.subplots(1, len(dims), figsize=(6.4 * len(dims), 4.2), sharey=True)
    if len(dims) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(all_cfgs)))

    for ax, dim in zip(axes, dims):
        vals = [avg_ranks_by_dim[dim].get(c, np.nan) for c in all_cfgs]
        labels = [cfg_label(c).replace(", ", "\n") for c in all_cfgs]
        bars = ax.bar(range(len(all_cfgs)), vals, color=colors, edgecolor="black")
        ax.set_xticks(range(len(all_cfgs)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"D={dim}", fontsize=12)
        ax.set_xlabel("Configuration")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Average rank (lower is better)")
    fig.suptitle("Average Friedman rank per configuration on cec2013", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# docx writing
# --------------------------------------------------------------------------- #
def add_caption(doc, text, bold_prefix):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run(bold_prefix), bold=True)
    style_run(p.add_run(" " + text), bold=False)


def write_cell(cell, segments):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for text, bold, sup in segments:
        style_run(p.add_run(text), bold=bold, superscript=sup)


def add_results_table(doc, df, configs, sel_cfg, avg_rank_cfg, fr, table_num, dim):
    """Results table (best per row in bold) + w/t/l, Wilcoxon p, Friedman rank rows,
    in the same layout as the algorithm-comparison tables."""
    labels = [cfg_label(c) for c in sel_cfg]
    ncols = len(sel_cfg) + 1
    table = doc.add_table(rows=1, cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for cell, h in zip(table.rows[0].cells, ["Func."] + labels):
        write_cell(cell, [(h, True, False)])

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        cells = table.add_row().cells
        write_cell(cells[0], [(f"F{int(row['_fid'])}", False, False)])

        for j, cfg in enumerate(sel_cfg):
            cols = configs[cfg]
            mean = row[cols["mean"]]
            std = row[cols["std"]] if "std" in cols and cols["std"] in row.index else float("nan")
            bold = (mean == row["_best"])
            write_cell(cells[1 + j], cell_segments(mean, std, bold))

    # --- Wilcoxon (best configuration vs each other one, on the mean errors) ---
    means_df = pd.DataFrame({l: df[configs[c]["mean"]].to_numpy(dtype=float)
                             for l, c in zip(labels, sel_cfg)})
    wil = comp_wilcoxon(means_df, ref=labels[0])

    cells = table.add_row().cells
    write_cell(cells[0], [("w/t/l (best)", False, False)])
    for j, l in enumerate(labels):
        txt = "--" if l not in wil else f"{wil[l][0]}/{wil[l][1]}/{wil[l][2]}"
        write_cell(cells[1 + j], [(txt, False, False)])

    cells = table.add_row().cells
    write_cell(cells[0], [("Wilcoxon p", False, False)])
    for j, l in enumerate(labels):
        txt = "--" if l not in wil else f"{p_fmt(wil[l][3])} ({wil[l][4]})"
        write_cell(cells[1 + j], [(txt, False, False)])

    # --- Friedman rank row (average rank over all tested configurations) ---
    ranks = [avg_rank_cfg[c] for c in sel_cfg]
    cells = table.add_row().cells
    write_cell(cells[0], [("Friedman rank", True, False)])
    for j, r in enumerate(ranks):
        write_cell(cells[1 + j], [(f"{r:.2f}", r == min(ranks), False)])

    if fr is not None:
        row = table.add_row()
        merged = row.cells[0].merge(row.cells[-1])
        write_cell(merged, [(f"Friedman test ({fr[2]} configurations): "
                             f"χ² = {fr[0]:.2f}, {p_eq(fr[1])}", False, False)])

    add_caption(doc,
                f"Mean ± std of the five best huem_abc configurations on cec2013 benchmark "
                f"functions (D={dim}). Bold = best per function. Wilcoxon signed-rank test (α={ALPHA:g}) "
                f"compares the best configuration (first column) with each other one "
                f"('+' best configuration significantly better, '{MINUS}' significantly worse, "
                f"'{APPROX}' no significant difference). Friedman rank: average rank over all "
                f"tested configurations, lower is better.",
                f"Table {table_num}.")


def add_friedman_table(doc, friedman_by_dim, table_num):
    """Single Friedman table combining all dimensions."""
    dims = sorted(friedman_by_dim)
    table = doc.add_table(rows=1, cols=len(dims) + 1)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = [""] + [f"D={d}" for d in dims]
    for cell, h in zip(table.rows[0].cells, headers):
        write_cell(cell, [(h, True, False)])

    def p_text(p):
        if p == 0:
            return "< 1e-300"
        return f"{p:.2e}"

    rows = [("Friedman statistic", lambda fr: f"{fr[0]:.3f}"),
            ("p-value", lambda fr: p_text(fr[1])),
            (f"Significant (α={ALPHA:g})", lambda fr: "yes" if fr[1] < ALPHA else "no")]
    for label, fmt in rows:
        cells = table.add_row().cells
        write_cell(cells[0], [(label, False, False)])
        for j, d in enumerate(dims):
            write_cell(cells[1 + j], [(fmt(friedman_by_dim[d]), False, False)])

    add_caption(doc,
                f"Friedman test on per-function ranks of all configurations, for D={dims[0]} and D={dims[-1]} "
                f"({friedman_by_dim[dims[0]][2]} configurations, {friedman_by_dim[dims[0]][3]} functions per dimension).",
                f"Table {table_num}.")


def add_picture(doc, img_path, caption_prefix, caption_text, width=8.2):
    doc.add_picture(img_path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption_text, caption_prefix)


def select_best_cfg(avg_ranks_by_dim):
    """Best configuration = lowest mean Friedman rank over the dimensions."""
    dims = sorted(avg_ranks_by_dim)
    overall = {c: float(np.mean([avg_ranks_by_dim[d][c] for d in dims]))
               for c in avg_ranks_by_dim[dims[0]]}
    best = min(overall, key=overall.get)
    print(f"[BEST] Meilleure combinaison (Friedman): {cfg_label(best)} "
          f"(rang moyen = {overall[best]:.2f})")
    return best


# =========================================================================== #
# Comparison ABC / ABC-Scout / ABC-HUFS / huem_abc
# =========================================================================== #
def _norm_name(s):
    """'ABC-HUFS ' / 'abc_ant' / 'ABC-HUFS ' -> 'abchufs'."""
    return re.sub(r"[-_\s]", "", s.lower())


def find_algo_csv(algo, dim):
    """Find '<algo>_cec2013_D<dim>.csv' in RESULTS (case/-/_ insensitive)."""
    rx = re.compile(rf"^(?:\d+_)?(.+?)_cec2013_d{dim}\.csv$", re.IGNORECASE)
    for f in sorted(os.listdir(RESULTS)):
        m = rx.match(f)
        if m and _norm_name(m.group(1)) == _norm_name(algo):
            return os.path.join(RESULTS, f)
    return None


def load_algo_stats(path):
    """Read the results of one algorithm. Two CSV layouts are supported:

    (1) summary : Algorithm, Function, Mean, Std, Min, Median, Max   (or Mean Error / Std Dev)
    (2) raw runs: func_id, 0, 1, ..., 29   (one row per function, one column per run)

    Returns (mean, std) as pandas Series indexed by function number.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    mcol = next((c for c in df.columns if c.lower().startswith("mean")), None)
    if mcol is not None:                                   # (1) summary file
        scol = next((c for c in df.columns if c.lower().startswith("std")), None)
        ids = function_ids(df)
        mean = pd.Series(df[mcol].to_numpy(dtype=float), index=ids).sort_index()
        std = (pd.Series(df[scol].to_numpy(dtype=float), index=ids).sort_index()
               if scol else pd.Series(np.nan, index=mean.index))
        return mean, std

    idcol = _id_column(df) or df.columns[0]                # (2) raw runs
    ids = function_ids(df)
    runs = df.drop(columns=[idcol]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mean = pd.Series(np.nanmean(runs, axis=1), index=ids).sort_index()
    std = pd.Series(np.nanstd(runs, axis=1), index=ids).sort_index()   # population std (ddof=0)
    print(f"     (raw runs file: {runs.shape[1]} runs per function -> mean/std computed)")
    return mean, std


def build_comparison(dim, huem=None):
    """Return (means, stds) DataFrames, columns = COMP_ALGOS (those available).

    huem = (df, configs, best_cfg) when huem_abc is stored as mean_a*_h* configurations
    (its best configuration is used); None -> huem_abc is read like the other algorithms.
    Only the functions available for ALL algorithms are kept.
    """
    means, stds = {}, {}
    for algo in COMP_ALGOS:
        if algo == "huem_abc" and huem is not None:
            h_df, h_cfgs, h_best = huem
            cols = h_cfgs[h_best]
            ids = h_df["_fid"].astype(int).to_numpy()
            means[algo] = pd.Series(h_df[cols["mean"]].to_numpy(dtype=float), index=ids)
            stds[algo] = (pd.Series(h_df[cols["std"]].to_numpy(dtype=float), index=ids)
                          if "std" in cols else pd.Series(np.nan, index=ids))
            continue
        path = find_algo_csv(algo, dim)
        if path is None:
            print(f"[WARNING] CSV introuvable pour {algo} (D={dim}) dans {RESULTS}")
            continue
        means[algo], stds[algo] = load_algo_stats(path)
        print(f"     {algo}: {os.path.basename(path)} ({len(means[algo])} functions)")

    if not means:
        return None
    common = sorted(set.intersection(*[set(m.index) for m in means.values()]))
    all_ids = sorted(set.union(*[set(m.index) for m in means.values()]))
    if len(common) < len(all_ids):
        for a, m in means.items():
            missing = sorted(set(all_ids) - set(m.index))
            if missing:
                print(f"[WARNING] D={dim}: {a} has no result for {['F%d' % i for i in missing]} "
                      f"-> these functions are left out of the comparison for all algorithms.")
    if not common:
        print(f"[WARNING] D={dim}: no function common to all algorithms.")
        return None
    idx = [f"F{i}" for i in common]
    mdf = pd.DataFrame({a: means[a].loc[common].to_numpy() for a in means}, index=idx)
    sdf = pd.DataFrame({a: stds[a].reindex(common).to_numpy() for a in means}, index=idx)
    return mdf, sdf


def comp_wilcoxon(means, ref=REF_ALGO):
    """Paired Wilcoxon signed-rank test over the functions (on mean errors), ref vs each algo.

    Returns {algo: (wins, ties, losses, p, symbol)} where wins/ties/losses are counted
    from the point of view of `ref`; symbol: '+' ref significantly better,
    '−' significantly worse, '≈' no significant difference.
    """
    res = {}
    if ref not in means.columns:
        return res
    for a in means.columns:
        if a == ref:
            continue
        x, y = means[ref].to_numpy(), means[a].to_numpy()
        w, t, l = int(np.sum(x < y)), int(np.sum(x == y)), int(np.sum(x > y))
        if not SCIPY_OK:
            res[a] = (w, t, l, float("nan"), "--")
            continue
        p = 1.0 if np.all(x == y) else stats.wilcoxon(
            x, y, zero_method="wilcox", alternative="two-sided").pvalue
        if p < ALPHA:
            sym = "+" if w > l else MINUS
        else:
            sym = APPROX
        res[a] = (w, t, l, p, sym)
    return res


def comp_friedman(means):
    """Average Friedman ranks (1 = best, ties -> mean rank) + (chi2, p)."""
    ranks = means.rank(axis=1, method="average")
    avg = ranks.mean(axis=0)
    fr = None
    if SCIPY_OK and means.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[means[a].to_numpy() for a in means.columns])
    return avg, fr


def p_eq(p):
    """'p < 0.001' or 'p = 0.431' for use in running text."""
    if p is None or not np.isfinite(p):
        return "p = --"
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def p_fmt(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def add_comparison_table(doc, means, stds, wil, avg_rank, fr, table_num, dim):
    """F | ABC | ABC-Scout | ABC-HUFS | HUEM-ABC  (best per row in bold)
    + w/t/l, Wilcoxon p-value (+/−/≈) and Friedman rank rows."""
    algos = list(means.columns)
    table = doc.add_table(rows=1, cols=len(algos) + 1)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for cell, h in zip(table.rows[0].cells, ["F"] + algos):
        write_cell(cell, [(h, True, False)])

    for f, mrow in means.iterrows():
        cells = table.add_row().cells
        write_cell(cells[0], [(f, False, False)])
        best = mrow.min()
        for j, a in enumerate(algos):
            write_cell(cells[1 + j],
                       cell_segments(mrow[a], stds.loc[f, a], bold=(mrow[a] == best)))

    # --- Wilcoxon rows (reference algorithm vs each competitor) ---
    cells = table.add_row().cells
    write_cell(cells[0], [(f"w/t/l ({REF_ALGO})", False, False)])
    for j, a in enumerate(algos):
        txt = "--" if a not in wil else f"{wil[a][0]}/{wil[a][1]}/{wil[a][2]}"
        write_cell(cells[1 + j], [(txt, False, False)])

    cells = table.add_row().cells
    write_cell(cells[0], [("Wilcoxon p", False, False)])
    for j, a in enumerate(algos):
        txt = "--" if a not in wil else f"{p_fmt(wil[a][3])} ({wil[a][4]})"
        write_cell(cells[1 + j], [(txt, False, False)])

    # --- Friedman rank row ---
    cells = table.add_row().cells
    write_cell(cells[0], [("Friedman rank", True, False)])
    for j, a in enumerate(algos):
        write_cell(cells[1 + j], [(f"{avg_rank[a]:.2f}", avg_rank[a] == avg_rank.min(), False)])

    if fr is not None:
        row = table.add_row()
        merged = row.cells[0].merge(row.cells[-1])
        write_cell(merged, [(f"Friedman test: χ² = {fr[0]:.2f}, {p_eq(fr[1])}", False, False)])

    add_caption(doc,
                f"Mean ± std error of the compared algorithms on cec2013 benchmark functions (D={dim}). "
                f"Bold = best per function. Wilcoxon signed-rank test (α={ALPHA:g}) compares {REF_ALGO} with each "
                f"competitor ('+' {REF_ALGO} significantly better, '{MINUS}' significantly worse, "
                f"'{APPROX}' no significant difference). Friedman rank: lower is better. "
                + (f"HUEM-ABC= best configuration selected from {CFG_TABLES_TXT}." if HUEM_FROM_CONFIGS else ""),
                f"Table {table_num}.")


def add_comp_friedman_table(doc, comp_ranks, comp_fr, table_num):
    """Friedman ranks for each algorithm and each dimension in a single table."""
    dims = sorted(comp_ranks)
    algos = list(comp_ranks[dims[0]].index)
    table = doc.add_table(rows=1, cols=len(algos) + 3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for cell, h in zip(table.rows[0].cells, ["D"] + algos + ["χ²", "p"]):
        write_cell(cell, [(h, True, False)])

    for d in dims:
        cells = table.add_row().cells
        write_cell(cells[0], [(str(d), False, False)])
        r = comp_ranks[d]
        for j, a in enumerate(algos):
            write_cell(cells[1 + j], [(f"{r[a]:.2f}", r[a] == r.min(), False)])
        fr = comp_fr.get(d)
        write_cell(cells[-2], [(f"{fr[0]:.2f}" if fr else "--", False, False)])
        write_cell(cells[-1], [(p_fmt(fr[1]) if fr else "--", False, False)])

    cells = table.add_row().cells
    mean_rank = pd.concat([comp_ranks[d] for d in dims], axis=1).mean(axis=1)
    write_cell(cells[0], [("Mean", True, False)])
    for j, a in enumerate(algos):
        write_cell(cells[1 + j], [(f"{mean_rank[a]:.2f}", mean_rank[a] == mean_rank.min(), False)])
    write_cell(cells[-2], [("--", False, False)])
    write_cell(cells[-1], [("--", False, False)])

    add_caption(doc,
                "Average Friedman ranks of the compared algorithms for each dimension on the cec2013 "
                "benchmark (lower is better, best in bold).",
                f"Table {table_num}.")


def plot_comp_ranks(comp_ranks, out_png):
    dims = sorted(comp_ranks)
    algos = list(comp_ranks[dims[0]].index)
    x = np.arange(len(algos))
    width = 0.8 / len(dims)
    colors = plt.cm.viridis(np.linspace(0.25, 0.8, len(dims)))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, d in enumerate(dims):
        vals = comp_ranks[d].to_numpy()
        bars = ax.bar(x + (i - (len(dims) - 1) / 2) * width, vals, width,
                      color=colors[i], edgecolor="black", label=f"D={d}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Average Friedman rank (lower is better)")
    ax.set_ylim(0, len(algos) + 0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# CSV discovery
# --------------------------------------------------------------------------- #
def find_csv(dim):
    patterns = [
        os.path.join(RESULTS, f"huem-abc_cec2013_D{dim}.csv"),
        os.path.join(RESULTS, f"huem_abc_cec2013_D{dim}.csv"),
        os.path.join(RESULTS, f"huem[-_]abc*_D{dim}.csv"),
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    return None


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def dims_text(dims):
    d = [f"D={x}" for x in dims]
    return d[0] if len(d) == 1 else ", ".join(d[:-1]) + " and " + d[-1]


def tables_text(nums):
    if len(nums) == 1:
        return f"Table {nums[0]}"
    return "Tables " + ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global CFG_TABLES_TXT, HUEM_FROM_CONFIGS

    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.isdir(RESULTS):
        raise SystemExit(f"[ERROR] Results directory not found: {RESULTS}")

    doc = Document()

    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    if section.page_width < section.page_height:
        section.page_width, section.page_height = section.page_height, section.page_width

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(SIZE)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(title.add_run("Results of HUEM_ABC on the cec2013 benchmark"), bold=True)

    if not SCIPY_OK:
        print("[WARNING] scipy not found -- statistical tests will be skipped.")

    table_no = FIRST_TABLE
    fig_no = FIRST_FIGURE

    avg_ranks_by_dim = {}   # dim -> {cfg: avg_rank}
    huem_data = {}          # dim -> (df, configs)   (kept for the comparison section)
    cfg_table_nums = []

    # --- huem_abc configuration tables (only if the CSV has mean_a*_h* columns) ---
    for dim in DIMS:
        path = find_csv(dim)
        if not path:
            print(f"[WARNING] No huem_abc CSV file found for D={dim} in: {RESULTS}")
            continue

        try:
            df, configs, sel_cfg, avg_rank_cfg = build_data(path)
        except ValueError:
            print(f"[INFO] D={dim}: {os.path.basename(path)} has no mean_a*_h* columns -> read as a "
                  f"plain algorithm file (no configuration table for HUEM_ABC).")
            continue
        print(f"[OK] D={dim} file processed ({os.path.basename(path)})")
        print("     Selected configurations: " + ", ".join(cfg_label(c) for c in sel_cfg))

        fr_cfg = friedman_test(df, configs)

        add_results_table(doc, df, configs, sel_cfg, avg_rank_cfg, fr_cfg, table_no, dim)
        cfg_table_nums.append(table_no)
        table_no += 1
        doc.add_paragraph()

        avg_ranks_by_dim[dim] = avg_rank_cfg
        huem_data[dim] = (df, configs)

    # --- Best configuration (used as huem_abc in the comparison tables) ---
    best_cfg = select_best_cfg(avg_ranks_by_dim) if avg_ranks_by_dim else None

    # --- Figure: average Friedman ranks of the configurations ---
    if avg_ranks_by_dim:
        fig_ranks = plot_avg_ranks(avg_ranks_by_dim,
                                   os.path.join(OUTDIR, "fig_friedman_ranks_cec2013.png"))
        print(f"[OK] Figure saved: {fig_ranks}")
        add_picture(doc, fig_ranks, f"Figure {fig_no}.",
                    f"Average Friedman rank of each configuration on the cec2013 benchmark "
                    f"({dims_text(sorted(avg_ranks_by_dim))}). Lower is better.")
        fig_no += 1

    # --- Comparison ABC / ABC-Scout / ABC-HUFS / huem_abc ---
    HUEM_FROM_CONFIGS = bool(huem_data)
    CFG_TABLES_TXT = tables_text(cfg_table_nums) if cfg_table_nums else ""
    comp_ranks, comp_fr = {}, {}
    for dim in DIMS:
        print(f"[..] Comparison D={dim}")
        huem = (huem_data[dim][0], huem_data[dim][1], best_cfg) if dim in huem_data else None
        res = build_comparison(dim, huem)
        if res is None:
            continue
        means, stds = res
        if means.shape[1] < 2:
            print(f"[WARNING] Not enough algorithms found for D={dim}; comparison skipped.")
            continue
        wil = comp_wilcoxon(means)
        avg_rank, fr = comp_friedman(means)

        if huem_data or comp_ranks:           # no page break before the very first table
            doc.add_page_break()
        comp_ranks[dim], comp_fr[dim] = avg_rank, fr
        add_comparison_table(doc, means, stds, wil, avg_rank, fr, table_no, dim)
        print(f"[OK] Comparison table D={dim} added as Table {table_no} ({means.shape[0]} functions)")
        table_no += 1

    if comp_ranks:
        doc.add_paragraph()
        fig_comp = plot_comp_ranks(comp_ranks,
                                   os.path.join(OUTDIR, "fig_friedman_comparison_cec2013.png"))
        print(f"[OK] Figure saved: {fig_comp}")
        add_picture(doc, fig_comp, f"Figure {fig_no}.",
                    "Average Friedman rank of ABC, ABC-Scout, ABC-HUFS, HUEM-ABC on the cec2013 "
                    f"benchmark ({dims_text(sorted(comp_ranks))}). Lower is better.", width=6.5)

    if comp_ranks:
        add_comp_friedman_table(doc, comp_ranks, comp_fr, table_no)
        table_no += 1

    out_path = os.path.join(OUTDIR, "Tables_HUEM_ABC_variants_CEC2013.docx")
    doc.save(out_path)
    print(f"\n[SUCCESS] Document generated: {out_path}")


if __name__ == "__main__":
    main()
