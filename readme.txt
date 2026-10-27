# HUEM-ABC: Source Code and Experimental Results

This repository contains the source code, raw results, and table-generation notebooks accompanying a paper submitted to *The Jordanian Journal of Computer and Information Technology (JJCIT)*.

HUEM-ABC is a variant of the Artificial Bee Colony (ABC) algorithm. It is benchmarked against ABC and several state-of-the-art competitors (ABC-ANT, ASABC, NABC, ABC-MNG) on the CEC2013 and CEC2022 test suites.

## Requirements

- Python 3.x
- Libraries:

```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib
pip install "setuptools<70.0.0" opfunu
```

> `setuptools<70.0.0` is pinned for compatibility with `opfunu`, which provides the CEC benchmark functions.

## Repository Structure

```
.
├── generate_experiments_tables_cec2013_competitors.py
├── generate_experiments_tables_cec2013_huem-abc-variants.py
├── generate_experiments_tables_cec2022_3competitors.py
├── README.md
├── experiments/                 # Generated figures and tables
│   ├── fig_friedman_comparison_cec2013.png
│   ├── fig_friedman_comparison_cec2022.png
│   ├── Tables_cec2013.docx
│   ├── Tables_CEC2022.docx
│   └── Tables_HUEM_ABC_variants_CEC2013.docx
├── results/                     # Raw results (CSV, one file per algorithm and dimension)
│   ├── cec2013/                 # D = 30, 50
│   └── cec2022/                 # D = 10, 20
└── src/
    ├── competitors/             # ABC, ABC-ANT, ASABC, NABC
    └── huem_abc/                # HUEM-ABC and its variants (ABC-HUFS, ABC-Scout)
```

## Benchmarks

| Suite   | Dimensions | Algorithms |
|---------|------------|------------|
| CEC2013 | 30, 50 | ABC, ABC-ANT, ABC-HUFS,ABC-Scout, ASABC, NABC, HUEM-ABC |
| CEC2022 | 10, 20 | ABC-ANT, ASABC, NABC, HUEM-ABC |

## Usage

### 1. Run an experiment

```bash
python src/huem_abc/huem_abc_cec2013_D30.py
python src/huem_abc/huem_abc_cec2022.py
python src/competitors/nabc_cec2013.py
```

Each script runs the independent runs for its algorithm and writes a CSV file to `results/<suite>/`.

### 2. Generate tables and figures

Open the notebooks at the repository root with Jupyter and run all cells. They read the CSV files in `results/` and produce the tables (`.docx`) and Friedman-ranking figures (`.png`) in `experiments/`.

```bash
jupyter notebook
```

## Reproducibility

- Raw results of all runs reported in the paper are provided in `results/`.
- [Number of independent runs, maximum function evaluations, population size, and random seeds: to be completed to match the paper.]

## Notes

Competitor implementations (ABC-ANT, ASABC, NABC, ABC-MNG) are re-implementations based on the cited original papers; all credit for the original algorithms belongs to their respective authors.