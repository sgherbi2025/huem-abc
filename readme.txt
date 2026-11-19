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
