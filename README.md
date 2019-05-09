# mips-revisit: Failure-driven Decision Making

This module, `mips_revisit`, defines the modules and evaluation scripts for TODO.

## Setup

```
# CPU
conda env create -f environment.yaml [--prefix /your/path/to/anaconda3]
source activate mips-revisit-env
```

## Usage

Run BERT pretraining with approximate K-MIPS (default K = 10)
```
python -m mips-revisit.main.pretrain_bert --attention approx-mips
# [2020-08-08 19:51:03 PDT mips_revisit/log.py:102] Hello, World! K = 10
```

## Dev info

All scripts are available in `scripts/`, and should be run from the repo root in the `fdd-env`.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo... I don't really use this |
| `format.sh` | auto-format the entire `mips_revisit` directory |

Use `conda env export > environment.yaml` to save new dependencies.
