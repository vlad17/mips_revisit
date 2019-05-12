# mips_revisit

This module, `mips_revisit`, defines the modules and evaluation scripts for MIPS-based
attention on transformer models.

## Setup

```
# CPU
conda env create -f [cpu-]environment.yaml [--prefix /your/path/to/anaconda3]
source activate mips-revisit-env
```

## Usage

Run BERT pretraining with approximate K-MIPS (default K = 10)
```
python -m mips_revisit.main.bert_finetune --task mrpc --out_dir gs://bert-mips/finetune/mrpc --overwrite

```

## Dev info

All scripts are available in `scripts/`, and should be run from the repo root in the `fdd-env`.

| script | purpose |
| ------ | ------- |
| `format.sh` | auto-format the entire `mips_revisit` directory |

Use `conda env export > [cpu-]environment.yaml` to bsave new dependencies.
