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
TASK=mrpc
python -m mips_revisit.main.bert_finetune --task ${TASK} --out_dir gs://bert-mips/finetune/${TASK} --overwrite
python -m mips_revisit.main.bert_eval --task ${TASK} --eval_dir gs://bert-mips/finetune/${TASK} --overwrite

```

## Dev info

All scripts are available in `scripts/`, and should be run from the repo root in the `fdd-env`.

| script | purpose |
| ------ | ------- |
| `format.sh` | auto-format the entire `mips_revisit` directory |

Use `conda env export > [cpu-]environment.yaml` to bsave new dependencies.
