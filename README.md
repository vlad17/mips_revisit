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
(mips-revisit-env) 10:23:58 vlad@vlad-T460:~/dev/mips-att$ python -m mips_revisit.main.attention_decay --task mrpc --output_directory gs://hello
[2019-05-11 10:24:35 PDT mips_revisit/utils.py:41] loading glue data
[2019-05-11 10:24:38 PDT mips_revisit/utils.py:47] took       2.21 sec toloading glue data
[2019-05-11 10:24:38 PDT mips_revisit/main/attention_decay.py:90] Hello, World! K = 10
```

## Dev info

All scripts are available in `scripts/`, and should be run from the repo root in the `fdd-env`.

| script | purpose |
| ------ | ------- |
| `format.sh` | auto-format the entire `mips_revisit` directory |

Use `conda env export > [cpu-]environment.yaml` to bsave new dependencies.
