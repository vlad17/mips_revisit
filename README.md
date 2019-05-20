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
for TASK in mrpc cola ; do
K=10
ATTN=topk
S3PREFIX=s3://vlad-deeplearn/mips-revisit/bert/${TASK}/k${K}/${ATTN}
python -m mips_revisit.main.bert_finetune --k ${K} --attn ${ATTN} --task ${TASK} --out_dir ${S3PREFIX}
python -m mips_revisit.main.bert_eval --task ${TASK} --eval_dir ${S3PREFIX}
done
```

For text updates, set the following env vars.

```
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TARGET_PHONE_FOR_SMS
ORIGIN_PHONE_FOR_SMS
```

## Dev info

All scripts are available in `scripts/`, and should be run from the repo root in the `fdd-env`.

| script | purpose |
| ------ | ------- |
| `format.sh` | auto-format the entire `mips_revisit` directory |

Use `conda env export > [cpu-]environment.yaml` to bsave new dependencies.
