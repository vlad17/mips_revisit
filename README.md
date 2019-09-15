# mips_revisit

This module, `mips_revisit`, defines the modules and evaluation scripts for MIPS-based
attention on transformer models.

Requirements: [Anaconda 3](https://www.anaconda.com/distribution/) in the path.

Make sure that `conda` is available in your path. Then run `conda activate`.

## Setup

```
# CPU
conda env create -f cpu-environment.yaml
source activate mips-revisit-env

# GPU
conda env create -f environment.yaml
source activate mips-revisit-env
```

Use `conda env export --no-builds > environment.yaml` to save new dependencies.

To be done when dependencies update in the yaml file but your local environment is out of date, use `conda env update -f environment.yaml --prune`.

## Usage

Run BERT pretraining with approximate K-MIPS.
```
for K in 0 10; do 
for ATTN in topk topk-50 ; do
for TASK in mrpc cola ; do
S3PREFIX=s3://vlad-deeplearn/mips-revisit/bert/${TASK}/k${K}/${ATTN}
python -m mips_revisit.main.bert_finetune --k ${K} --attn ${ATTN} --task ${TASK} --out_dir ${S3PREFIX} --overwrite
python -m mips_revisit.main.bert_eval --task ${TASK} --eval_dir ${S3PREFIX} --overwrite
done
done
done

for TASK in mrpc cola ; do
S3PREFIX=s3://vlad-deeplearn/mips-revisit/bert/${TASK}
python -m mips_revisit.main.bert_agg --prefix ${S3PREFIX} --task ${TASK} --overwrite
done

K=0
ATTN=soft
TASK=mrpc
for SEED in 3 5 ; do
S3PREFIX=s3://vlad-deeplearn/mips-revisit/bert-4321/${TASK}/k${K}/${ATTN}/seed${SEED}
python -m mips_revisit.main.bert_finetune --k ${K} --attn ${ATTN} --task ${TASK} --out_dir ${S3PREFIX} && \
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

All scripts are available in `scripts/`, and should be run from the repo root in the `mips-revisit-env`.

| script | purpose |
| ------ | ------- |
| `format.sh` | auto-format the entire `mips_revisit` directory |

