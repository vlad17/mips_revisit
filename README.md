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

For text updates, set the following env vars.

```
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TARGET_PHONE_FOR_SMS
ORIGIN_PHONE_FOR_SMS
```

### Pretrained BERT, Motivational Top-K Finetune

Motivational experiments. On a pretrained base BERT, per the paper, we can run fin-tuning with the paper's parameters for:

* regular `soft` attention
* exact `top-k` MIPS attention (either with `--resoftmax` or `--noresoftmax`)
* `top-k-XX` inexact MIPS attention, with each of the top `k` entries resampled with probability `XX/100` as random tokens, also with optional softmax.

Example with local scripts.

```
EXPERIMENT=bert-finetune-with-softmax
TASK=mrpc
ATTN=topk-25
K=40

for SEED in 3 5 8 13 21 ; do
S3PREFIX=s3://vlad-deeplearn/mips-revisit/$EXPERIMENT/${TASK}/k${K}/${ATTN}/seed${SEED}
python -m mips_revisit.main.bert_finetune --k ${K} --attn ${ATTN} --task ${TASK} --overwrite --out_dir ${S3PREFIX} --seed ${SEED} && \
python -m mips_revisit.main.bert_eval --task ${TASK} --eval_dir ${S3PREFIX} --overwrite
done

S3PREFIX=s3://vlad-deeplearn/mips-revisit/bert-finetune-with-softmax/${TASK}
python -m mips_revisit.main.bert_agg --prefix ${S3PREFIX} --task ${TASK} --overwrite
```

Example with cluster execution.

## Dev info

All scripts are available in `scripts/`, and should be run from the repo root in the `mips-revisit-env`.

| script | purpose |
| ------ | ------- |
| `format.sh` | auto-format the entire `mips_revisit` directory |

