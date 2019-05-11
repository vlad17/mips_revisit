"""
Usage: python -m mips_revisit.main.attention_decay.py --task mrpc --output_directory gs://bert-mips/attention_decay

Given a task TASK and output directory OUTPUT_DIRECTORY, this file, when run,
performs the following:

* Pull in a pre-trained cased BERT base model.
* Fine-tunes the model to the task using the parameters in the paper.
* Evaluates the model on the test set
* Inspects the softmax activations, before dropout or masking,
  and generates various graphics and diagnostics.

In particular, inside $OUTPUT_DIRECTORY/$TASK, creates the following:

# fine tuning checkpoints
fine_tuning/*.ckpt-*

# link to CV-chosen checkpoint
fine_tuning/final_model.ckpt

# pictures of activation distribution
plots/{layer,head,from_index}.pdf

# detailed TF logs from fine tuning
logs/fine_tuning.txt

# detailed TF logs from evaluation
logs/eval.txt

# tee'd stdout [TODO: tee to gs://?]
logs/stdout.txt

# array, indexed by "to" position, of average activations
activations.npy

# succinct results overview
summary.json

TODO: structure of summary.json
"""

# [LATER!] compare to fine-tuning and evaluating with top-k and top-k 50% decay

# inferred specific for task:
# --> bert base
# --> bert params
# --> comparison dev/test

# procedure:
# pull cased bert base (stdout: success/timeit)
# fine tuning (stdout: fine tuning params FOR TASK)
#             (stdout: 100-update training procedure logs)
#             (stdout: final timeit)
#             (stdout: final dev eval, compare to paper FOR TASK)
# test eval (stdout) -- whole test
# eval for task, compare to paper for task
#
# get giant activation tensor -- whole test, batched
# make pretty above plots
# (stdout: writing plot to ...)
# (stdout: writing marginal activations to ...npy)
# (print #activations > 1e-1, 1e-2, 1e-3).

import os

from absl import app, flags

from ..utils import seed_all, timeit
from .. import log
from ..bert.finetune_data import get_glue

flags.DEFINE_enum(
    "task", None, ["mrpc"],
    "BERT fine-tuning task"
)

flags.DEFINE_string(
    "output_directory", None,
    "generated artifact output directory"
)

def _main(_argv):
    log.init()

    with timeit(name='loading glue data'):
        dd = get_glue(flags.FLAGS.task)

    K = 10

    log.info("Hello, World! K = {}", K)

if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("output_directory")
    app.run(_main)
