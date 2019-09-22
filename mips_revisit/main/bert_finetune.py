"""
Given a task and output directory OUT_DIR, this file, when run,
performs the following:

* Pull in a pre-trained cased BERT base model.
* Fine-tunes the model to the task using the parameters in the paper.

Inside $OUT_DIR, creates the following files:

config.json - BERT model configuration params
train.npz - examples_seen and train_loss 1D arrays
train.pdf - simple training curve based on the above
pytorch_model.bin - binary pytorch state dict tuned classification BERT
vocab.txt - tokens used by the model
"""

import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags

from .. import log
from ..glue import get_glue
# imports flags.FLAGS.{attn,k} via huggingface.modeling
from ..huggingface.run_classifier import main
from ..params import GLUE_TASK_NAMES, bert_glue_params
from ..sms import makesms
from ..sync import exists, sync, simplehash
from ..utils import import_matplotlib, seed_all, timeit

flags.DEFINE_enum("task", None, GLUE_TASK_NAMES, "BERT fine-tuning task")

flags.DEFINE_string("out_dir", None, "checkpoint output directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory files")

flags.DEFINE_integer("seed", 1, "randomness seed")

def _main(_argv):
    log.init()

    out_dir = flags.FLAGS.out_dir
    main(out_dir, flags.FLAGS.overwrite, flags.FLAGS.task, flags.FLAGS.seed)

if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("out_dir")

    app.run(_main)
