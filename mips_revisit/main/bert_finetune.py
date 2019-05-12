"""
Usage: python -m mips_revisit.main.bert_finetune --task mrpc --out_dir gs://bert-mips/finetune/mrpc --overwrite

Given a task and output directory OUT_DIR, this file, when run,
performs the following:

* Pull in a pre-trained cased BERT base model.
* Fine-tunes the model to the task using the parameters in the paper.

If $OUT_DIR exists, this does not do anything unless --overwrite is
specified.

Inside $OUT_DIR, creates the following files:

config.json - BERT model configuration params
pytorch_model.bin - binary pytorch state dict tuned classification BERT
vocab.txt - tokens used by the model
"""

import os
import shutil

import tensorflow as tf
from absl import app, flags

from .. import log
from ..glue import get_glue
from ..huggingface.run_classifier import main
from ..params import bert_glue_params
from ..tpu_setup import colab_env, make_tpu_estimator
from ..utils import import_matplotlib, seed_all

flags.DEFINE_enum("task", None, ["mrpc"], "BERT fine-tuning task")

flags.DEFINE_string("out_dir", None, "checkpoint directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory")


def _main(_argv):
    log.init()

    out_dir = flags.FLAGS.out_dir
    if tf.gfile.Exists(out_dir) and not flags.FLAGS.overwrite:
        log.info("output directory {} already exists, exiting", out_dir)
        return
    elif tf.gfile.Exists(out_dir) and flags.FLAGS.overwrite:
        log.info("output directory {} exists, will overwrite", out_dir)

    glue_data = get_glue(flags.FLAGS.task)
    seed_all(1234)

    args = bert_glue_params(flags.FLAGS.task)
    args.data_dir = glue_data

    local_dir = os.path.join(os.getcwd(), "generated", flags.FLAGS.task)
    log.info(
        "using dir {} for local weights (final weights will be in {})",
        local_dir,
        out_dir,
    )
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    args.output_dir = local_dir
    args.cache_dir = "/tmp/bert_cache"
    args.load_dir = None
    args.do_train = True
    args.do_eval = False
    args.no_cuda = False
    args.local_rank = -1

    # inferrable from GPU mem?
    args.gradient_accumulation_steps = 1
    args.fp16 = False
    args.loss_scale = 0

    # might be useful later
    args.server_ip = ""
    args.server_port = ""

    main(args)

    expected_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    for f in expected_files:
        src = os.path.join(local_dir, f)
        dst = os.path.join(out_dir, f)
        tf.gfile.Copy(src, dst, overwrite=True)


if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("out_dir")
    app.run(_main)
