"""
Usage: python -m mips_revisit.main.bert_eval --task mrpc --eval_dir gs://bert-mips/finetune/mrpc --overwrite

Given a task and evaluation directory EVAL_DIR, this file, when run,
performs the following:

* Attempt to load the model in $EVAL_DIR, which should have been saved there
  by mips_revisit.main.bert_train
* Evaluate the model on the dev set

Inside $EVAL_DIR/eval, creates the following files:

plots/{layer,head,from_index}.pdf - pictures of activation distribution
activations.npy - array, indexed by "to" position, of average activations
summary.json - succinct results overview

TODO: structure of summary.json

"""

import os
import shutil
import tempfile

import tensorflow as tf
from absl import app, flags

from .. import log
from ..glue import get_glue
from ..huggingface.run_classifier import main
from ..params import bert_glue_params
from ..utils import import_matplotlib, seed_all, timeit

flags.DEFINE_enum("task", None, ["mrpc"], "BERT fine-tuning task")

flags.DEFINE_string("eval_dir", None, "evaluation directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory files")


def _main(_argv):
    log.init()

    eval_dir = flags.FLAGS.eval_dir
    train_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    for f in train_files:
        f = os.path.join(eval_dir, f)
        if not tf.gfile.Exists(f):
            log.info("expected file {} to exist but it didn't", f)
            return

    eval_files = [
        "plots/layer.pdf",
        "plots/head.pdf",
        "plots/from_index.pdf",
        "activations.npy",
        "summary.json",
    ]

    for f in eval_files:
        f = os.path.join(eval_dir, f)
        if tf.gfile.Exists(f) and not flags.FLAGS.overwrite:
            log.info(
                "file {} exists and would be overwritten, but "
                "--overwrite not specified",
                f,
            )
            return

    glue_data = get_glue(flags.FLAGS.task)
    seed_all(1234)

    args = bert_glue_params(flags.FLAGS.task)
    args.data_dir = glue_data
    args.cache_dir = "/tmp/bert_cache"
    args.do_train = False
    args.do_eval = True
    args.no_cuda = False
    args.local_rank = -1
    args.gradient_accumulation_steps = 1  # inferrable from GPU mem?
    args.fp16 = False
    args.loss_scale = 0
    args.server_ip = ""
    args.server_port = ""

    local_dir = tempfile.mkdtemp()
    log.info("work dir {}", local_dir)

    local_weights = os.path.join(local_dir, "weights")
    local_output = os.path.join(local_dir, "output")
    os.makedirs(local_weights)
    os.makedirs(local_output)

    with timeit(name="load train weights"):
        for f in train_files:
            tf.gfile.Copy(
                os.path.join(eval_dir, f), os.path.join(local_weights, f)
            )

    args.output_dir = local_output
    args.load_dir = local_weights

    res, attn = main(args, return_attn=True)

    log.info("{}", res)

    log.info("len {} shape {}", len(res), res[0])

    # TODO: hugging_face/minimal.py: eval
    # do eval on train + val
    # summary = eval(model, glue_data.train())
    # activations, summary = eval(model, glue_data.val())

    # make activation plots

    # summary should have acc, loss for train x test

    # gfile upload (create a helper in utils using a temp directory)

    # TODO can delete attention decay after this

    log.info("cleaning up work dir {}", local_dir)
    s = ""
    for root, dirs, files in os.walk(local_output):
        level = root.replace(local_output, "").count(os.sep)
        indent = " " * 4 * (level)
        s += "{}{}/\n".format(indent, os.path.basename(root))
        subindent = " " * 4 * (level + 1)
        for f in files:
            s += "{}{}\n".format(subindent, f)
    log.info("output dirtree\n{}", s)

    # shutil.rmtree(local_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("eval_dir")
    app.run(_main)
