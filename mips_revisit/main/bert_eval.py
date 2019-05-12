"""
Usage: python -m mips_revisit.main.bert_eval --task mrpc --out_dir gs://bert-mips/finetune/mrpc --overwrite

Given a task and output directory OUT_DIR, this file, when run,
performs the following:

* Attempt to load the model in $OUT_DIR, which should have been saved there
  by mips_revisit.main.bert_train
* Evaluate the model on the dev set

If $OUT_DIR/eval exists, this does not do anything unless --overwrite is
specified.

Inside $OUT_DIR/eval, creates the following files:

plots/{layer,head,from_index}.pdf - pictures of activation distribution
activations.npy - array, indexed by "to" position, of average activations
summary.json - succinct results overview

TODO: structure of summary.json

"""

import os

import tensorflow as tf
from absl import app, flags

from .. import log
from ..bert.finetune_data import get_glue
from ..tpu_setup import colab_env, make_tpu_estimator
from ..utils import import_matplotlib, seed_all

flags.DEFINE_enum("task", None, ["mrpc"], "BERT fine-tuning task")

flags.DEFINE_string("out_dir", None, "checkpoint directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory")


def _main(_argv):
    log.init()

    expected_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    for f in expected_files:
        f = os.path.join(flags.FLAGS.out_dir, f)
        if not tf.gfile.Exists(f):
            log.info("expected file {} to exist but it didn't", f)
            return

    eval_dir = os.path.join(flags.FLAGS.out_dir, "eval")
    needs_delete = False
    if tf.gfile.Exists(eval_dir) and not flags.FLAGS.overwrite:
        log.info("eval directory {} already exists, exiting", eval_dir)
        return
    elif tf.gfile.Exists(eval_dir) and flags.FLAGS.overwrite:
        log.info("eval directory {} exists, will delete", eval_dir)
        needs_delete = True

    glue_data = get_glue(flags.FLAGS.task)
    seed_all(1234)

    # TODO load
    # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
    # tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    # TODO: hugging_face/minimal.py: eval
    # do eval on train + val
    # summary = eval(model, glue_data.train())
    # activations, summary = eval(model, glue_data.val())

    # make activation plots

    # summary should have acc, loss for train x test

    # gfile upload (create a helper in utils using a temp directory)

    # TODO can delete attention decay after this

    if needs_delete:
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("out_dir")
    app.run(_main)
