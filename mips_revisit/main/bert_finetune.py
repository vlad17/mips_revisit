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
from ..huggingface.run_classifier import main
from ..params import GLUE_TASK_NAMES, bert_glue_params
from ..sms import makesms
from ..sync import exists, sync
from ..utils import import_matplotlib, seed_all

flags.DEFINE_enum("task", None, GLUE_TASK_NAMES, "BERT fine-tuning task")

flags.DEFINE_enum(
    "attn", "soft", ["soft", "topk", "topk-50"], "attention type"
)

flags.DEFINE_string("out_dir", None, "checkpoint output directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory files")

flags.DEFINE_integer("k", 0, "k to use when using k-attention")


def _main(_argv):
    log.init()

    out_dir = flags.FLAGS.out_dir
    expected_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    for f in expected_files:
        f = os.path.join(flags.FLAGS.out_dir, f)
        if exists(f) and not flags.FLAGS.overwrite:
            log.info(
                "file {} exists and would be overwritten, but "
                "--overwrite not specified",
                f,
            )
            return

    makesms(
        "STARTING bert finetune\nk={}\nattn={}\ntask={}\nout_dir={}".format(
            flags.FLAGS.k, flags.FLAGS.attn, flags.FLAGS.task, out_dir
        )
    )

    glue_data = get_glue(flags.FLAGS.task)
    seed_all(1234)

    args = bert_glue_params(flags.FLAGS.task)
    args.data_dir = glue_data

    local_dir = os.path.join(
        os.getcwd(), "generated", "initial", flags.FLAGS.task
    )
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

    try:
        result = main(args, None)
        save_train_results(
            local_dir, result["train_curve"], args.train_batch_size
        )

        sync(local_dir, out_dir)
        log.info("removing work dir {}", local_dir)
        shutil.rmtree(local_dir)
    except:
        makesms(
            "ERROR in bert finetune\nk={}\nattn={}\ntask={}\nout_dir={}".format(
                flags.FLAGS.k, flags.FLAGS.attn, flags.FLAGS.task, out_dir
            )
        )
        raise

    makesms(
        "COMPLETED bert finetune\nk={}\nattn={}\ntask={}\nout_dir={}".format(
            flags.FLAGS.k, flags.FLAGS.attn, flags.FLAGS.task, out_dir
        )
    )


def save_train_results(local_dir, train_loss_pairs, bsz):

    outfile = os.path.join(local_dir, "train.npz")
    examples_seen, train_loss = map(np.array, zip(*train_loss_pairs))
    np.savez(outfile, examples_seen=examples_seen, train_loss=train_loss)

    plt = import_matplotlib()
    log.info("generating {}", outfile)

    plt.plot(
        examples_seen,
        train_loss,
        ls=":",
        label="orig",
        color="blue",
        alpha=0.7,
    )

    window = len(examples_seen) // 50

    if window > 1:
        s = pd.Series(data=train_loss, index=examples_seen)
        s = s.rolling(window=window).mean()
        plt.plot(s.index, list(s), color="blue", label="MA({})".format(window))

    plt.legend()
    plt.xlabel("examples seen")
    plt.ylabel("training loss")
    title = "k={} attn={} task={} bsz=".format(
        flags.FLAGS.k, flags.FLAGS.attn, flags.FLAGS.task, bsz
    )
    plt.title(title)
    outfile = os.path.join(local_dir, "train.pdf")
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("out_dir")

    app.run(_main)
