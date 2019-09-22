"""
See main/local_motivation_bert_finetune.py module doc
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

def main(out_dir, overwrite, task, seed):
    seed_all(seed)
    expected_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    for f in expected_files:
        f = os.path.join(out_dir, f)
        if exists(f) and not overwrite:
            log.info(
                "file {} exists and would be overwritten, but "
                "--overwrite not specified",
                f,
            )
            return

    makesms(
        "STARTING bert finetune\nseed={}\nk={}\nattn={}\ntask={}".format(
            seed, flags.FLAGS.k, flags.FLAGS.attn, task
        )
    )

    glue_data = get_glue(task)

    args = bert_glue_params(task)
    args.data_dir = glue_data

    local_dir = os.path.join(
        os.getcwd(), "generated", simplehash(out_dir)
    )
    log.info(
        "using dir {} for local weights (final weights will be in {})",
        local_dir,
        out_dir,
    )
    if overwrite and os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=False)

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

        with timeit(name="saving outputs"):
            sync(local_dir, out_dir)
    except:
        makesms(
            "ERROR in bert finetune\nseed={}\nk={}\nattn={}\ntask={}".format(
                seed,
                flags.FLAGS.k,
                flags.FLAGS.attn,
                task,
            )
        )
        raise

    makesms(
        "COMPLETED bert finetune\nseed={}\nk={}\nattn={}\ntask={}".format(
            seed, flags.FLAGS.k, flags.FLAGS.attn, task
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
    title = "k={} attn={} task={} bsz={}".format(
        flags.FLAGS.k, flags.FLAGS.attn, task, bsz
    )
    plt.title(title)
    outfile = os.path.join(local_dir, "train.pdf")
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.clf()
