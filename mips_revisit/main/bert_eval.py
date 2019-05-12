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
summary.json - dev set evaluation scores.
"""

import os
import json
import shutil
import tempfile

import tensorflow as tf
from absl import app, flags
import numpy as np

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

    with timeit(name="load train weights"):
        for f in train_files:
            tf.gfile.Copy(
                os.path.join(eval_dir, f), os.path.join(local_dir, f)
            )

    args.output_dir = os.path.join(local_dir, "output")
    args.load_dir = local_dir

    res, marginal, attns = main(args, return_attn=True, attn_subsample_size=64)

    log.info("dev results {}", res)

    # record results
    outfile = os.path.join(local_dir, "summary.json")
    with open(outfile, 'w') as f:
        json.dump(res, f, sort_keys = True, indent = 4)
    upload = os.path.join(eval_dir, "summary.json")
    tf.gfile.Copy(outfile, upload, overwrite=True)
    log.info("uploaded dev results to {}", upload)

    # record marginal activations
    np.save(os.path.join(local_dir, "activations.npy"), marginal)
    upload = os.path.join(eval_dir, "activations.npy")
    tf.gfile.Copy(os.path.join(local_dir, "activations.npy"),
                  upload, overwrite=True)
    log.info("uploaded marginal activations to {}", upload)

    # generate and save plots
    # attns = Batch x Layer x Head x Seqlen (from) x SeqLen (to)
    # marginal = layer x head x from x to
    # recall it's *ordered* on to index.
    plt = import_matplotlib()
    marginal = marginal.mean(axis=2).mean(axis=1).mean(axis=0)

    def semilogy(mat_bx):
        """plots b logscale curves, min max and median values"""
        lo, med, hi = (x(mat_bx, axis=0) for x in [np.min, np.median, np.max])
        plt.semilogy(med, ls=":", label="median", color="black", lw=2)
        nx = mat_bx.shape[1]
        plt.xlim(0, nx)
        plt.fill_between(range(nx), lo, hi, color="grey", alpha=0.5)
        plt.semilogy(lo, ls="--", label="min", color="grey")
        plt.semilogy(hi, ls="--", label="max", color="grey")
        plt.ylim(10 ** -3, 1)

        plt.semilogy(
            marginal,
            ls="-",
            color="red",
            label="marginal",
            lw=1,
        )
        plt.legend()
        plt.xlabel("sorted activation index")
        plt.ylabel("softmax weight")

    out_dir = os.path.join(local_dir, "plots")
    os.makedirs(out_dir)

    outfile = os.path.join(out_dir, "layer.pdf")
    log.info("generating {}", outfile)
    semilogy(attns.mean(axis=3).mean(axis=2).mean(axis=0))
    plt.title("attn by layer")
    plt.savefig(outfile, format="pdf", bbox_inches="tight")

    outfile = os.path.join(out_dir, "head.pdf")
    log.info("generating {}", outfile)
    semilogy(attns.mean(axis=3).mean(axis=1).mean(axis=0))
    plt.title("attn by head")
    plt.savefig(outfile, format="pdf", bbox_inches="tight")

    outfile = os.path.join(out_dir, "from_index.pdf")
    log.info("generating {}", outfile)
    semilogy(attns.mean(axis=2).mean(axis=1).mean(axis=0))
    plt.title("attn by from idx")
    plt.savefig(outfile, format="pdf", bbox_inches="tight")

    upload = os.path.join(eval_dir, "plots")
    log.info("uploading {} to {}", out_dir, upload)
    tf.gfile.MakeDirs("plots")
    for f in eval_files:
        if not f.startswith("plots/"):
            continue
        dst = os.path.join(upload, f)
        src = os.path.join(local_dir, f)
        tf.gfile.Copy(src, dst, overwrite=True)

    log.info("removing work dir {}", local_dir)
    shutil.rmtree(local_dir)



if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("eval_dir")
    app.run(_main)
