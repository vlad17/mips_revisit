"""
Given a task and evaluation directory EVAL_DIR, this file, when run,
performs the following:

* Attempt to load the model in $EVAL_DIR, which should have been saved there
  by mips_revisit.main.bert_train
* Evaluate the model on the dev set

Inside $EVAL_DIR, creates the following files:

for each of {dev,train}, in a subdirectory:

average_activations.npy - array of average activations, sorted
layer x head x from x to
from and to are indices into the attention sequence (both span the seq len)
the "to" axis is softmaxed, sorted, and then averaged.

average_norms.npy - array of average attention norms, sorted
layer x head x norm-sorted seq index
for each attention vector before self-attention, we take its norm
this contains the average norm of the i-th largest vector in each
activation

attention_sample.npy - array of sampled attention vectors
batch x layer x seq x dim
a random sample of attention vectors, for the first head.

activation_sample.npy - array of sampled attention inner
product activations
batch x layer x from x to
sorted on the last index by value as above

summary.json - evaluation scores.

act_{layer,head,from_index}.pdf - pictures of activation distribution
based on a sample of the activations

nrm_{layer,head}.pdf - pictures of (marginal) norm distribution,
complete marginal vs layer-conditioned vs head-conditioned.
"""

import json
import os
import shutil

import numpy as np
from absl import app, flags
from scipy.special import softmax

from .. import log
from ..glue import get_glue
from ..huggingface.run_classifier import main
from ..params import GLUE_TASK_NAMES, bert_glue_params
from ..sms import makesms
from ..sync import exists, sync
from ..utils import OnlineSampler, import_matplotlib, seed_all, timeit

flags.DEFINE_enum("task", None, GLUE_TASK_NAMES, "BERT fine-tuning task")

flags.DEFINE_string("eval_dir", None, "evaluation directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory files")

flags.DEFINE_integer(
    "attention_samples", 64, "number of attention samples to grab"
)


def _main(_argv):

    log.init()

    eval_dir = flags.FLAGS.eval_dir
    train_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    for f in train_files:
        f = os.path.join(eval_dir, f)
        if not exists(f):
            log.info("expected file {} to exist but it didn't", f)
            return

    eval_files = [
        "act_layer.pdf",
        "act_head.pdf",
        "act_from_index.pdf",
        "nrm_layer.pdf",
        "nrm_head.pdf",
        "average_activations.npy",
        "average_norms.npy",
        "attention_sample.npy",
        "activation_sample.npy",
        "average_activations.npy",
        "average_norms.npy",
        "attention_sample.npy",
        "activation_sample.npy",
        "summary.json",
    ]

    for d in ["dev", "train"]:
        for f in eval_files:
            f = os.path.join(eval_dir, d, f)
            if exists(f) and not flags.FLAGS.overwrite:
                log.info(
                    "file {} exists and would be overwritten, but "
                    "--overwrite not specified",
                    f,
                )
                return

    glue_data = get_glue(flags.FLAGS.task)
    seed_all(1234)

    makesms("STARTING bert eval\neval_dir={}".format(eval_dir))

    local_dir = os.path.join(
        os.getcwd(), "generated", "eval", flags.FLAGS.task
    )
    log.info("work dir {}", local_dir, glue_data)

    with timeit(name="load train weights"):
        sync(eval_dir, local_dir)

    try:
        _setup_main(local_dir, glue_data)
    except:
        makesms("ERROR in bert eval\neval_dir={}".format(eval_dir))
        raise

    with timeit(name="saving outputs"):
        sync(local_dir, eval_dir)

    log.info("removing work dir {}", local_dir)
    shutil.rmtree(local_dir)

    makesms("COMPLETED bert eval\neval_dir={}".format(eval_dir))


class AttentionObserver:
    # called by eval function
    #
    # each call updates with a batch's new activations
    #
    # attn, or attention vectors, which are the self-attention
    # keys, queries and values (only first head). shape is
    # batch x layer x seq x dim
    #
    # scrs, or scores, which are the result of the matrix
    # multiplication of the attention vectors with their own
    # transpose in self-attention. Here we preserve head info.
    # batch x layer x head x from x to
    #
    # seq, from, to are all indexed over the sequence length
    #
    # no masking is performed.
    #
    # b = batch, l = layer, h = head, d = dim
    # f = from, t = to, s = seq
    # |f| = |t| = |s|
    #
    # note that in the paper the scrs come into this
    # observer normalized by
    # a scale 1 / sqrt(d). For consistency:
    #
    # attn is normalized by d^(-1/4)
    # scrs is normalized by d^(-1/2)

    def __init__(self, attn_sample_size, l, h, s):
        self.attn_reservoir = OnlineSampler(k=attn_sample_size)
        self.acts_reservoir = OnlineSampler(k=attn_sample_size)
        self.sum_activations_lhft = np.zeros((l, h, s, s))
        self.sum_norms_lhs = np.zeros((l, h, s))
        self.s = s
        self.count = 0

    def __call__(self, attn_blsd, scrs_blhft):
        for ex_lsd in attn_blsd:
            self.attn_reservoir.update(ex_lsd)

        for ex_lhft in scrs_blhft:
            ex_lhft = softmax(ex_lhft, axis=-1)
            ex_lhft.sort(axis=-1)
            self.acts_reservoir.update(ex_lhft)

            self.sum_activations_lhft += ex_lhft

        ix = np.arange(int(self.s))
        sqnorm_blhs = scrs_blhft[..., ix, ix]
        sqnorm_blhs[(sqnorm_blhs < 0) | np.isnan(sqnorm_blhs)] = 0
        norm_blhs = np.sqrt(sqnorm_blhs)
        norm_blhs.sort(axis=-1)
        self.sum_norms_lhs += norm_blhs.sum(axis=0)

        self.count += norm_blhs.shape[0]

    def reify(self):
        """
        x = sample size
        """
        attn_xlsd = np.stack(self.attn_reservoir.sample)
        acts_xlhft = np.stack(self.acts_reservoir.sample)
        avg_act_lhft = self.sum_activations_lhft / self.count
        avg_nrm_lhs = self.sum_norms_lhs / self.count

        return (attn_xlsd, acts_xlhft, avg_act_lhft, avg_nrm_lhs)


def _setup_main(local_dir, glue_data):
    """
    Runs locally, saving all intended output to local_dir.
    """

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

    args.output_dir = os.path.join(local_dir, "output")
    args.load_dir = local_dir

    for v in ["dev", "train"]:
        d = os.path.join(local_dir, v)
        os.makedirs(d, exist_ok=True)
        _eval(v, args, d)

    shutil.rmtree(args.output_dir)


def _eval(eval_set_name, args, target_dir):
    """
    Runs evaluation on the given evaluation set
    (dev or train), saving outputs to the target_dir
    """

    bert_base_layers = 12
    bert_base_heads = 12
    seqlen = args.max_seq_length

    observer = AttentionObserver(
        flags.FLAGS.attention_samples,
        bert_base_layers,
        bert_base_heads,
        seqlen,
    )

    with timeit(name="run {} eval".format(eval_set_name)):
        args.eval_set_name = eval_set_name
        res = main(args, observer)

    (attn_xlsd, acts_xlhft, avg_act_lhft, avg_nrm_lhs) = observer.reify()

    log.info("{} results {}", eval_set_name, res)

    # record results
    outfile = os.path.join(target_dir, "summary.json")
    with open(outfile, "w") as f:
        json.dump(res, f, sort_keys=True, indent=4)

    # record activations
    to_persist = [
        (avg_act_lhft, "average_activations.npy"),
        (avg_nrm_lhs, "average_norms.npy"),
        (attn_xlsd, "attention_sample.npy"),
        (acts_xlhft, "activation_sample.npy"),
    ]
    with timeit(name=f"persist {eval_set_name} attn"):
        for val, name in to_persist:
            np.save(os.path.join(target_dir, name), val)

    # TODO: separate bert_pic.py to allow fast overwrite

    attns = acts_xlhft

    # now sort attns
    attns.sort(axis=-1)

    # generate and save plots
    # attns = Batch x Layer x Head x Seqlen (from) x SeqLen (to)
    # marginal = layer x head x from x to
    # recall it's *ordered* on to index.
    plt = import_matplotlib()

    def semilogy(mat_bx, *, scale, xlabel, ylabel, title, outfile, marginal):
        """plots b logscale curves, min max and median values"""
        minv, lo, med, hi, maxv = np.percentile(
            mat_bx, [0, 25, 50, 75, 100], axis=0
        )
        plt.semilogy(med, ls=":", label="median", color="black", lw=2)
        nx = mat_bx.shape[1]
        plt.xlim(0, nx)
        plt.fill_between(range(nx), minv, maxv, color="grey", alpha=0.5)
        plt.semilogy(lo, ls="--", label="25th", color="blue")
        plt.semilogy(hi, ls="--", label="75th", color="blue")
        plt.semilogy(minv, ls="--", label="min", color="grey", alpha=0.75)
        plt.semilogy(maxv, ls="--", label="max", color="grey", alpha=0.75)
        if scale:
            plt.ylim(*scale)

        plt.semilogy(marginal, ls="-", color="red", label="marginal", lw=1)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(outfile, format="pdf", bbox_inches="tight")
        plt.clf()

    marginal = avg_act_lhft.mean(axis=2).mean(axis=1).mean(axis=0)

    def semilogy_acts(mat_bx, *, outfile, title):
        semilogy(
            mat_bx,
            scale=(10 ** -4, 1),
            xlabel="sorted activation index",
            ylabel="softmax weight",
            marginal=marginal,
            title=title,
            outfile=outfile,
        )

    out_dir = target_dir
    outfile = os.path.join(out_dir, "act_layer.pdf")
    log.info("generating {}", outfile)
    semilogy_acts(
        attns.mean(axis=3).mean(axis=2).mean(axis=0),
        outfile=outfile,
        title="attn by layer",
    )

    outfile = os.path.join(out_dir, "act_head.pdf")
    log.info("generating {}", outfile)
    semilogy_acts(
        attns.mean(axis=3).mean(axis=1).mean(axis=0),
        outfile=outfile,
        title="attn by head",
    )

    outfile = os.path.join(out_dir, "act_from_index.pdf")
    log.info("generating {}", outfile)
    semilogy_acts(
        attns.mean(axis=2).mean(axis=1).mean(axis=0),
        outfile=outfile,
        title="attn by from idx",
    )

    marginal = avg_nrm_lhs.mean(axis=1).mean(axis=0)

    def semilogy_nrms(mat_bx, outfile, title):
        semilogy(
            mat_bx,
            scale=None,
            xlabel="sorted norm index",
            ylabel="norm",
            marginal=marginal,
            title=title,
            outfile=outfile,
        )

    outfile = os.path.join(out_dir, "nrm_layer.pdf")
    log.info("generating {}", outfile)
    semilogy_nrms(
        avg_nrm_lhs.mean(axis=1), title="norm by layer", outfile=outfile
    )

    outfile = os.path.join(out_dir, "nrm_head.pdf")
    log.info("generating {}", outfile)
    semilogy_nrms(
        avg_nrm_lhs.mean(axis=0), title="norm by head", outfile=outfile
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("eval_dir")
    app.run(_main)
