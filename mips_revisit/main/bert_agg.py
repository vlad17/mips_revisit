"""
Given a path prefix PREFIX of directories containing files

config.json
dev/{summary.json,average_activations.npy,averge_norms.npy}
train/summary.json,average_activations.npy,averge_norms.npy}

In the directory associated with the prefix, outputs the following files

dev_loss.pdf
train_loss.pdf
dev_train_gap.pdf

Plots for all available attention types available in the prefix (e.g.,
soft, topk, topk-50) at all available k values. The plot is a line plot
of the corresponding loss for the different attention type at different
k values (soft is assumed to have k=0).

Similarly, plots the activation and norms distributions for the marginal
activation block aggregates into

train_act.pdf
train_nrm.pdf
dev_act.pdf
dev_nrm.pdf

All results should be from the same task for viz to make sense.
"""

import json
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np
from absl import app, flags

from .. import log
from ..huggingface.run_classifier import main
from ..params import GLUE_TASK_NAMES
from ..sync import exists, sync
from ..utils import import_matplotlib, timeit

flags.DEFINE_string("prefix", None, "prefix directory")

flags.DEFINE_string(
    "cache", None, "cache directory (augogenerated based on task)"
)


flags.DEFINE_bool("overwrite", False, "overwrite previous directory files")

flags.DEFINE_enum("task", None, GLUE_TASK_NAMES, "BERT fine-tuning task")


def _main(_argv):

    log.init()

    prefix = flags.FLAGS.prefix
    expected_files = [
        "dev_loss.pdf",
        "train_loss.pdf",
        "dev_train_gap.pdf",
        "train_act.pdf",
        "train_nrm.pdf",
        "dev_act.pdf",
        "dev_nrm.pdf",
    ]

    for f in expected_files:
        f = os.path.join(prefix, f)
        if exists(f) and not flags.FLAGS.overwrite:
            log.info("file {} exists but --overwrite is not specified", f)
            return

    workdir = flags.FLAGS.cache or "/tmp/bert_agg_{}".format(flags.FLAGS.task)
    log.info("work dir {}", workdir)

    with timeit(name="load results"):
        sync(prefix, workdir,
             '--exclude', '*',
             '--include', 'summary.json',
             '--include', 'average_activations.npy',
             '--include', 'averge_norms.npy')

    _setup_main(workdir)

    with timeit(name="saving outputs"):
        sync(workdir, prefix, '--exclude', '*', '--include', '*.pdf')


def _setup_main(workdir):
    """
    Runs locally, saving all intended output to workdir
    """

    loss = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # nested mapping is
    # eval (dev, train) -> attn (soft, topk, topk-50) -> k -> [loss]
    # list is over seeds

    # similar mapping, but only for topk
    # eval -> attn (soft, topk) -> k -> [marginal act/nrm]
    act = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nrm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, dirnames, filenames in os.walk(workdir):
        if not _is_resultdir(root, dirnames, filenames):
            continue

        config = _get_json(os.path.join(root, "config.json"))
        attn = config["attn"]
        k = config["k"]

        for folder in ["dev", "train"]:
            dir_summary = _get_json(os.path.join(root, folder, "summary.json"))
            dir_act = np.load(
                os.path.join(root, folder, "average_activations.npy")
            )
            dir_nrm = np.load(os.path.join(root, folder, "average_norms.npy"))

            loss[folder][attn][k].append(dir_summary["eval_loss"])
            act[folder][attn][k].append(dir_act.mean(2).mean(1).mean(0))
            nrm[folder][attn][k].append(dir_nrm.mean(1).mean(0))

    trials = set()
    for folder in loss:
        for attn in loss[folder]:
            for k in loss[folder][attn]:
                trials.add(len(loss[folder][attn][k]))
                loss[folder][attn][k] = np.median(loss[folder][attn][k])
                act[folder][attn][k] = np.median(act[folder][attn][k])
                nrm[folder][attn][k] = np.median(nrm[folder][attn][k])

    if len(trials) > 1:
        log.debug(
            "UH-OH! some trials incomplete across runs. trial counts {}",
            trials,
        )
    trials = min(trials)

    attns = set(attn for folder_dict in loss.values() for attn in folder_dict)
    for attn in attns:
        for k in set(loss["dev"][attn]) & set(loss["train"][attn]):
            loss["diff"][attn][k] = (
                loss["dev"][attn][k] - loss["train"][attn][k]
            )

    plt = import_matplotlib()

    for folder in ["dev", "train", "diff"]:
        attn = "soft"
        if 0 in loss[folder][attn]:
            plt.axhline(loss[folder][attn][0], label=attn, color="r")

        attn = "topk"
        if loss[folder][attn]:
            ks = list(loss[folder][attn])
            ks.sort()
            losses = [loss[folder][attn][k] for k in ks]
            plt.plot(ks, losses, ls="--", color="blue", label=attn)

        attns = [attn for attn in loss[folder] if attn.startswith("topk-")]

        for attn, c in zip(attns, colors_transition(plt, len(attns)):
            if loss[folder][attn]:
                ks = list(loss[folder][attn])
                ks.sort()
                losses = [loss[folder][attn][k] for k in ks]
                plt.plot(ks, losses, ls=":", color=c, label=attn)

        if folder == "diff":
            name = "dev-train loss gap"
            filename = "dev_train_gap.pdf"
        else:
            name = folder + " loss"
            filename = "{}_loss.pdf".format(folder)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("loss")
        plt.title("{} for {}".format(name, flags.FLAGS.task))
        plt.savefig(
            os.path.join(workdir, filename), format="pdf", bbox_inches="tight"
        )
        plt.clf()

    for name, shortname in [("activation", "act"), ("norm", "nrm")]:
        distrib = eval(shortname)
        for folder in ["dev", "train"]:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            attn = "soft"
            c = color_cycle[0]
            if 0 in distrib[folder][attn]:
                plt.semilogy(
                    distrib[folder][attn][0], label=attn, color=c, ls="--"
                )

            attn = "topk"
            if distrib[folder][attn]:
                ks = list(distrib[folder][attn])
                ks.sort()
                ks = np.array(ks)
                if len(ks) >= 5:
                    ixs = [i * len(ks) // 4 for i in range(4)]
                    ixs.append(len(ks) - 1)
                    ks = ks[ixs]
                distribs = [distrib[folder][attn][k] for k in ks]

                for k, d, c in zip(ks, distribs, color_cycle[1:])):
                    plt.semilogy(
                        d,
                        ls="-",
                        color=c,
                        alpha=0.7,
                        label="topk k={}".format(k),
                    )

            filename = "{}_{}.pdf".format(folder, shortname)
            plt.legend()
            plt.xlabel("k")
            plt.ylabel(name)
            plt.title("{} distribution for {}".format(name, flags.FLAGS.task))
            plt.savefig(
                os.path.join(workdir, filename),
                format="pdf",
                bbox_inches="tight",
            )
            plt.clf()


def _is_resultdir(root, dirnames, filenames):
    if not (
        "dev" in dirnames
        and "train" in dirnames
        and "config.json" in filenames
    ):
        return False

    for folder in ["dev", "train"]:
        for base in ["average_activations.npy", "average_norms.npy"]:
            if not os.path.isfile(os.path.join(root, folder, base)):
                return False

    return True


def _get_json(filepath):
    with open(filepath, "r") as f:
        d = json.load(f)
    return d

def colors_transition(plt, ncolors):
    import matplotlib.cm as mplcm
    import matplotlib.colors as colors

    cm = plt.get_cmap('spring')
    cNorm  = colors.Normalize(vmin=0, vmax=ncolors-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    return [scalarMap.to_rgba(i) for i in range(ncolors)]



if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("prefix")
    app.run(_main)
