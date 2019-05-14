"""
This module defines functions for loading and serving
fine-tuning data for the BERT models.
"""

import os
import subprocess
import sys

from . import log
from .utils import timeit


def get_glue(task):
    """
    Loads the data directory for the given glue task
    into local ./data folder if it's not
    there already. Only checks if the directory is present.

    EXPECTS
    task, a GLUE task string, one of
    mrpc, cola, xnli, mnli

    RETURNS
    the directory containing the task's GLUE data.
    """
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    glue_dir = os.path.join(data_dir, "glue_data")
    os.makedirs(glue_dir, exist_ok=True)

    glue_tasks = {
        "cola": "CoLA",
        "sst-2": "SST-2",
        "mrpc": "MRPC",
        "qqp": "QQP",
        "sts-b": "STS-B",
        "mnli": "MNLI",
        "snli": "SNLI",
        "qnli": "QNLI",
        "rte": "RTE",
        "wnli": "WNLI",
    }
    utask = glue_tasks[task]

    task_dir = os.path.join(glue_dir, utask)

    if os.path.isdir(task_dir):
        log.info("glue data for task {} already present in {}", task, task_dir)
        return task_dir

    with timeit(name="load glue data for {} into {}".format(task, task_dir)):
        return _get_glue(data_dir, glue_dir, task_dir)


def _get_glue(data_dir, glue_dir, task_dir):

    glue_repo_dir = os.path.join(data_dir, "download_glue_repo")

    if not os.path.isdir(glue_repo_dir):
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git",
                glue_repo_dir,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    subprocess.check_call(
        [
            sys.executable,
            os.path.join(glue_repo_dir, "download_glue_data.py"),
            "--data_dir={}".format(glue_dir),
            "--tasks=all",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return task_dir
