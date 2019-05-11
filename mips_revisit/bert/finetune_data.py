"""
This module defines functions for loading and serving
fine-tuning data for the BERT models.
"""

import os
import subprocess
import sys

def get_glue(task):
    """
    Loads the data directory for the given glue task
    into local ./data folder if it's not
    there already. Only checks if the directory is present.
    """
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    glue_dir = os.path.join(data_dir, "glue_data")
    os.makedirs(glue_dir, exist_ok=True)
    task_dir = os.path.join(glue_dir, task)

    if os.path.isdir(task_dir):
        return task_dir

    glue_repo_dir = os.path.join(data_dir, 'download_glue_repo')

    # TODO: create a wrapper around log.debug which captures streams
    # line-by-line to pass these along

    if not os.path.isdir(glue_repo_dir):
        subprocess.check_call([
            'git',
            'clone',
            'https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git',
            glue_repo_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.check_call([
        sys.executable,
        os.path.join(glue_repo_dir, 'download_glue_data.py'),
        "--data_dir={}".format(glue_dir),
        "--tasks={}".format(task.upper())],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return os.path.join(glue_dir, task)
