"""
Convenience functions for moving around runtime configuration parameters
(e.g., batch size), and neural network training parameters (e.g.,
hyperparameters).

This file also contains the fixed parameters used by previous paper
for various different tasks.
"""

import argparse


def bert_glue_params(task):
    """
    Transcribed from the paper.

    https://arxiv.org/pdf/1810.04805.pdf

    Uses base, not large

    EXPECTS
    task, the GLUE task string

    RETURNS
    an argparse namespace of params used for fine tuning
    pre-trained BERT models on each given task.
    """
    if task == "ner":
        # only this is cased
        bert_model = "bert-base-cased"
        is_lower = False
    else:
        bert_model = "bert-base-uncased"
        is_lower = True

    args = argparse.Namespace()
    args.bert_model = bert_model
    args.task_name = task

    glue = [
            "cola",
        "sst-2",
        "mrpc",
        "sts-b",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "wnli"]

    if task in glue:
        args.max_seq_length = 128
        args.do_lower_case = is_lower
        args.train_batch_size = 32
        args.eval_batch_size = 8
        # paper says 5e-5, github repo says 2e-5...
        args.learning_rate = 5e-5
        args.num_train_epochs = 3
        args.warmup_proportion = 0.1
    else:
        raise KeyError(task)

    return args
