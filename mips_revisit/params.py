"""
Convenience functions for moving around runtime configuration parameters
(e.g., batch size), and neural network training parameters (e.g.,
hyperparameters).

This file also contains the fixed parameters used by previous paper
for various different tasks.
"""


class TEP:
    """
    Any parameter that depends on whether we're training,
    evaluation, or prediction stages can be an instance of this.
    """

    def __init__(self, train=None, eval=None, predict=None):
        self.train = train
        self.eval = eval
        self.predict = predict


# params for the pre-trained model
def bert_pretrain_params(bert_model):
    pretrain_dir = "gs://cloud-tpu-checkpoints/bert/" + bert_model
    return {
        "pretrain_dir": pretrain_dir,
        "config_file": pretrain_dir + "/bert_config.json",
        "init_checkpoint": pretrain_dir + "/bert_model.ckpt",
        "tokens": pretrain_dir + "/vocab.txt",
        "is_cased": "uncased" not in bert_model,
    }


# fine-tuning params
def bert_task_fine_tuning_params(task):
    if task == "mrpc":
        return {
            "batch_sizes": TEP(32, 8, 8),
            "base_lr": 2e-5,
            "max_epochs": 3,
            "max_seq_length": 128,
            "warmup_proportion": 0.1,
        }
    raise ValueError("unknown task " + task)
