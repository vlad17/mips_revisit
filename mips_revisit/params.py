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
