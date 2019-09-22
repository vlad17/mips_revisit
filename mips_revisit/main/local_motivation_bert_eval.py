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

from absl import app, flags

from ..motivation.bert_eval import main
from ..params import GLUE_TASK_NAMES
from .. import log

flags.DEFINE_enum("task", None, GLUE_TASK_NAMES, "BERT fine-tuning task")

flags.DEFINE_string("eval_dir", None, "evaluation directory")

flags.DEFINE_bool("overwrite", False, "overwrite previous directory files")

def _main(_argv):
    log.init()
    main(flags.FLAGS.eval_dir,
         flags.FLAGS.overwrite,
         flags.FLAGS.task)

if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("eval_dir")
    app.run(_main)
