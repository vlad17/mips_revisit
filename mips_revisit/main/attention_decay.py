"""
Usage: python -m mips_revisit.main.attention_decay.py --task mrpc --output_directory gs://bert-mips/attention_decay

Given a task TASK and output directory OUTPUT_DIRECTORY, this file, when run,
performs the following:

* Pull in a pre-trained cased BERT base model.
* Fine-tunes the model to the task using the parameters in the paper.
* Evaluates the model on the test set
* Inspects the softmax activations, before dropout or masking,
  and generates various graphics and diagnostics.

In particular, inside $OUTPUT_DIRECTORY/$TASK, creates the following:

# fine tuning checkpoints
fine_tuning/*.ckpt-*

# link to CV-chosen checkpoint
fine_tuning/final_model.ckpt

# pictures of activation distribution
plots/{layer,head,from_index}.pdf

# detailed TF logs from fine tuning
logs/fine_tuning.txt

# detailed TF logs from evaluation
logs/eval.txt

# tee'd stdout [TODO: tee to gs://?]
logs/stdout.txt
[TODO: do it like this:
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            writer.write("%s = %s\n" % (key, str(result[key])))
]

# array, indexed by "to" position, of average activations
activations.npy

# succinct results overview
summary.json

TODO: structure of summary.json
"""

# [LATER!] compare to fine-tuning and evaluating with top-k and top-k 50% decay

# inferred specific for task:
# --> bert base
# --> bert params
# --> comparison dev/test

# procedure:
# pull cased bert base (stdout: success/timeit)
# fine tuning (stdout: fine tuning params FOR TASK)
#             (stdout: 100-update training procedure logs)
#             (stdout: final timeit)
#             (stdout: final dev eval, compare to paper FOR TASK)
# test eval (stdout) -- whole test
# eval for task, compare to paper for task
#
# get giant activation tensor -- whole test, batched
# make pretty above plots
# (stdout: writing plot to ...)
# (stdout: writing marginal activations to ...npy)
# (print #activations > 1e-1, 1e-2, 1e-3).

import os
import tempfile
import datetime

import tensorflow as tf
from absl import app, flags

from .. import log
from ..bert.finetune_data import get_glue
from ..tpu_setup import colab_env, make_tpu_estimator
from ..utils import import_matplotlib, seed_all, timeit

flags.DEFINE_enum("task", None, ["mrpc"], "BERT fine-tuning task")

flags.DEFINE_string(
    "output_directory", None, "generated artifact output directory"
)


def _main(_argv):
    log.init()

    with timeit(name="load glue data for {}".format(flags.FLAGS.task)):
        glue_dir = get_glue(flags.FLAGS.task)
    log.info("glue data loaded in {}", glue_dir)

    with timeit(name="auth colab tpu"):
        tpu_addr, num_tpu_cores = colab_env()
    log.info("tpu at {}", tpu_addr)

    seed_all(1234)

    out_dir = os.path.join(flags.FLAGS.output_directory, flags.FLAGS.task)
    tf.gfile.MakeDirs(out_dir)

    BERT_MODEL = "cased_L-12_H-768_A-12"

    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 8
    PREDICT_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    MAX_SEQ_LENGTH = 128
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1

    from ..bert import run_classifier, modeling
    from ..params import TEP

    TASK_DATA_DIR = glue_dir
    BERT_PRETRAINED_DIR = "gs://cloud-tpu-checkpoints/bert/" + BERT_MODEL
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, "bert_config.json")
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, "bert_model.ckpt")
    IS_CASED = True

    processors = {
        "cola": run_classifier.ColaProcessor,
        "mnli": run_classifier.MnliProcessor,
        "mrpc": run_classifier.MrpcProcessor,
    }
    processor = processors[flags.FLAGS.task.lower()]()
    label_list = processor.get_labels()

    # Compute number of train and warmup steps from batch size
    train_examples = processor.get_train_examples(TASK_DATA_DIR)
    num_train_steps = int(
        len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS
    )
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=True,
        use_one_hot_embeddings=True,
    )

    estimator_from_checkpoints = make_tpu_estimator(
        ckpt_dir=os.path.join(out_dir, "fine_tuning"),
        tpu_addr=tpu_addr,
        num_tpu_cores=num_tpu_cores,
        model_fn=model_fn,
        batch_sizes=TEP(32, 8, 8),
        save_checkpoints_steps=1000,
    )

    from ..bert import tokenization

    TOKENIZATION_VOCAB = os.path.join(BERT_PRETRAINED_DIR, "vocab.txt")

    tokenizer = tokenization.FullTokenizer(
        TOKENIZATION_VOCAB, do_lower_case=(not IS_CASED)
    )

    # Train the model
    # https://www.tensorflow.org/api_docs/python/tf/estimator/experimental/make_early_stopping_hook
    def model_train(estimator):
        print(
            "MRPC/CoLA on BERT base model normally takes about 2-3 minutes. Please wait..."
        )
        # We'll set sequences to be at most 128 tokens long.
        train_features = run_classifier.convert_examples_to_features(
            train_examples, label_list, MAX_SEQ_LENGTH, tokenizer
        )
        print(
            "***** Started training at {} *****".format(
                datetime.datetime.now()
            )
        )
        print("  Num examples = {}".format(len(train_examples)))
        print("  Batch size = {}".format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = run_classifier.input_fn_builder(
            features=train_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True,
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print(
            "***** Finished training at {} *****".format(
                datetime.datetime.now()
            )
        )

    def model_eval(estimator):
        # Eval the model.
        eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
        eval_features = run_classifier.convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer
        )

        print(
            "***** Started evaluation at {} *****".format(
                datetime.datetime.now()
            )
        )
        print("  Num examples = {}".format(len(eval_examples)))
        print("  Batch size = {}".format(EVAL_BATCH_SIZE))

        # Eval will be slightly WRONG on the TPU because it will truncate
        # the last batch.
        eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_input_fn = run_classifier.input_fn_builder(
            features=eval_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=True,
        )
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        print(
            "***** Finished evaluation at {} *****".format(
                datetime.datetime.now()
            )
        )
        # output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  {} = {}".format(key, str(result[key])))

    def model_predict(estimator, exs):
        # Make predictions on a subset of eval examples
        prediction_examples = exs
        input_features = run_classifier.convert_examples_to_features(
            prediction_examples, label_list, MAX_SEQ_LENGTH, tokenizer
        )
        predict_input_fn = run_classifier.input_fn_builder(
            features=input_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=True,
        )
        predictions = estimator.predict(predict_input_fn)
        predictions = list(predictions)

        for example, prediction in zip(prediction_examples, predictions):
            print(
                "text_a: %s\ntext_b: %s\nlabel:%s\nprediction:%s\n"
                % (
                    example.text_a,
                    example.text_b,
                    str(example.label),
                    prediction["probabilities"],
                )
            )
        return predictions

    model_train(estimator_from_checkpoints)
    model_eval(estimator_from_checkpoints)
    ex = processor.get_dev_examples(TASK_DATA_DIR)[:64]
    ls = model_predict(estimator_from_checkpoints, ex)

    import numpy as np

    # Batch x Layer x Head x Seqlen x SeqLen
    # prediction attentions
    attn = np.stack([x["attn"] for x in ls], axis=0)
    from scipy.special import softmax

    sattn = softmax(attn, -1)
    sattn_sorted = np.sort(sattn) # axis=-1

    plt = import_matplotlib()

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
            sattn_sorted.mean(axis=3).mean(axis=2).mean(axis=1).mean(axis=0),
            ls="-",
            color="red",
            label="marginal",
            lw=1,
        )
        plt.legend()
        plt.xlabel("sorted activation index")
        plt.ylabel("softmax weight")

    # batch x layer x head x from x to

    outfile = os.path.join(out_dir, "im1.pdf")
    with tempfile.NamedTemporaryFile() as tmp:
        semilogy(sattn_sorted.mean(axis=3).mean(axis=2).mean(axis=0))
        plt.title("attn by layer")
        plt.savefig(tmp, format="pdf", bbox_inches="tight")
        tmp.flush()
        tf.io.gfile.copy(tmp.name, outfile)

    outfile = os.path.join(out_dir, "im2.pdf")
    with tempfile.NamedTemporaryFile() as tmp:
        semilogy(sattn_sorted.mean(axis=3).mean(axis=1).mean(axis=0))
        plt.title("attn by head")
        plt.savefig(tmp, format="pdf", bbox_inches="tight")
        tmp.flush()
        tf.io.gfile.copy(tmp.name, outfile)

    outfile = os.path.join(out_dir, "im3.pdf")
    with tempfile.NamedTemporaryFile() as tmp:
        semilogy(sattn_sorted.mean(axis=2).mean(axis=1).mean(axis=0))
        plt.title("attn by from idx")
        plt.savefig(tmp, format="pdf", bbox_inches="tight")
        tmp.flush()
        tf.io.gfile.copy(tmp.name, outfile)


if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("output_directory")
    app.run(_main)
