import json
import os

import tensorflow as tf

from . import log
from .utils import timeit


def colab_env():
    with timeit(name="auth colab tpu"):
        tpu_addr, num_tpu_cores = _colab_env()
    log.info("tpu at {}", tpu_addr)
    return tpu_addr, num_tpu_cores


def _colab_env():
    assert "COLAB_TPU_ADDR" in os.environ
    TPU_ADDRESS = "grpc://" + os.environ["COLAB_TPU_ADDR"]

    from google.colab import auth

    auth.authenticate_user()
    with tf.Session(TPU_ADDRESS) as session:
        # Upload credentials to TPU.
        with open("/content/adc.json", "r") as f:
            auth_info = json.load(f)
        tf.contrib.cloud.configure_gcs(session, credentials=auth_info)

    tpu_cores = 8
    return TPU_ADDRESS, tpu_cores


def make_tpu_estimator(
    *,
    ckpt_dir,
    tpu_addr,
    num_tpu_cores,
    model_fn,
    batch_sizes,  # expected to be params.TEP instance
    save_checkpoints_steps,
):
    """
    Generates a TPUEstimator for the given model fn,
    train batch size, eval batch size, and predict batch size.
    """
    return tf.contrib.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=_get_run_config(
            ckpt_dir, tpu_addr, num_tpu_cores, save_checkpoints_steps
        ),
        train_batch_size=batch_sizes.train,
        eval_batch_size=batch_sizes.eval,
        predict_batch_size=batch_sizes.predict,
    )


def _get_run_config(ckpt_dir, tpu_addr, num_tpu_cores, save_checkpoints_steps):
    ITERATIONS_PER_LOOP = min(1000, save_checkpoints_steps)
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_addr
    )
    return tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=ckpt_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=num_tpu_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2,
        ),
    )
