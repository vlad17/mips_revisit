"""
Various utility functions used across several files.
"""

import os
import collections
import hashlib
import itertools
import json
import random
import time
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from . import log


class _timeit:
    def __init__(self):
        self.seconds = 0

    def set_seconds(self, x):
        self.seconds = x


@contextmanager
def timeit(info=True, debug=False, before=None, after=None, name=None):
    """
    Enclose a with-block with to print out block runtime.

    The "after" string should contain an argument for seconds.
    """
    assert not (info and debug)

    def logf(*args, **kwargs):
        pass

    if info:
        logf = log.info
    if debug:
        logf = log.debug
    if before:
        logf(before)
    if name:
        logf(name)
    x = _timeit()
    t = time.time()
    yield x
    x.set_seconds(time.time() - t)
    if name:
        logf("took {:10.2f} sec to" + name, x.seconds)
    if after:
        logf(after, x.seconds)


def _next_seeds(n):
    # deterministically generate seeds for envs
    # not perfect due to correlation between generators,
    # but we can't use urandom here to have replicable experiments
    # https://stats.stackexchange.com/questions/233061
    mt_state_size = 624
    seeds = []
    for _ in range(n):
        state = np.random.randint(2 ** 32, size=mt_state_size)
        digest = hashlib.sha224(state.tobytes()).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)[0]
        seeds.append(int(seed))
        if seeds[-1] is None:
            seeds[-1] = int(state.sum())
    return seeds


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    log.debug("seeding with seed {}", seed)
    np.random.seed(seed)
    rand_seed, tf_seed = _next_seeds(3)
    random.seed(rand_seed)
    tf.random.set_random_seed(tf_seed)


class RollingAverageWindow:
    """Creates an automatically windowed rolling average."""

    def __init__(self, window_size):
        self._window_size = window_size
        self._items = collections.deque([], window_size)
        self._total = 0

    def update(self, value):
        """updates the rolling window"""
        if len(self._items) < self._window_size:
            self._total += value
            self._items.append(value)
        else:
            self._total -= self._items.popleft()
            self._total += value
            self._items.append(value)

    def value(self):
        """returns the current windowed avg"""
        if not self._items:
            return 0
        return self._total / len(self._items)


def import_matplotlib():
    """import and return the matplotlib module in a way that uses
    a display-independent backend (import when generating images on
    servers"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def intfmt(maxval, fill=" "):
    """
    returns the appropriate format string for integers that can go up to
    maximum value maxvalue, inclusive.
    """
    vallen = len(str(maxval))
    return "{:" + fill + str(vallen) + "d}"


def chunkify(iterable, n):
    """
    Break up an iterable into chunks of size n, except for the last chunk,
    if the iterable does not divide evenly.
    """
    # https://stackoverflow.com/questions/8991506
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class OnlineSampler:
    """
    Online sampling algorithm. Given an arbitrary stream of data, this online
    sampler maintains a set of a pre-determined size k that is a simple random
    sample, without replacement, from all observed data in the stream so far.

    In other words, if this sampler has seen n > k data points so far then
    its sample member is a uniformly selected set of k data points among
    those seen.

    Note that observing sample repeatedly does NOT give iid samples between
    updates. Depends on random seed.
    """

    def __init__(self, k):
        self.k = k
        self.n = 0
        self.sample = []

    def update(self, example):
        """
        Observe a datapoint from the incoming stream and possibly include it
        in the sampled set.
        """
        self.n += 1

        if len(self.sample) < self.k:
            self.sample.append(example)
            return

        # We wish to show by induction that the sample list will always be a
        # uniform k-sample without replacement of the n items seen so far
        # through all update calls. When n == k in the base case the unique
        # set of all observed points is vacuously uniformly randomly selected.
        # Now assume the inductive hypothesis holds for n > k.
        # The current set of k samples is a uniform selection without
        # replacement from the n previously observed data points.
        # Now we observe the next example e.
        #
        # Let S be a k-sized set uniformly selected without replacement from
        # our n + 1 points.
        #
        # P{e in S} = Binomial(n, k-1) / Binomial(n, k) = k / (n+1)
        #
        # Then if we include e with the above probability the distribution of S
        # remains uniform (there's a conditioning argument here that I'm too
        # lazy to make).
        #
        # Note we incremented n already at the beginning of this method.

        if random.random() < self.k / self.n:
            i = random.randrange(self.k)
            self.sample[i] = example


def colab_env():
    assert "COLAB_TPU_ADDR" in os.environ
    TPU_ADDRESS = "grpc://" + os.environ["COLAB_TPU_ADDR"]

    from google.colab import auth

    auth.authenticate_user()
    with tf.Session(TPU_ADDRESS) as session:
        # Upload credentials to TPU.
        with open("/content/adc.json", "r") as f:
            auth_info = json.load(f)
        tf.contrib.cloud.configure_gcs(session, credentials=auth_info)

    return TPU_ADDRESS
