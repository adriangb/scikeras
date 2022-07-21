from typing import List

import tensorflow as tf

from scikeras.utils.random_state import tensorflow_random_state


def test_random_state():
    without_seed: List[float] = []
    with_seed: List[float] = []

    for _ in range(10):
        # without setting the random state we should get distinct random values
        # and having set the random state previously does not carry over
        without_seed.append(tf.random.uniform([1]).numpy()[0])
        without_seed.append(tf.random.uniform([1]).numpy()[0])
        # if we set the random state twice, we get the same value
        with tensorflow_random_state(42):
            with_seed.append(tf.random.uniform([1]).numpy()[0])
        with tensorflow_random_state(42):
            with_seed.append(tf.random.uniform([1]).numpy()[0])

    assert len(without_seed) == len(set(without_seed))
    assert len(set(with_seed)) == 1
