import os
import random

from contextlib import contextmanager
from typing import Generator

import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import config, ops


DIGITS = frozenset(str(i) for i in range(10))


@contextmanager
def tensorflow_random_state(seed: int) -> Generator[None, None, None]:
    # Save values
    origin_gpu_det = os.environ.get("TF_DETERMINISTIC_OPS", None)
    orig_random_state = random.getstate()
    orig_np_random_state = np.random.get_state()
    if context.executing_eagerly():
        tf_random_seed = context.global_seed()
    else:
        tf_random_seed = ops.get_default_graph().seed

    determism_enabled = config.is_op_determinism_enabled()
    config.enable_op_determinism()

    # Set values
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    yield

    # Reset values
    if origin_gpu_det is not None:
        os.environ["TF_DETERMINISTIC_OPS"] = origin_gpu_det
    else:
        os.environ.pop("TF_DETERMINISTIC_OPS")
    random.setstate(orig_random_state)
    np.random.set_state(orig_np_random_state)
    tf.random.set_seed(tf_random_seed)
    if not determism_enabled:
        config.disable_op_determinism()
