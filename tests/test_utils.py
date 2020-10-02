import numpy as np
import pytest

from scikeras.utils import type_of_target


@pytest.mark.parametrize(
    "y, expected",
    [
        (np.array([[1, 0], [0, 1]]), "multiclass-one-hot"),
        (np.array([[1, 0], [1, 1]]), "multilabel-indicator"),
    ],
)
def test_type_of_target(y, expected):
    got = type_of_target(y)
    assert got == expected
