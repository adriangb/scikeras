from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pytest
from keras.backend import config as backend_config
from keras.backend import set_floatx

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@pytest.fixture(autouse=True)
def set_floatx_and_backend_config(request: FixtureRequest) -> Iterator[None]:
    current = backend_config.floatx()
    floatx = getattr(request, "param", "float32")
    set_floatx(floatx)
    try:
        yield
    finally:
        set_floatx(current)
