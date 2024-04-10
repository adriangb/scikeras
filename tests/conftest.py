from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pytest
from keras import backend

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@pytest.fixture(autouse=True)
def set_floatx_and_backend_config(request: FixtureRequest) -> Iterator[None]:
    current = backend.floatx()
    floatx = getattr(request, "param", "float32")
    backend.set_floatx(floatx)
    try:
        yield
    finally:
        backend.set_floatx(current)
