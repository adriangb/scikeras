import pytest

from scikeras._utils import pack_keras_model
from scikeras._utils import unpack_keras_model


@pytest.mark.parametrize("obj", [None, "notamodel"])
def test_pack_unpack_not_model(obj):
    with pytest.raises(TypeError):
        pack_keras_model(obj, 0)
    with pytest.raises(TypeError):
        unpack_keras_model(obj, 0)
