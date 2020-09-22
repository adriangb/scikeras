from inspect import isclass

import pytest

from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module
from tensorflow.keras import optimizers as optimizers_module

from scikeras._utils import (
    compile_with_params,
    pack_keras_model,
    route_params,
    unpack_keras_model,
)


@pytest.mark.parametrize("obj", [None, "notamodel"])
def test_pack_unpack_not_model(obj):
    with pytest.raises(TypeError):
        pack_keras_model(obj, 0)
    with pytest.raises(TypeError):
        unpack_keras_model(obj, 0)


def test_route_params():
    """Test the `route_params` function.
    """
    params = {"model__foo": object()}
    destination = "model"
    pass_filter = set()
    out = route_params(params, destination, pass_filter)
    assert out["foo"] is params["model__foo"]
