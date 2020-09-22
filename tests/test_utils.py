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


def test_compile_with_params_nesting():
    """Test the `compile_with_params` function's
    ability to recursively resolve nested structures.
    """

    class Foo:
        got = dict()

        def __init__(self, **kwargs):
            self.got = kwargs

    class Bar:
        got = dict()

        def __init__(self, **kwargs):
            self.got = kwargs

    res = compile_with_params(
        items={
            "optimizer": [
                Foo,  # 0
                Foo,  # 1
                [Foo, Foo,],  # 2
                {"foo1": Foo, "foo2": Foo,},  # 3
                (Foo,),  # 4
            ],
        },
        params={
            "kwarg": -1,
            "otherkwarg": -2,
            "optimizer__kwarg": 1,
            "optimizer__1__kwarg": 2,
            "optimizer__2__kwarg": 3,
            "optimizer__2__1__kwarg": 4,
            "optimizer__3__foo1__kwarg": 5,
        },
    )

    # Checks
    assert isinstance(res, dict)
    assert set(res.keys()) == {"optimizer"}
    assert set(res["optimizer"][0].got.keys()) == {"kwarg"}
    assert res["optimizer"][0].got["kwarg"] == 1
    assert set(res["optimizer"][1].got.keys()) == {"kwarg"}
    assert res["optimizer"][1].got["kwarg"] == 2
    assert isinstance(res["optimizer"][2], list)
    assert set(res["optimizer"][2][0].got.keys()) == {"kwarg"}
    assert res["optimizer"][2][0].got["kwarg"] == 3
    assert set(res["optimizer"][2][1].got.keys()) == {"kwarg"}
    assert res["optimizer"][2][1].got["kwarg"] == 4
    assert isinstance(res["optimizer"][3], dict)
    assert set(res["optimizer"][3]["foo1"].got.keys()) == {"kwarg"}
    assert res["optimizer"][3]["foo1"].got["kwarg"] == 5
    assert set(res["optimizer"][3]["foo2"].got.keys()) == set()
    assert isinstance(res["optimizer"][4], tuple)
    assert isinstance(res["optimizer"][4][0], Foo)
    assert set(res["optimizer"][4][0].got.keys()) == set()


def test_compile_with_params_dependency():
    """Test the `compile_with_params` function's
    ability to compile the params themselves.
    """

    class Foo:
        got = dict()

        def __init__(self, **kwargs):
            self.got = kwargs

    class Bar:
        got = dict()

        def __init__(self, **kwargs):
            self.got = kwargs

    res = compile_with_params(
        items={"optimizer": Foo,},
        params={"optimizer__bar": Bar, "optimizer__bar__bar_kwarg": 1,},
    )
    # Checks
    assert isinstance(res["optimizer"].got["bar"], Bar)
    assert set(res["optimizer"].got["bar"].got.keys()) == {"bar_kwarg"}
    assert res["optimizer"].got["bar"].got["bar_kwarg"] == 1
