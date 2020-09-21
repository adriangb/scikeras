import pytest

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

    instance = Foo(kwarg=7)
    res = compile_with_params(
        items={
            "optimizer": [
                Foo,  # 0
                Foo,  # 1
                [Foo, Foo,],  # 2
                {"foo1": Foo, "foo2": Foo,},  # 3
                "foostring",  # 4
                instance,  # 5
                (Foo,),  # 6
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
            "optimizer__4__kwarg": 6,
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
    assert res["optimizer"][4] == "foostring"
    assert res["optimizer"][5] is instance
    assert set(res["optimizer"][5].got.keys()) == {"kwarg"}
    assert res["optimizer"][5].got["kwarg"] == 7
    assert isinstance(res["optimizer"][6], tuple)
    assert isinstance(res["optimizer"][6][0], Foo)
    assert set(res["optimizer"][6][0].got.keys()) == set()


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
        items={"optimizer": [Foo, [Foo,]],},  # 0  # 1
        params={
            "optimizer__0__bar": Bar,
            "optimizer__0__bar_class": Bar,
            "optimizer__0__bar__bar_kwarg": 10,
            "optimizer__1__0__bar": Bar,
            "optimizer__1__0__bar__bar_kwarg": 11,
        },
    )
    # Checks
    assert isinstance(res["optimizer"][0].got["bar"], Bar)
    assert isinstance(res["optimizer"][0].got["bar_class"], Bar)
    assert set(res["optimizer"][0].got["bar_class"].got.keys()) == set()
    assert set(res["optimizer"][0].got["bar"].got.keys()) == {"bar_kwarg"}
    assert res["optimizer"][0].got["bar"].got["bar_kwarg"] == 10
    assert isinstance(res["optimizer"][1][0].got["bar"], Bar)
    assert set(res["optimizer"][1][0].got["bar"].got.keys()) == {"bar_kwarg"}
    assert res["optimizer"][1][0].got["bar"].got["bar_kwarg"] == 11


def test_compile_with_params_params_to_uncompilable():
    """Test that a warning is raised if parameters are routed
    to a non-class.
    """
    with pytest.warns(
        UserWarning, match="SciKeras does not know how to compile"
    ):
        compile_with_params(
            items={"test": "not_a_class"}, params={"test__kwarg": None},
        )
