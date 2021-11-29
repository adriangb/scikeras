"""Wrapper for using the Scikit-Learn API with Keras models.
"""
import inspect
import warnings

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Set, Tuple, Type, Union

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight, check_array, check_X_y
from tensorflow.keras import losses as losses_module
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

from scikeras._utils import (
    accepts_kwargs,
    get_loss_class_function_or_string,
    get_metric_class,
    get_optimizer_class,
    has_param,
    route_params,
    try_to_convert_strings_to_classes,
    unflatten_params,
)
from scikeras.utils import loss_name, metric_name
from scikeras.utils.random_state import tensorflow_random_state
from scikeras.utils.transformers import ClassifierLabelEncoder, RegressorTargetEncoder


class BaseWrapper(BaseEstimator):
    """Implementation of the scikit-learn classifier API for Keras.

    Below are a list of SciKeras specific parameters. For details on other parameters,
    please see the see the `tf.keras.Model documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

    Parameters
    ----------
    model : Union[None, Callable[..., tf.keras.Model], tf.keras.Model], default None
        Used to build the Keras Model. When called,
        must return a compiled instance of a Keras Model
        to be used by `fit`, `predict`, etc.
        If None, you must implement ``_keras_build_fn``.
    optimizer : Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]], default "rmsprop"
        This can be a string for Keras' built in optimizers,
        an instance of tf.keras.optimizers.Optimizer
        or a class inheriting from tf.keras.optimizers.Optimizer.
        Only strings and classes support parameter routing.
    loss : Union[Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None], default None
        The loss function to use for training.
        This can be a string for Keras' built in losses,
        an instance of tf.keras.losses.Loss
        or a class inheriting from tf.keras.losses.Loss .
        Only strings and classes support parameter routing.
    random_state : Union[int, np.random.RandomState, None], default None
        Set the Tensorflow random number generators to a
        reproducible deterministic state using this seed.
        Pass an int for reproducible results across multiple
        function calls.
    warm_start : bool, default False
        If True, subsequent calls to fit will _not_ reset
        the model parameters but *will* reset the epoch to zero.
        If False, subsequent fit calls will reset the entire model.
        This has no impact on partial_fit, which always trains
        for a single epoch starting from the current epoch.
    batch_size : Union[int, None], default None
        Number of samples per gradient update.
        This will be applied to both `fit` and `predict`. To specify different numbers,
        pass `fit__batch_size=32` and `predict__batch_size=1000` (for example).
        To auto-adjust the batch size to use all samples, pass `batch_size=-1`.

    Attributes
    ----------
    model_ : tf.keras.Model
        The instantiated and compiled Keras Model. For pre-built models, this
        will just be a reference to the passed Model instance.
    history_ : Dict[str, List[Any]]
        Dictionary of the format ``{metric_str_name: [epoch_0_data, epoch_1_data, ..., epoch_n_data]}``.
    initialized_ : bool
        True if this estimator has been initialized (i.e. predict can be called upon it).
        Note that this does not guarantee that the model is "fitted": if ``BaseWrapper.initialize``
        was called instead of fit the model wil likely have random weights.
    target_encoder_ : sklearn-transformer
        Transformer used to pre/post process the target y.
    feature_encoder_ : sklearn-transformer
        Transformer used to pre/post process the features/input X.
    n_outputs_expected_ : int
        The number of outputs the Keras Model is expected to have, as determined by ``target_transformer_``.
    target_type_ : str
        One of:

        * 'continuous': y is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': y is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': y contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': y contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': y is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': y is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': y is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.
    y_shape_ : Tuple[int]
        Shape of the target y that the estimator was fitted on.
    y_dtype_ : np.dtype
        Dtype of the target y that the estimator was fitted on.
    X_shape_ : Tuple[int]
        Shape of the input X that the estimator was fitted on.
    X_dtype_ : np.dtype
        Dtype of the input X that the estimator was fitted on.
    n_features_in_ : int
        The number of features seen during `fit`.
    """

    _tags = {
        "poor_score": True,
        "multioutput": True,
    }

    _fit_kwargs = {
        # parameters destined to keras.Model.fit
        "batch_size",
        "epochs",
        "verbose",
        "validation_split",
        "shuffle",
        "class_weight",
        "sample_weight",
        "initial_epoch",
        "validation_steps",
        "validation_batch_size",
        "validation_freq",
    }

    _predict_kwargs = {
        # parameters destined to keras.Model.predict
        "batch_size",
        "verbose",
        "steps",
    }

    _compile_kwargs = {
        # parameters destined to keras.Model.compile
        "optimizer",
        "loss",
        "metrics",
        "loss_weights",
        "weighted_metrics",
        "run_eagerly",
    }

    _wrapper_params = {
        # parameters consumed by the wrappers themselves
        "warm_start",
        "random_state",
    }

    _routing_prefixes = {
        "model",
        "fit",
        "compile",
        "predict",
        "optimizer",
        "loss",
        "metrics",
    }

    def __init__(
        self,
        model: Union[None, Callable[..., tf.keras.Model], tf.keras.Model] = None,
        *,
        build_fn: Union[
            None, Callable[..., tf.keras.Model], tf.keras.Model
        ] = None,  # for backwards compatibility
        warm_start: bool = False,
        random_state: Union[int, np.random.RandomState, None] = None,
        optimizer: Union[
            str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]
        ] = "rmsprop",
        loss: Union[
            Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None
        ] = None,
        metrics: Union[
            List[
                Union[
                    str,
                    tf.keras.metrics.Metric,
                    Type[tf.keras.metrics.Metric],
                    Callable,
                ]
            ],
            None,
        ] = None,
        batch_size: Union[int, None] = None,
        validation_batch_size: Union[int, None] = None,
        verbose: int = 1,
        callbacks: Union[
            List[Union[tf.keras.callbacks.Callback, Type[tf.keras.callbacks.Callback]]],
            None,
        ] = None,
        validation_split: float = 0.0,
        shuffle: bool = True,
        run_eagerly: bool = False,
        epochs: int = 1,
        **kwargs,
    ):

        # Parse hardcoded params
        self.model = model
        self.build_fn = build_fn
        self.warm_start = warm_start
        self.random_state = random_state
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.run_eagerly = run_eagerly
        self.epochs = epochs

        # Unpack kwargs
        vars(self).update(**kwargs)

        # Save names of kwargs into set
        if kwargs:
            self._user_params = set(kwargs)

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def current_epoch(self) -> int:
        """Returns the current training epoch.

        Returns
        -------
        int
            Current training epoch.
        """
        if not hasattr(self, "history_"):
            return 0
        return len(self.history_["loss"])

    @staticmethod
    def _validate_sample_weight(
        X: np.ndarray, sample_weight: Union[np.ndarray, Iterable],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate that the passed sample_weight and ensure it is a Numpy array."""
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.dtype(tf.keras.backend.floatx())
        )
        if np.all(sample_weight == 0):
            raise ValueError(
                "No training samples had any weight; only zeros were passed in sample_weight."
                " That means there's nothing to train on by definition, so training can not be completed."
            )
        return X, sample_weight

    def _check_model_param(self):
        """Checks ``model`` and returns model building
        function to use.

        Raises
        ------
            ValueError: if ``self.model`` is not valid.
        """
        model = self.model
        build_fn = self.build_fn
        if model is None and build_fn is not None:
            model = build_fn
            warnings.warn(
                "``build_fn`` will be renamed to ``model`` in a future release,"
                " at which point use of ``build_fn`` will raise an Error instead."
            )
        if model is None:
            # no model, use this class' _keras_build_fn
            if not hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "If not using the ``build_fn`` param, "
                    "you must implement ``_keras_build_fn``"
                )
            final_build_fn = self._keras_build_fn
        elif isinstance(model, Model):
            # pre-built Keras Model
            def final_build_fn():
                return model

        elif inspect.isfunction(model):
            if hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "This class cannot implement ``_keras_build_fn`` if"
                    " using the `model` parameter"
                )
            # a callable method/function
            final_build_fn = model
        else:
            raise TypeError(
                "``model`` must be a callable, a Keras Model instance or None"
            )

        return final_build_fn

    def _get_compile_kwargs(self):
        """Convert all __init__ params destined to
        `compile` into valid kwargs for `Model.compile` by parsing
        routed parameters and compiling optimizers, losses and metrics
        as needed.

        Returns
        -------
        dict
            Dictionary of kwargs for `Model.compile`.
        """
        init_params = self.get_params()
        compile_kwargs = route_params(
            init_params, destination="compile", pass_filter=self._compile_kwargs,
        )
        compile_kwargs["optimizer"] = try_to_convert_strings_to_classes(
            compile_kwargs["optimizer"], get_optimizer_class
        )
        compile_kwargs["optimizer"] = unflatten_params(
            items=compile_kwargs["optimizer"],
            params=route_params(
                init_params, destination="optimizer", pass_filter=set(), strict=True,
            ),
        )
        compile_kwargs["loss"] = try_to_convert_strings_to_classes(
            compile_kwargs["loss"], get_loss_class_function_or_string
        )
        compile_kwargs["loss"] = unflatten_params(
            items=compile_kwargs["loss"],
            params=route_params(
                init_params, destination="loss", pass_filter=set(), strict=False,
            ),
        )
        compile_kwargs["metrics"] = try_to_convert_strings_to_classes(
            compile_kwargs["metrics"], get_metric_class
        )
        compile_kwargs["metrics"] = unflatten_params(
            items=compile_kwargs["metrics"],
            params=route_params(
                init_params, destination="metrics", pass_filter=set(), strict=False,
            ),
        )
        return compile_kwargs

    def _build_keras_model(self):
        """Build the Keras model.

        This method will process all arguments and call the model building
        function with appropriate arguments.

        Returns
        -------
        tensorflow.keras.Model
            Instantiated and compiled keras Model.
        """
        # dynamically build model, i.e. final_build_fn builds a Keras model

        # determine what type of build_fn to use
        final_build_fn = self._check_model_param()

        # collect parameters
        params = self.get_params()
        build_params = route_params(
            params,
            destination="model",
            pass_filter=getattr(self, "_user_params", set()),
            strict=True,
        )
        compile_kwargs = None
        if has_param(final_build_fn, "meta") or accepts_kwargs(final_build_fn):
            # build_fn accepts `meta`, add it
            build_params["meta"] = self._get_metadata()
        if has_param(final_build_fn, "compile_kwargs") or accepts_kwargs(
            final_build_fn
        ):
            # build_fn accepts `compile_kwargs`, add it
            compile_kwargs = self._get_compile_kwargs()
            build_params["compile_kwargs"] = compile_kwargs
        if has_param(final_build_fn, "params") or accepts_kwargs(final_build_fn):
            # build_fn accepts `params`, i.e. all of get_params()
            build_params["params"] = self.get_params()

        # build model
        if self._random_state is not None:
            with tensorflow_random_state(self._random_state):
                model = final_build_fn(**build_params)
        else:
            model = final_build_fn(**build_params)

        return model

    def _ensure_compiled_model(self) -> None:
        # compile model if user gave us an un-compiled model
        if not (hasattr(self.model_, "loss") and hasattr(self.model_, "optimizer")):
            kw = self._get_compile_kwargs()
            self.model_.compile(**kw)

    def _fit_keras_model(
        self,
        X: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        sample_weight: Union[np.ndarray, None],
        warm_start: bool,
        epochs: int,
        initial_epoch: int,
        **kwargs,
    ) -> None:
        """Fits the Keras model.

        Parameters
        ----------
        X : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Training samples, as accepted by tf.keras.Model
        y : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Target data, as accepted by tf.keras.Model
        sample_weight : Union[np.ndarray, None]
            Sample weights. Ignored by Keras if None.
        warm_start : bool
            If True, don't don't overwrite
            the history_ attribute and append to it instead.
        epochs : int
            Number of epochs for which the model will be trained.
        initial_epoch : int
            Epoch at which to begin training.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.

        Returns
        -------
        BaseWrapper
            A reference to the instance that can be chain called
            (ex: instance.fit(X,y).transform(X) )
        """

        # Make sure model has a loss function
        loss = self.model_.loss
        no_loss = False
        if isinstance(loss, list) and not any(
            callable(loss_) or isinstance(loss_, str) for loss_ in loss
        ):
            no_loss = True
        if isinstance(loss, dict) and not any(
            callable(loss_) or isinstance(loss_, str) for loss_ in loss.values()
        ):
            no_loss = True
        if no_loss:
            raise ValueError(
                "No valid loss function found."
                " You must provide a loss function to train."
                "\n\nTo resolve this issue, do one of the following:"
                "\n 1. Provide a loss function via the loss parameter."
                "\n 2. Compile your model with a loss function inside the"
                " model-building method."
                "\n\nSee https://www.adriangb.com/scikeras/stable/advanced.html#compilation-of-model"
                " for more information on compiling SciKeras models."
                "\n\nSee https://www.tensorflow.org/api_docs/python/tf/keras/losses"
                " for more information on Keras losses."
            )

        # collect parameters
        params = self.get_params()
        fit_args = route_params(params, destination="fit", pass_filter=self._fit_kwargs)
        fit_args["sample_weight"] = sample_weight
        fit_args["epochs"] = initial_epoch + epochs
        fit_args["initial_epoch"] = initial_epoch
        fit_args.update(kwargs)
        for bs_kwarg in ("batch_size", "validation_batch_size"):
            if bs_kwarg in fit_args:
                if fit_args[bs_kwarg] == -1:
                    try:
                        fit_args[bs_kwarg] = X.shape[0]
                    except AttributeError:
                        raise ValueError(
                            f"`{bs_kwarg}=-1` requires that `X` implement `shape`"
                        )
        fit_args = {k: v for k, v in fit_args.items() if not k.startswith("callbacks")}
        fit_args["callbacks"] = self._fit_callbacks

        if self._random_state is not None:
            with tensorflow_random_state(self._random_state):
                hist = self.model_.fit(x=X, y=y, **fit_args)
        else:
            hist = self.model_.fit(x=X, y=y, **fit_args)

        if not warm_start or not hasattr(self, "history_") or initial_epoch == 0:
            self.history_ = defaultdict(list)

        for key, val in hist.history.items():
            try:
                key = metric_name(key)
            except ValueError as e:
                # Keras puts keys like "val_accuracy" and "loss" and
                # "val_loss" in hist.history
                if "Unknown metric function" not in str(e):
                    raise e
            self.history_[key] += val

    def _check_model_compatibility(self, y: np.ndarray) -> None:
        """Checks that the model output number and y shape match.

        This is in place to avoid cryptic TF errors.
        """
        # check if this is a multi-output model
        if getattr(self, "n_outputs_expected_", None):
            # n_outputs_expected_ is generated by data transformers
            # we recognize the attribute but do not force it to be
            # generated
            if self.n_outputs_expected_ != len(self.model_.outputs):
                raise ValueError(
                    "Detected a Keras model input of size"
                    f" {self.n_outputs_expected_ }, but {self.model_} has"
                    f" {len(self.model_.outputs)} outputs"
                )
        # check that if the user gave us a loss function it ended up in
        # the actual model
        init_params = inspect.signature(self.__init__).parameters
        if "loss" in init_params:
            default_val = init_params["loss"].default
            if all(
                isinstance(x, (str, losses_module.Loss, type))
                for x in [self.loss, self.model_.loss]
            ):  # filter out loss list/dicts/etc.
                if default_val is not None:
                    default_val = loss_name(default_val)
                given = loss_name(self.loss)
                got = loss_name(self.model_.loss)
                if given != default_val and got != given:
                    raise ValueError(
                        f"loss={self.loss} but model compiled with {self.model_.loss}."
                        " Data may not match loss function!"
                    )

    def _validate_data(
        self, X=None, y=None, reset: bool = False, y_numeric: bool = False
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Validate input arrays and set or check their meta-parameters.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape \
           (n_samples, n_features)
            The input samples. If None, ``check_array`` is called on y and
            ``check_X_y`` is called otherwise.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,), default=None
            The targets. If None, ``check_array`` is called on X and
            ``check_X_y`` is called otherwise.
        reset : bool, default=False
            If True, override all meta attributes.
            If False, verify that they haven't changed.
        y_numeric : bool, default = False
            If True, ensure y is a numeric dtype.
            If False, allow non-numeric y to pass through.

        Returns
        -------
        Tuple[np.ndarray, Union[np.ndarray, None]]
            The validated input.
        """

        def _check_array_dtype(arr, force_numeric):
            if not isinstance(arr, np.ndarray):
                return _check_array_dtype(np.asarray(arr), force_numeric=force_numeric)
            elif (
                arr.dtype.kind not in ("O", "U", "S") or not force_numeric
            ):  # object, unicode or string
                # already numeric
                return None  # check_array won't do any casting with dtype=None
            else:
                # default to TFs backend float type
                # instead of float64 (sklearn's default)
                return tf.keras.backend.floatx()

        if X is not None and y is not None:
            X, y = check_X_y(
                X,
                y,
                allow_nd=True,  # allow X to have more than 2 dimensions
                multi_output=True,  # allow y to be 2D
                dtype=None,
            )

        if y is not None:
            y = check_array(
                y,
                ensure_2d=False,
                allow_nd=False,
                dtype=_check_array_dtype(y, force_numeric=y_numeric),
            )
            y_dtype_ = y.dtype
            y_ndim_ = y.ndim
            if reset:
                self.target_type_ = self._type_of_target(y)
                self.y_dtype_ = y_dtype_
                self.y_ndim_ = y_ndim_
            else:
                if not np.can_cast(y_dtype_, self.y_dtype_):
                    raise ValueError(
                        f"Got y with dtype {y_dtype_},"
                        f" but this {self.__name__} expected {self.y_dtype_}"
                        f" and casting from {y_dtype_} to {self.y_dtype_} is not safe!"
                    )
                if self.y_ndim_ != y_ndim_:
                    raise ValueError(
                        f"y has {y_ndim_} dimensions, but this {self.__name__}"
                        f" is expecting {self.y_ndim_} dimensions in y."
                    )
        if X is not None:
            X = check_array(
                X, allow_nd=True, dtype=_check_array_dtype(X, force_numeric=True)
            )
            X_dtype_ = X.dtype
            X_shape_ = X.shape
            n_features_in_ = X.shape[1]
            if reset:
                self.X_dtype_ = X_dtype_
                self.X_shape_ = X_shape_
                self.n_features_in_ = n_features_in_
            else:
                if not np.can_cast(X_dtype_, self.X_dtype_):
                    raise ValueError(
                        f"Got X with dtype {X_dtype_},"
                        f" but this {self.__name__} expected {self.X_dtype_}"
                        f" and casting from {X_dtype_} to {self.X_dtype_} is not safe!"
                    )
                if len(X_shape_) != len(self.X_shape_):
                    raise ValueError(
                        f"X has {len(X_shape_)} dimensions, but this {self.__name__}"
                        f" is expecting {len(self.X_shape_)} dimensions in X."
                    )
        return X, y

    def _type_of_target(self, y: np.ndarray) -> str:
        return type_of_target(y)

    @property
    def target_encoder(self):
        """Retrieve a transformer for targets / y.

        Metadata will be collected from ``get_metadata`` if
        the transformer implements that method.
        Override this method to implement a custom data transformer
        for the target.

        Returns
        -------
        target_encoder
            Transformer implementing the sklearn transformer
            interface.
        """
        return FunctionTransformer()

    @property
    def feature_encoder(self):
        """Retrieve a transformer for features / X.

        Metadata will be collected from ``get_metadata`` if
        the transformer implements that method.
        Override this method to implement a custom data transformer
        for the features.

        Returns
        -------
        sklearn transformer
            Transformer implementing the sklearn transformer
            interface.
        """
        return FunctionTransformer()

    def fit(self, X, y, sample_weight=None, **kwargs) -> "BaseWrapper":
        """Constructs a new model with ``model`` & fit the model to ``(X, y)``.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Training samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.

        Warnings
        --------
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``fit`` is not supported by the Scikit-Learn API,
            and will be removed in a future version of SciKeras.
            These parameters can also be specified by prefixing ``fit__`` to a parameter at initialization
            (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``)
            or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``).

        Returns
        -------
        BaseWrapper
            A reference to the instance that can be chain called (``est.fit(X,y).transform(X)``).
        """
        # epochs via kwargs > fit__epochs > epochs
        kwargs["epochs"] = kwargs.get(
            "epochs", getattr(self, "fit__epochs", self.epochs)
        )
        kwargs["initial_epoch"] = kwargs.get("initial_epoch", 0)

        self._fit(
            X=X, y=y, sample_weight=sample_weight, warm_start=self.warm_start, **kwargs,
        )

        return self

    @property
    def initialized_(self) -> bool:
        """Checks if the estimator is intialized.

        Returns
        -------
        bool
            True if the estimator is initialized (i.e., it can
            be used for inference or is ready to train),
            otherwise False.
        """
        return hasattr(self, "model_")

    def _initialize_callbacks(self) -> None:
        params = self.get_params()

        def initialize(destination: str):
            if params.get(destination) is not None:
                callback_kwargs = route_params(
                    params, destination=destination, pass_filter=set()
                )
                callbacks = unflatten_params(
                    items=params[destination], params=callback_kwargs
                )
                if isinstance(callbacks, Mapping):
                    # Keras does not officially support dicts, convert to a list
                    callbacks = list(callbacks.values())
                elif isinstance(callbacks, tf.keras.callbacks.Callback):
                    # a single instance, not officially supported so wrap in a list
                    callbacks = [callbacks]
                err = False
                if not isinstance(callbacks, List):
                    err = True
                for cb in callbacks:
                    if isinstance(cb, List):
                        for nested_cb in cb:
                            if not isinstance(nested_cb, tf.keras.callbacks.Callback):
                                err = True
                    elif not isinstance(cb, tf.keras.callbacks.Callback):
                        err = True
                if err:
                    raise TypeError(
                        "If specified, ``callbacks`` must be one of:"
                        "\n - A dict of string keys with callbacks or lists of callbacks as values"
                        "\n - A list of callbacks or lists of callbacks"
                        "\n - A single callback"
                        "\nWhere each callback can be a instance of `tf.keras.callbacks.Callback` or a sublass of it to be compiled by SciKeras"
                    )
            else:
                callbacks = []
            return callbacks

        all_callbacks = initialize("callbacks")
        self._fit_callbacks = all_callbacks + initialize("fit__callbacks")
        self._predict_callbacks = all_callbacks + initialize("predict__callbacks")

    def _initialize(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Handle random state
        if isinstance(self.random_state, np.random.RandomState):
            # Keras needs an integer
            # we sample an integer and use that as a seed
            # Given the same RandomState, the seed will always be
            # the same, thus giving reproducible results
            state = self.random_state.get_state()
            r = np.random.RandomState()
            r.set_state(state)
            self._random_state = r.randint(low=1)
        else:
            # int or None
            self._random_state = self.random_state

        X, y = self._validate_data(X, y, reset=True)

        self.target_encoder_ = self.target_encoder.fit(y)
        target_metadata = getattr(self.target_encoder_, "get_metadata", dict)()
        vars(self).update(**target_metadata)
        self.feature_encoder_ = self.feature_encoder.fit(X)
        feature_meta = getattr(self.feature_encoder, "get_metadata", dict)()
        vars(self).update(**feature_meta)

        self.model_ = self._build_keras_model()
        self._initialize_callbacks()

        return X, y

    def initialize(self, X, y=None) -> "BaseWrapper":
        """Initialize the model without any fitting.

        You only need to call this model if you explicitly do not want to do any fitting
        (for example with a pretrained model). You should _not_ call this
        right before calling ``fit``, calling ``fit`` will do this automatically.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
                Training samples where n_samples is the number of samples
                and `n_features` is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs), default None
            True labels for X.

        Returns
        -------
        BaseWrapper
            A reference to the BaseWrapper instance for chained calling.
        """
        self._initialize(X, y)
        return self  # to allow chained calls like initialize(...).predict(...)

    def _fit(
        self,
        X,
        y,
        sample_weight,
        warm_start: bool,
        epochs: int,
        initial_epoch: int,
        **kwargs,
    ) -> None:
        """Constructs a new model with ``model`` & fit the model to ``(X, y)``.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
        y :Union[array-like, sparse matrix, dataframe] of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        warm_start : bool
            If True, don't rebuild the model.
        epochs : int
            Number of passes over the entire dataset for which to train the
            model.
        initial_epoch : int
            Epoch at which to begin training.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.
        """
        # Data checks
        if not ((self.warm_start or warm_start) and self.initialized_):
            X, y = self._initialize(X, y)
        else:
            X, y = self._validate_data(X, y)
        self._ensure_compiled_model()

        if sample_weight is not None:
            X, sample_weight = self._validate_sample_weight(X, sample_weight)

        y = self.target_encoder_.transform(y)
        X = self.feature_encoder_.transform(X)

        self._check_model_compatibility(y)

        self._fit_keras_model(
            X,
            y,
            sample_weight=sample_weight,
            warm_start=warm_start,
            epochs=epochs,
            initial_epoch=initial_epoch,
            **kwargs,
        )

    def partial_fit(self, X, y, sample_weight=None, **kwargs) -> "BaseWrapper":
        """Fit the estimator for a single epoch, preserving the current
        training history and model parameters.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Training samples where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.

        Returns
        -------
        BaseWrapper
            A reference to the instance that can be chain called
            (ex: instance.partial_fit(X, y).transform(X) )
        """
        if "epochs" in kwargs:
            raise TypeError(
                "Invalid argument `epochs` to `partial_fit`: `partial_fit` always trains for 1 epoch"
            )
        if "initial_epoch" in kwargs:
            raise TypeError(
                "Invalid argument `initial_epoch` to `partial_fit`: `partial_fit` always trains for from the current epoch"
            )

        self._fit(
            X,
            y,
            sample_weight=sample_weight,
            warm_start=True,
            epochs=1,
            initial_epoch=self.current_epoch,
            **kwargs,
        )
        return self

    def _predict_raw(self, X, **kwargs):
        """Obtain raw predictions from Keras Model.

        For classification, this corresponds to predict_proba.
        For regression, this corresponds to predict.
        """
        # check if fitted
        if not self.initialized_:
            raise NotFittedError(
                "Estimator needs to be fit before `predict` " "can be called"
            )
        # basic input checks
        X, _ = self._validate_data(X=X, y=None)

        # pre process X
        X = self.feature_encoder_.transform(X)

        # filter kwargs and get attributes for predict
        params = self.get_params()
        pred_args = route_params(
            params, destination="predict", pass_filter=self._predict_kwargs, strict=True
        )
        pred_args = {
            k: v for k, v in pred_args.items() if not k.startswith("callbacks")
        }
        pred_args["callbacks"] = self._predict_callbacks
        pred_args.update(kwargs)
        if "batch_size" in pred_args:
            if pred_args["batch_size"] == -1:
                try:
                    pred_args["batch_size"] = X.shape[0]
                except AttributeError:
                    raise ValueError(
                        "`batch_size=-1` requires that `X` implement `shape`"
                    )

        # predict with Keras model
        y_pred = self.model_.predict(x=X, **pred_args)

        return y_pred

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Training samples where n_samples is the number of samples
            and n_features is the number of features.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.predict``.

        Warnings
        --------
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API,
            and will be removed in a future version of SciKeras.
            These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization
            (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``)
            or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``).

        Returns
        -------
        array-like
            Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).
        """
        # predict with Keras model
        y_pred = self._predict_raw(X=X, **kwargs)

        # post process y
        y_pred = self.target_encoder_.inverse_transform(y_pred)

        return y_pred

    @staticmethod
    def scorer(y_true, y_pred, **kwargs) -> float:
        """Scoring function for model.

        This is not implemented in BaseWrapper, it exists
        as a stub for documentation.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels.
        **kwargs: dict
            Extra parameters passed to the scorer.

        Returns
        -------
        float
            Score for the test data set.
        """
        raise NotImplementedError("Scoring is not implemented on BaseWrapper.")

    def score(self, X, y, sample_weight=None) -> float:
        """Returns the score on the given test data and labels.

        No default scoring function is implemented in BaseWrapper,
        you must subclass and implement one.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Test input samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        float
            Score for the test data set.
        """
        # validate sample weights
        if sample_weight is not None:
            X, sample_weight = self._validate_sample_weight(
                X=X, sample_weight=sample_weight
            )

        # validate y
        _, y = self._validate_data(X=None, y=y)

        # compute Keras model score
        y_pred = self.predict(X)

        # filter kwargs and get attributes for score
        params = self.get_params()
        score_args = route_params(params, destination="score", pass_filter=set())

        return self.scorer(y, y_pred, sample_weight=sample_weight, **score_args)

    def _get_metadata(self) -> Dict[str, Any]:
        """Meta parameters (parameters created by fit, like
        n_features_in_ or target_type_).

        Returns
        -------
        Dict[str, Any]
            Dictionary of meta parameters
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if (len(k) > 1 and k[-1] == "_" and k[-2] != "_" and k[0] != "_")
        }

    def set_params(self, **params) -> "BaseWrapper":
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        This also supports routed parameters, eg: ``classifier__optimizer__learning_rate``.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        BaseWrapper
            Estimator instance.
        """
        for param, value in params.items():
            if any(
                param.startswith(prefix + "__") for prefix in self._routing_prefixes
            ):
                # routed param
                setattr(self, param, value)
            else:
                try:
                    super().set_params(**{param: value})
                except ValueError:
                    # Give a SciKeras specific user message to aid
                    # in moving from the Keras wrappers
                    raise ValueError(
                        f"Invalid parameter {param} for estimator {self.__name__}."
                        "\nThis issue can likely be resolved by setting this parameter"
                        f" in the {self.__name__} constructor:"
                        f"\n`{self.__name__}({param}={value})`"
                        "\nCheck the list of available parameters with"
                        " `estimator.get_params().keys()`"
                    ) from None
        return self

    def _get_param_names(self):
        """Get parameter names for the estimator"""
        return (
            k for k in self.__dict__ if not k.endswith("_") and not k.startswith("_")
        )

    def _more_tags(self):
        """Get sklearn tags for the estimator"""
        tags = super()._more_tags()
        tags.update(self._tags)
        return tags

    def __repr__(self):
        repr_ = str(self.__name__)
        repr_ += "("
        params = self.get_params()
        if params:
            repr_ += "\n"
        for key, val in params.items():
            repr_ += "\t" + key + "=" + str(val) + "\n"
        repr_ += ")"
        return repr_


class KerasClassifier(BaseWrapper):
    """Implementation of the scikit-learn classifier API for Keras.

    Below are a list of SciKeras specific parameters. For details on other parameters,
    please see the see the `tf.keras.Model documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

    Parameters
    ----------
    model : Union[None, Callable[..., tf.keras.Model], tf.keras.Model], default None
        Used to build the Keras Model. When called,
        must return a compiled instance of a Keras Model
        to be used by `fit`, `predict`, etc.
        If None, you must implement ``_keras_build_fn``.
    optimizer : Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]], default "rmsprop"
        This can be a string for Keras' built in optimizers,
        an instance of tf.keras.optimizers.Optimizer
        or a class inheriting from tf.keras.optimizers.Optimizer.
        Only strings and classes support parameter routing.
    loss : Union[Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None], default None
        The loss function to use for training.
        This can be a string for Keras' built in losses,
        an instance of tf.keras.losses.Loss
        or a class inheriting from tf.keras.losses.Loss .
        Only strings and classes support parameter routing.
    random_state : Union[int, np.random.RandomState, None], default None
        Set the Tensorflow random number generators to a
        reproducible deterministic state using this seed.
        Pass an int for reproducible results across multiple
        function calls.
    warm_start : bool, default False
        If True, subsequent calls to fit will _not_ reset
        the model parameters but *will* reset the epoch to zero.
        If False, subsequent fit calls will reset the entire model.
        This has no impact on partial_fit, which always trains
        for a single epoch starting from the current epoch.
    batch_size : Union[int, None], default None
        Number of samples per gradient update.
        This will be applied to both `fit` and `predict`. To specify different numbers,
        pass `fit__batch_size=32` and `predict__batch_size=1000` (for example).
        To auto-adjust the batch size to use all samples, pass `batch_size=-1`.
    class_weight : Union[Dict[Any, float], str, None], default None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    model_ : tf.keras.Model
        The instantiated and compiled Keras Model. For pre-built models, this
        will just be a reference to the passed Model instance.
    history_ : Dict[str, List[Any]]
        Dictionary of the format ``{metric_str_name: [epoch_0_data, epoch_1_data, ..., epoch_n_data]}``.
    initialized_ : bool
        True if this estimator has been initialized (i.e. predict can be called upon it).
        Note that this does not guarantee that the model is "fitted": if ``BaseWrapper.initialize``
        was called instead of fit the model wil likely have random weights.
    target_encoder_ : sklearn-transformer
        Transformer used to pre/post process the target y.
    feature_encoder_ : sklearn-transformer
        Transformer used to pre/post process the features/input X.
    n_outputs_expected_ : int
        The number of outputs the Keras Model is expected to have, as determined by ``target_transformer_``.
    target_type_ : str
        One of:

        * 'continuous': y is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': y is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': y contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': y contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': y is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': y is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': y is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.
    y_shape_ : Tuple[int]
        Shape of the target y that the estimator was fitted on.
    y_dtype_ : np.dtype
        Dtype of the target y that the estimator was fitted on.
    X_shape_ : Tuple[int]
        Shape of the input X that the estimator was fitted on.
    X_dtype_ : np.dtype
        Dtype of the input X that the estimator was fitted on.
    n_features_in_ : int
        The number of features seen during `fit`.
    n_outputs_ : int
        Dimensions of y that the transformer was trained on.
    n_outputs_expected_ : int
        Number of outputs the Keras Model is expected to have.
    classes_ : Iterable
        The classes seen during `fit`.
    n_classes_ : int
        The number of classes seen during `fit`.
    """

    _estimator_type = "classifier"
    _tags = {
        "multilabel": True,
        "_xfail_checks": {
            "check_fit_idempotent": "tf does not use \
            sparse tensors",
            "check_no_attributes_set_in_init": "can only \
            pass if all params are hardcoded in __init__",
        },
        **BaseWrapper._tags,
    }

    def __init__(
        self,
        model: Union[None, Callable[..., tf.keras.Model], tf.keras.Model] = None,
        *,
        build_fn: Union[
            None, Callable[..., tf.keras.Model], tf.keras.Model
        ] = None,  # for backwards compatibility
        warm_start: bool = False,
        random_state: Union[int, np.random.RandomState, None] = None,
        optimizer: Union[
            str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]
        ] = "rmsprop",
        loss: Union[
            Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None
        ] = None,
        metrics: Union[
            List[
                Union[
                    str,
                    tf.keras.metrics.Metric,
                    Type[tf.keras.metrics.Metric],
                    Callable,
                ]
            ],
            None,
        ] = None,
        batch_size: Union[int, None] = None,
        validation_batch_size: Union[int, None] = None,
        verbose: int = 1,
        callbacks: Union[
            List[Union[tf.keras.callbacks.Callback, Type[tf.keras.callbacks.Callback]]],
            None,
        ] = None,
        validation_split: float = 0.0,
        shuffle: bool = True,
        run_eagerly: bool = False,
        epochs: int = 1,
        class_weight: Union[Dict[Any, float], str, None] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            build_fn=build_fn,
            warm_start=warm_start,
            random_state=random_state,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            shuffle=shuffle,
            run_eagerly=run_eagerly,
            epochs=epochs,
            **kwargs,
        )
        self.class_weight = class_weight

    def _type_of_target(self, y: np.ndarray) -> str:
        target_type = type_of_target(y)
        if target_type == "binary" and self.classes_ is not None:
            # check that this is not a multiclass problem missing categories
            target_type = type_of_target(self.classes_)
        return target_type

    @property
    def _fit_kwargs(self) -> Set[str]:
        # remove class_weight since KerasClassifier re-processes it into sample_weight
        return BaseWrapper._fit_kwargs - {"class_weight"}

    @staticmethod
    def scorer(y_true, y_pred, **kwargs) -> float:
        """Scoring function for KerasClassifier.

        KerasClassifier uses ``sklearn_accuracy_score`` by default.
        To change this, override this method.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels.
        **kwargs: dict
            Extra parameters passed to the scorer.

        Returns
        -------
        float
            Score for the test data set.
        """
        return sklearn_accuracy_score(y_true, y_pred, **kwargs)

    @property
    def target_encoder(self):
        """Retrieve a transformer for targets / y.

        For ``KerasClassifier.predict_proba`` to
        work, this transformer must accept a ``return_proba``
        argument in ``inverse_transform`` with a default value
        of False.

        Metadata will be collected from ``get_metadata`` if
        the transformer implements that method.
        Override this method to implement a custom data transformer
        for the target.

        Returns
        -------
        sklearn-transformer
            Transformer implementing the sklearn transformer
            interface.
        """
        categories = "auto" if self.classes_ is None else [self.classes_]
        return ClassifierLabelEncoder(loss=self.loss, categories=categories)

    def initialize(self, X, y) -> "KerasClassifier":
        """Initialize the model without any fitting.
        You only need to call this model if you explicitly do not want to do any fitting
        (for example with a pretrained model). You should _not_ call this
        right before calling ``fit``, calling ``fit`` will do this automatically.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
                Training samples where n_samples is the number of samples
                and `n_features` is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs), default None
            True labels for X.

        Returns
        -------
        KerasClassifier
            A reference to the KerasClassifier instance for chained calling.
        """
        self.classes_ = None
        super().initialize(X=X, y=y)
        return self

    def fit(self, X, y, sample_weight=None, **kwargs) -> "KerasClassifier":
        """Constructs a new classifier with ``model`` & fit the model to ``(X, y)``.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Training samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.

        Warnings
        --------
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``fit`` is not supported by the Scikit-Learn API,
            and will be removed in a future version of SciKeras.
            These parameters can also be specified by prefixing ``fit__`` to a parameter at initialization
            (``KerasClassifier(..., fit__batch_size=32, predict__batch_size=1000)``)
            or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``).

        Returns
        -------
        KerasClassifier
            A reference to the instance that can be chain called (``est.fit(X,y).transform(X)``).
        """
        self.classes_ = None
        if self.class_weight is not None:
            sample_weight = 1 if sample_weight is None else sample_weight
            sample_weight *= compute_sample_weight(class_weight=self.class_weight, y=y)
        super().fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
        return self

    def partial_fit(
        self, X, y, classes=None, sample_weight=None, **kwargs
    ) -> "KerasClassifier":
        """Fit classifier for a single epoch, preserving the current epoch
        and all model parameters and state.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Training samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        classes: ndarray of shape (n_classes,), default=None
            Classes across all calls to partial_fit. Can be obtained by via
            np.unique(y_all), where y_all is the target vector of the entire dataset.
            This argument is only needed for the first call to partial_fit and can be
            omitted in the subsequent calls. Note that y doesn’t need to contain
            all labels in classes. If you do not pass this argument, SciKeras
            will use ``classes=np.all(y)`` with the y passed in the first call.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.

        Returns
        -------
        KerasClassifier
            A reference to the instance that can be chain called
            (ex: instance.fit(X,y).transform(X) )
        """
        self.classes_ = (
            classes if classes is not None else getattr(self, "classes_", None)
        )
        if self.class_weight is not None:
            sample_weight = 1 if sample_weight is None else sample_weight
            sample_weight *= compute_sample_weight(class_weight=self.class_weight, y=y)
        super().partial_fit(X, y, sample_weight=sample_weight, **kwargs)
        return self

    def predict_proba(self, X, **kwargs):
        """Returns class probability estimates for the given test data.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)
            Training samples, where n_samples is the number of samples
            and n_features is the number of features.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.predict``.

        Warnings
        --------
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict_proba`` is not supported by the Scikit-Learn API,
            and will be removed in a future version of SciKeras.
            These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization
            (``KerasClassifier(..., fit__batch_size=32, predict__batch_size=1000)``)
            or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``).

        Returns
        -------
        array-like, shape (n_samples, n_outputs)
            Class probability estimates.
            In the case of binary classification,
            to match the scikit-learn API,
            SciKeras will return an array of shape (n_samples, 2)
            (instead of `(n_sample, 1)` as in Keras).
        """
        # call the Keras model's predict
        outputs = self._predict_raw(X=X, **kwargs)

        # post process y
        y = self.target_encoder_.inverse_transform(outputs, return_proba=True)

        return y


class KerasRegressor(BaseWrapper):
    """Implementation of the scikit-learn classifier API for Keras.

    Below are a list of SciKeras specific parameters. For details on other parameters,
    please see the see the `tf.keras.Model documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

    Parameters
    ----------

    model : Union[None, Callable[..., tf.keras.Model], tf.keras.Model], default None
        Used to build the Keras Model. When called,
        must return a compiled instance of a Keras Model
        to be used by `fit`, `predict`, etc.
        If None, you must implement ``_keras_build_fn``.
    optimizer : Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]], default "rmsprop"
        This can be a string for Keras' built in optimizers,
        an instance of tf.keras.optimizers.Optimizer
        or a class inheriting from tf.keras.optimizers.Optimizer.
        Only strings and classes support parameter routing.
    loss : Union[Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None], default None
        The loss function to use for training.
        This can be a string for Keras' built in losses,
        an instance of tf.keras.losses.Loss
        or a class inheriting from tf.keras.losses.Loss .
        Only strings and classes support parameter routing.
    random_state : Union[int, np.random.RandomState, None], default None
        Set the Tensorflow random number generators to a
        reproducible deterministic state using this seed.
        Pass an int for reproducible results across multiple
        function calls.
    warm_start : bool, default False
        If True, subsequent calls to fit will _not_ reset
        the model parameters but *will* reset the epoch to zero.
        If False, subsequent fit calls will reset the entire model.
        This has no impact on partial_fit, which always trains
        for a single epoch starting from the current epoch.
    batch_size : Union[int, None], default None
        Number of samples per gradient update.
        This will be applied to both `fit` and `predict`. To specify different numbers,
        pass `fit__batch_size=32` and `predict__batch_size=1000` (for example).
        To auto-adjust the batch size to use all samples, pass `batch_size=-1`.

    Attributes
    ----------

    model_ : tf.keras.Model
        The instantiated and compiled Keras Model. For pre-built models, this
        will just be a reference to the passed Model instance.

    history_ : Dict[str, List[Any]]
        Dictionary of the format ``{metric_str_name: [epoch_0_data, epoch_1_data, ..., epoch_n_data]}``.

    initialized_ : bool
        True if this estimator has been initialized (i.e. predict can be called upon it).
        Note that this does not guarantee that the model is "fitted": if ``BaseWrapper.initialize``
        was called instead of fit the model wil likely have random weights.

    target_encoder_ : sklearn-transformer
        Transformer used to pre/post process the target y.

    feature_encoder_ : sklearn-transformer
        Transformer used to pre/post process the features/input X.

    n_outputs_expected_ : int
        The number of outputs the Keras Model is expected to have, as determined by ``target_transformer_``.

    target_type_ : str
        One of:

        * 'continuous': y is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': y is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': y contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': y contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': y is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': y is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': y is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    y_shape_ : Tuple[int]
        Shape of the target y that the estimator was fitted on.

    y_dtype_ : np.dtype
        Dtype of the target y that the estimator was fitted on.

    X_shape_ : Tuple[int]
        Shape of the input X that the estimator was fitted on.

    X_dtype_ : np.dtype
        Dtype of the input X that the estimator was fitted on.

    n_features_in_ : int
        The number of features seen during `fit`.

    n_outputs_ : int
        Dimensions of y that the transformer was trained on.

    n_outputs_expected_ : int
        Number of outputs the Keras Model is expected to have.
    """

    _estimator_type = "regressor"
    _tags = {
        "multilabel": True,
        "_xfail_checks": {
            "check_fit_idempotent": "tf does not use sparse tensors",
            "check_no_attributes_set_in_init": "can only pass if all \
            params are hardcoded in __init__",
            "check_regressor_multioutput": "checks for float64 output \
            but Keras generally uses float32",
        },
        **BaseWrapper._tags,
    }

    @staticmethod
    def scorer(y_true, y_pred, **kwargs) -> float:
        """Scoring function for KerasRegressor.

        KerasRegressor uses ``sklearn_r2_score`` by default.
        To change this, override this method.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels.
        **kwargs: dict
            Extra parameters passed to the scorer.

        Returns
        -------
        float
            Score for the test data set.
        """
        return sklearn_r2_score(y_true, y_pred, **kwargs)

    def _validate_data(
        self, X=None, y=None, reset: bool = False, y_numeric: bool = False
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        # For regressors, y should ALWAYS be numeric
        # To enforce this without additional dtype checks, we set `y_numeric=True`
        # when calling `_validate_data` which will force casting to numeric for
        # non-numeric data.
        return super()._validate_data(X=X, y=y, reset=reset, y_numeric=True)

    @property
    def target_encoder(self):
        """Retrieve a transformer for targets / y.

        Metadata will be collected from ``get_metadata`` if
        the transformer implements that method.
        Override this method to implement a custom data transformer
        for the target.

        Returns
        -------
        sklearn-transformer
            Transformer implementing the sklearn transformer
            interface.
        """
        return RegressorTargetEncoder()

    @staticmethod
    @register_keras_serializable()
    def r_squared(y_true, y_pred):
        """A simple Keras implementation of R^2 that can be used as a Keras
        metric function.

        Larger values indicate a better fit, with 1.0 representing a perfect fit.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels.
        """
        # Ensure input dytpes match
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # Calculate R^2
        ss_res = tf.math.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=0)
        ss_tot = tf.math.reduce_sum(
            tf.math.squared_difference(y_true, tf.math.reduce_mean(y_true, axis=0)),
            axis=0,
        )
        return tf.math.reduce_mean(
            1 - ss_res / (ss_tot + tf.keras.backend.epsilon()), axis=-1
        )
