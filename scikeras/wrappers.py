"""Wrapper for using the Scikit-Learn API with Keras models.
"""
import inspect
import os
import warnings

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import _check_sample_weight, check_array, check_X_y
from tensorflow.keras import losses as losses_module
from tensorflow.keras import metrics as metrics_module
from tensorflow.keras import optimizers as optimizers_module
from tensorflow.keras.models import Model
from tensorflow.python.keras.losses import is_categorical_crossentropy
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable

from scikeras._utils import (
    TFRandomState,
    _class_from_strings,
    _windows_upcast_ints,
    accepts_kwargs,
    has_param,
    make_model_picklable,
    route_params,
    unflatten_params,
)
from scikeras.utils import loss_name, metric_name, type_of_target
from scikeras.utils.transformers import Ensure2DTransformer


class BaseWrapper(BaseEstimator):
    """Base class for the Keras scikit-learn wrapper.

    Warning: This class should not be used directly.
    Use descendant classes instead.

    Arguments:
        build_fn: callable function or class instance
            Used to build the Keras Model. When called,
            must return a compiled instance of a Keras Model
            to be used by `fit`, `predict`, etc.
        random_state : int, RandomState instance, default=None
            Set the Tensorflow random number generators to a
            reproducible deterministic state using this seed.
            Pass an int for reproducible results across multiple
            function calls.
        For all other parameters see tf.keras.Model documentation.
    """

    is_fitted_ = False

    _tags = {
        "poor_score": True,
        "multioutput": True,
    }

    _fit_kwargs = {
        # parameters destined to keras.Model.fit
        "callbacks",
        "batch_size",
        "epochs",
        "verbose",
        "callbacks",
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
        "callbacks",
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

    _meta = {
        # public attributes created by wrappers within `fit`
        "n_features_in_",
        "X_dtype_",
        "y_dtype_",
        "X_shape_",
        "y_ndim_",
        "model_",
        "history_",
        "is_fitted_",
        "target_type_",
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
        model=None,
        *,
        build_fn=None,  # for backwards compatibility
        warm_start=False,
        random_state=None,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        batch_size=None,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        shuffle=True,
        run_eagerly=False,
        epochs=1,
        **kwargs,
    ):

        if isinstance(build_fn, Model):
            # ensure prebuilt model can be serialized
            make_model_picklable(build_fn)

        # Parse hardcoded params
        self.model = model
        self.build_fn = build_fn
        self.warm_start = warm_start
        self.random_state = random_state
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
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
    def _model_params(self):
        return {
            k[len("model__") :]
            for k in self.get_params()
            if "model__" == k[: len("model__")]
            or k in getattr(self, "_user_params", set())
        }

    def _check_model_param(self):
        """Checks `model` and returns model building
        function to use.

        Raises:
            ValueError: if `self.model` is not valid.
        """
        model = self.model
        build_fn = self.build_fn
        if model is None and build_fn is not None:
            model = build_fn
            warnings.warn(
                "`build_fn` will be renamed to `model` in a future release,"
                " at which point use of `build_fn` will raise an Error instead."
            )
        if model is None:
            # no model, use this class' _keras_build_fn
            if not hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "If not using the `build_fn` param, "
                    "you must implement `_keras_build_fn`"
                )
            final_build_fn = self._keras_build_fn
        elif isinstance(model, Model):
            # pre-built Keras Model
            def final_build_fn():
                return model

        elif inspect.isfunction(model):
            if hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "This class cannot implement `_keras_build_fn` if"
                    " using the `model` parameter"
                )
            # a callable method/function
            final_build_fn = model
        else:
            raise TypeError(
                "`model` must be a callable, a Keras Model instance or None"
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
        compile_kwargs["optimizer"] = _class_from_strings(
            compile_kwargs["optimizer"], optimizers_module.get
        )
        compile_kwargs["optimizer"] = unflatten_params(
            items=compile_kwargs["optimizer"],
            params=route_params(
                init_params, destination="optimizer", pass_filter=set(), strict=True,
            ),
        )
        compile_kwargs["loss"] = _class_from_strings(
            compile_kwargs["loss"], losses_module.get
        )
        compile_kwargs["loss"] = unflatten_params(
            items=compile_kwargs["loss"],
            params=route_params(
                init_params, destination="loss", pass_filter=set(), strict=False,
            ),
        )
        compile_kwargs["metrics"] = _class_from_strings(
            compile_kwargs["metrics"], metrics_module.get
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

        Returns:
            model : tensorflow.keras.Model
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
        )
        compile_kwargs = None
        if has_param(final_build_fn, "meta") or accepts_kwargs(final_build_fn):
            # build_fn accepts `meta`, add it
            meta = route_params(
                self.get_meta(), destination=None, pass_filter=self._meta,
            )
            build_params["meta"] = meta
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
            with TFRandomState(self._random_state):
                model = final_build_fn(**build_params)
        else:
            model = final_build_fn(**build_params)

        # make serializable
        make_model_picklable(model)

        # compile model if user gave us an un-compiled model
        if not (hasattr(model, "loss") and hasattr(model, "optimizer")):
            if compile_kwargs is None:
                compile_kwargs = self._get_compile_kwargs()
            model.compile(**compile_kwargs)

        if not getattr(model, "loss", None) or (
            isinstance(model.loss, list)
            and not any(callable(loss) or isinstance(loss, str) for loss in model.loss)
        ):
            raise ValueError(
                "No valid loss function found."
                " You must provide a loss function to train."
                "\n\nTo resolve this issue, do one of the following:"
                "\n 1. Provide a loss function via the loss parameter."
                "\n 2. Compile your model with a loss function inside the"
                " model-building method."
                "\n\nSee https://www.tensorflow.org/api_docs/python/tf/keras/losses"
                " for more information on Keras losses."
            )

        return model

    def _fit_keras_model(self, X, y, sample_weight, warm_start):
        """Fits the Keras model.

        This method will process all arguments and call the Keras
        model's `fit` method with appropriate arguments.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,)
                Sample weights. The Keras Model must support this.
            warm_start : bool
                If ``warm_start`` is True, don't don't overwrite
                the ``history_`` attribute and append to it instead.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValueError : In case sample_weight != None and the Keras model's
                        `fit` method does not support that parameter.
        """
        if os.name == "nt":
            # see tensorflow/probability#886
            X = _windows_upcast_ints(X)
            y = _windows_upcast_ints(y)

        # collect parameters
        params = self.get_params()
        fit_args = route_params(params, destination="fit", pass_filter=self._fit_kwargs)
        fit_args["sample_weight"] = sample_weight

        if self._random_state is not None:
            with TFRandomState(self._random_state):
                hist = self.model_.fit(x=X, y=y, **fit_args)
        else:
            hist = self.model_.fit(x=X, y=y, **fit_args)

        if not warm_start or not hasattr(self, "history_"):
            self.history_ = dict()
            for key, val in hist.history.items():
                try:
                    key = metric_name(key)
                except ValueError:
                    pass
                self.history_[key] = [val]
        else:
            for key, val in hist.history.items():
                if key in self.history_:
                    self.history_[key] += [val]
                    continue
                # it's possible for the name to change from one iteration
                # to another if a shorthand name was given since
                # a pickle->un-pickle round trip may result in the name changing
                key = metric_name(key)
                self.history_[key] += [val]
        self.is_fitted_ = True

        # return self to allow fit_transform and such to work
        return self

    def _check_output_model_compatibility(self, y: np.ndarray) -> None:
        """Checks that the model output number and y shape match.

        This is in place to avoid cryptic TF errors.
        """
        # check if this is a multi-output model
        if self.model_n_outputs_ != len(self.model_.outputs):
            raise RuntimeError(
                "Detected an input of size"
                f" {y[0].shape[0]}, but {self.model_} has"
                f" {self.model_.outputs} outputs"
            )

    def _validate_data(self, X, y=None):
        """Validate input data and set or check the `n_features_in_` attribute.
        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """

        def _check_array_dtype(arr):
            if not isinstance(arr, np.ndarray):
                return _check_array_dtype(np.asarray(arr))
            elif arr.dtype.kind != "O":
                return None  # check_array won't do any casting with dtype=None
            else:
                # default to TFs backend float type
                # instead of float64 (sklearns default)
                return tf.keras.backend.floatx()

        if y is not None:
            X, y = check_X_y(
                X,
                y,
                allow_nd=True,  # allow X to have more than 2 dimensions
                multi_output=True,  # allow y to be 2D
                dtype=None,
            )
            y = check_array(
                y, ensure_2d=False, allow_nd=False, dtype=_check_array_dtype(y)
            )
        X = check_array(X, allow_nd=True, dtype=_check_array_dtype(X))

        if y is None:
            return X
        return X, y

    def _get_meta(self, X=None, y=None):
        meta = {}
        if X is not None:
            n_features = X.shape[1]
            meta.update(
                {"X_dtype_": X.dtype, "X_shape_": X.shape, "n_features_in_": n_features}
            )
        if y is not None:
            target_type = None if y is None else type_of_target(y)
            meta.update(
                {"y_dtype_": y.dtype, "y_ndim_": y.ndim, "target_type_": target_type}
            )
        return meta

    def preprocess_y(self, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handles manipulation of y inputs to fit or score.

        Parameters
        ----------
        y : np.ndarray
            1D or 2D numpy array.

        Returns
        -------
        y : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Transformed target that will be passed directly to Keras.
        meta : Dict[str, Any]
            Meta parameters, as determined from input.
        """
        if not np.can_cast(y.dtype, self.y_dtype_):
            raise ValueError(
                f"Got `y` with dtype {y.dtype},"
                f" but this {self.__name__} expected {self.X_dtype_}"
                f" and casting from {y.dtype} to {self.X_dtype_} is not safe!"
            )
        if self.y_ndim_ != y.ndim:
            raise ValueError(
                f"`y` has {y.ndim} dimensions, but this {self.__name__}"
                f" is expecting {self.y_ndim_} dimensions in `y`."
            )
        return y, self._get_meta(y=y)

    def postprocess_y(self, y: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Handles manipulation of predicted `y` values.

        By default, it joins lists of predictions for multi-output models
        into a single numpy array.
        Override this method to customize processing.

        Arguments:
            y : 2D numpy array or list of numpy arrays
                (the latter is for multi-output models)

        Returns:
            y : 2D numpy array with singular dimensions stripped
                or 1D numpy array
        """
        return np.squeeze(np.column_stack(y))

    def preprocess_X(
        self, X: np.ndarray
    ) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """Handles manipulation of X before fitting.

        Override this method to customize processing for X,
        for example accommodate a multi-input model.

        Arguments:
            X : 2D numpy array

        Returns:
            X : unchanged 2D numpy array
        """
        if not np.can_cast(X.dtype, self.X_dtype_):
            raise ValueError(
                f"Got `X` with dtype {X.dtype},"
                f" but this {self.__name__} expected {self.X_dtype_}"
                f" and casting from {X.dtype} to {self.X_dtype_} is not safe!"
            )
        return X, self._get_meta(X=X)

    def fit(self, X, y, sample_weight=None):
        """Constructs a new model with `build_fn` & fit the model to `(X, y)`.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. The Keras Model must support this.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValueError : In case of invalid shape for `y` argument.
        """
        return self._fit(
            X=X, y=y, sample_weight=sample_weight, warm_start=self.warm_start
        )

    def _initialized(self):
        return hasattr(self, "n_features_in_")

    def _initialize(self, X, y):
        X, y = self._validate_data(X=X, y=y)

        meta = self._get_meta(X, y)
        vars(self).update(**meta)

        X, X_meta = self.preprocess_X(X)
        vars(self).update(**X_meta)

        y, y_meta = self.preprocess_y(y)
        vars(self).update(**y_meta)

        self._meta.update(set(X_meta.keys()))
        self._meta.update(set(y_meta.keys()))

        self.model_ = self._build_keras_model()

        return X, y, meta

    def _fit(self, X, y, sample_weight=None, warm_start=False):
        """Constructs a new model with `build_fn` & fit the model to `(X, y)`.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. The Keras Model must support this.
            warm_start : bool, default False
                If ``warm_start`` is True, don't rebuild the model.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValueError : In case of invalid shape for `y` argument.
        """
        self._meta = self._meta.copy()  # avoid mutating class attribute

        # Handle random state
        if isinstance(self.random_state, np.random.RandomState):
            # Keras needs an integer
            # we sample an integer and use that as a seed
            # Given the same RandomState, the seed will always be
            # the same, thus giving reproducible results
            state = self.random_state.get_state()
            self._random_state = self.random_state.randint(low=1)
            self.random_state.set_state(state)
        else:
            # int or None
            self._random_state = self.random_state

        # Data checks
        should_init = True
        if (self.warm_start or warm_start) and self._initialized():
            should_init = False
        if should_init:
            X, y, meta = self._initialize(X, y)
        else:
            X, meta = self.preprocess_X(X)
            y, _ = self.preprocess_y(y)
        self._check_output_model_compatibility(y)

        if meta["n_features_in_"] != self.n_features_in_:
            raise ValueError(
                f"`X` has {meta['n_features_in_']} features, but this "
                f"{self.__name__} is expecting {self.n_features_in_} "
                f"features as input."
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=np.dtype(tf.keras.backend.floatx())
            )
            # Scikit-Learn expects a 0 in sample_weight to mean
            # "ignore the sample", but because of how Keras applies
            # sample_weight to the loss function, this doesn't
            # exactly work out (as in, sklearn estimator checks fail
            # because the predictions differ by a small margin).
            # To get around this, we manually delete these samples here
            zeros = sample_weight == 0
            if zeros.sum() == zeros.size:
                raise RuntimeError(
                    "Cannot train because there are no samples"
                    " left after deleting points with zero sample weight!"
                )
            if np.any(zeros):
                X = X[~zeros]
                y = y[~zeros]
                sample_weight = sample_weight[~zeros]

        # fit model
        return self._fit_keras_model(
            X, y, sample_weight=sample_weight, warm_start=warm_start
        )

    def partial_fit(self, X, y, sample_weight=None):
        """
        Partially fit a model.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. The Keras Model must support this.

        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.partial_fit(X, y).transform(X) )
        Raises:
            ValueError : In case of invalid shape for `y` argument.
        """
        return self._fit(X, y, sample_weight=sample_weight, warm_start=True)

    def predict(self, X):
        """Returns predictions for the given test data.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.

        Returns:
            preds: array-like, shape `(n_samples,)`
                Predictions.
        """
        # check if fitted
        if not self.is_fitted_:
            raise NotFittedError(
                "Estimator needs to be fit before `predict` " "can be called"
            )

        # basic input checks
        X = self._validate_data(X=X, y=None)

        # pre process X
        X, _ = self.preprocess_X(X)

        # filter kwargs and get attributes for predict
        params = self.get_params()
        pred_args = route_params(
            params, destination="predict", pass_filter=self._predict_kwargs
        )

        # predict with Keras model
        y_pred = self.model_.predict(X, **pred_args)

        # post process y
        y = self.postprocess_y(y_pred)
        return y

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. The Keras Model must support this.

        Returns:
            score: float
                Mean accuracy of predictions on `X` wrt. `y`.
        """
        # validate sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # validate y
        y = check_array(y, ensure_2d=False)

        # compute Keras model score
        y_pred = self.predict(X)

        # filter kwargs and get attributes for score
        params = self.get_params()
        score_args = route_params(params, destination="score", pass_filter=set())

        return self.scorer(y, y_pred, sample_weight=sample_weight, **score_args)

    def get_meta(self) -> Dict[str, Any]:
        """Get meta parameters (parameters created by fit, like
        n_features_in_ or target_type_).

        Returns
        -------
        Dict[str, Any]
            Dictionary of meta parameters
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if (
                k not in type(self)().__dict__
                and k not in self.get_params()
                and (k.startswith("_") or k.endswith("_"))
            )
        }

    def set_params(self, **params) -> "BaseWrapper":
        """Override BaseEstimator.set_params to allow setting of routed params.
        """
        passthrough = dict()
        for param, value in params.items():
            if any(
                param.startswith(prefix + "__") for prefix in self._routing_prefixes
            ):
                # routed param
                setattr(self, param, value)
            else:
                passthrough[param] = value
        return super().set_params(**passthrough)

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
    """

    _estimator_type = "classifier"
    _tags = {
        "multilabel": True,
        "_xfail_checks": {
            "check_classifiers_classes": "can't meet \
            performance target",
            "check_fit_idempotent": "tf does not use \
            sparse tensors",
            "check_no_attributes_set_in_init": "can only \
            pass if all params are hardcoded in __init__",
        },
        **BaseWrapper._tags,
    }

    @staticmethod
    def scorer(y_true, y_pred, **kwargs) -> float:
        """Accuracy score based on true and predicted target values.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        score
            float
        """
        return sklearn_accuracy_score(y_true, y_pred, **kwargs)

    def preprocess_y(
        self, y: np.ndarray
    ) -> Tuple[
        Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]], Dict[str, Any]
    ]:
        """Handles manipulation of y inputs to fit or score.

        For KerasClassifier, this handles interpreting classes from `y`.

        Parameters
        ----------
        y : np.ndarray
            1D or 2D numpy array.

        Returns
        -------
        y : Union[np.ndarray, List[np.ndarray], Dict[np.ndarray]]
            Transformed target that will be passed directly to Keras.
        meta : Dict[str, Any]
            Meta parameters, as determined from input.
        """
        y, meta = super().preprocess_y(y)

        loss = self.loss

        encoders = {
            "binary": make_pipeline(
                Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32),
            ),
            "multiclass": make_pipeline(
                Ensure2DTransformer(), OrdinalEncoder(dtype=np.float32),
            ),
            "multiclass-one-hot": FunctionTransformer(),
        }
        if is_categorical_crossentropy(loss):
            encoders["multiclass"] = make_pipeline(
                Ensure2DTransformer(), OneHotEncoder(sparse=False, dtype=np.float32),
            )

        if self.target_type_ not in encoders:
            raise ValueError(
                f"Unknown label type: {self.target_type_}."
                "\n\nTo implement support, subclass KerasClassifier and override"
                " `preprocess_y` and `postprocess_y`."
                "\n\nSee (TODO: link to docs) for more information."
            )

        if not hasattr(self, "target_encoder_"):
            self.target_encoder_ = encoders[self.target_type_].fit(y)

        y = self.target_encoder_.transform(y)
        encoder = self.target_encoder_

        if self.target_type_ in ["binary", "multiclass"]:
            meta.update(
                {
                    "classes_": encoder[1].categories_[0],
                    "n_classes_": encoder[1].categories_[0].size,
                }
            )
        elif self.target_type_ == "multiclass-one-hot":
            meta.update(
                {"classes_": np.arange(0, y.shape[1]), "n_classes_": y.shape[1],}
            )

        if self.target_type_ not in ["binary", "multiclass", "multiclass-one-hot"]:
            raise ValueError(
                f"Unknown label type: {self.target_type_}."  # To match errors used by sklearn
                "\n\nTo implement support, subclass KerasClassifier and override"
                " `preprocess_y` and `postprocess_y`."
                "\n\nSee (TODO: link to docs) for more information."
            )
        meta.update(
            {
                "target_encoder_": self.target_encoder_,
                "model_n_outputs_": 1,
                "n_outputs_": 1,
            }
        )

        return y, meta

    def postprocess_y(self, y, return_proba=False):
        """Take class probabilities and inverse transform
        back into input type, returning either probabilites
        or class predictions depending on the `return_proba`
        parameter.
        """
        if self.target_type_ == "binary":
            # array([0.9, 0.1], [.2, .8]) -> array(['yes', 'no'])
            if self.n_classes_ == 1:
                # special case: single input label for sigmoid output
                # may give more predicted classes than inputs for
                # small sample sizes!
                # don't even bother inverse transforming, just fill.
                class_predictions = np.full(
                    shape=(y.shape[0], 1), fill_value=self.classes_[0]
                )
            else:
                if y.shape == 1 or (y.shape[1] == 1 and self.n_classes_ == 2):
                    # result from a single sigmoid output
                    # reformat so that we have 2 columns
                    y = np.column_stack([1 - y, y])
                y_ = np.argmax(y, axis=1).reshape(-1, 1)
                class_predictions = self.target_encoder_.inverse_transform(y_)
        elif self.target_type_ == "multiclass":
            # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
            # array(['apple', 'orange'])
            idx = np.argmax(y, axis=-1)
            if not is_categorical_crossentropy(self.loss):
                y_ = idx.reshape(-1, 1)
            else:
                y_ = np.zeros(y.shape, dtype=int)
                y_[np.arange(y.shape[0]), idx] = 1
            class_predictions = self.target_encoder_.inverse_transform(y_)
        else:  # "multiclass-one-hot"
            # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
            # array([[1, 0, 0], [0, 1, 0]])
            idx = np.argmax(y, axis=-1)
            y_ = np.zeros(y.shape, dtype=int)
            y_[np.arange(y.shape[0]), idx] = 1
            class_predictions = y_

        if return_proba:
            return y
        else:
            return np.squeeze(np.column_stack(class_predictions)).astype(
                self.y_dtype_, copy=False
            )

    def _check_output_model_compatibility(self, y):
        """Checks that the model output number and loss functions match
        what SciKeras expects.
        """
        super()._check_output_model_compatibility(y)

        # check that if the user gave us a loss function it ended up in
        # the actual model
        if self.loss is not None:
            try:
                given = loss_name(self.loss)
                got = loss_name(self.model_.loss)
                if got is not given:
                    warnings.warn(
                        f"loss={self.loss} but model compiled with {self.model_.loss}."
                        " Data may not match loss function!"
                    )
            except ValueError:
                # unknown loss (ex: list of loss functions or custom loss)
                pass

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially fit a model.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            classes: ndarray of shape (n_classes,), default=None
                Classes across all calls to partial_fit. Can be obtained by via
                np.unique(y_all), where y_all is the target vector of the entire dataset.
                This argument is required for the first call to partial_fit and can be
                omitted in the subsequent calls. Note that y doesn’t need to contain
                all labels in classes.
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. The Keras Model must support this.

        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.partial_fit(X, y).transform(X) )
        Raises:
            ValueError : In case of invalid shape for `y` argument.
        """
        return super().partial_fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        """Returns class probability estimates for the given test data.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.

        Returns:
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                to match the scikit-learn API,
                will return an array of shape `(n_samples, 2)`
                (instead of `(n_sample, 1)` as in Keras).
        """
        # check if fitted
        if not self.is_fitted_:
            raise NotFittedError(
                "Estimator needs to be fit before `predict` " "can be called"
            )

        # basic input checks
        X = self._validate_data(X=X, y=None)

        # pre process X
        X, _ = self.preprocess_X(X)

        # collect arguments
        predict_args = route_params(
            self.get_params(), destination="predict", pass_filter=self._predict_kwargs,
        )

        # call the Keras model's predict
        outputs = self.model_.predict(X, **predict_args)

        # join list of outputs into single output array
        y = self.postprocess_y(outputs, return_proba=True)

        return y


class KerasRegressor(BaseWrapper):
    """Implementation of the scikit-learn regressor API for Keras.
    """

    _estimator_type = "regressor"
    _tags = {
        "multilabel": True,
        "_xfail_checks": {
            "check_fit_idempotent": "tf does not use sparse tensors",
            "check_methods_subset_invariance": "can't meet tol",
            "check_no_attributes_set_in_init": "can only pass if all \
            params are hardcoded in __init__",
        },
        **BaseWrapper._tags,
    }

    @staticmethod
    def scorer(y_true, y_pred, **kwargs) -> float:
        """R^2 score based on true and predicted target values.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        score
            float
        """
        return sklearn_r2_score(y_true, y_pred, **kwargs)

    def postprocess_y(self, y):
        """Ensures output is floatx and squeeze."""
        if np.can_cast(self.y_dtype_, np.float32):
            return np.squeeze(y.astype(np.float32, copy=False))
        else:
            return np.squeeze(y.astype(np.float64, copy=False))

    def preprocess_y(self, y):
        """Split y for multi-output tasks.
        """
        y, meta = super().preprocess_y(y)

        n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        # for regression, multi-output is handled by single Keras output
        model_n_outputs_ = 1

        # Make sure consisent
        if hasattr(self, "n_outputs_") and self.n_outputs_ != n_outputs_:
            raise ValueError(
                f"Detected `y` to have {n_outputs_},"
                f" but this {self.__name__} expects"
                f" {self.n_outputs_} for `y`."
            )
            # No need to check model_n_outputs_ since that's hardcoded
        meta.update(
            {"n_outputs_": n_outputs_, "model_n_outputs_": model_n_outputs_,}
        )
        return y, meta

    def score(self, X, y, sample_weight=None):
        """Returns the mean loss on the given test data and labels.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)`
                True labels for `X`.

        Returns:
            score: float
                Mean accuracy of predictions on `X` wrt. `y`.
        """
        # check loss function and warn if it is not the same as score function
        if self.model_.loss is not self.r_squared:
            warnings.warn(
                "Since ScikitLearn's `score` uses R^2 by default, it is "
                "advisable to use the same loss/metric when optimizing the "
                "model.This class provides an R^2 implementation in "
                "`KerasRegressor.r_squared`."
            )

        return super().score(X, y, sample_weight=sample_weight)

    @staticmethod
    @register_keras_serializable()
    def r_squared(y_true, y_pred):
        """A simple Keras implementation of R^2 that can be used as a Keras
        loss function.

        Since ScikitLearn's `score` uses R^2 by default, it is
        advisable to use the same loss/metric when optimizing the model.
        """
        # Ensure input dytpes match
        # y_pred will always be float32 so we cast y_true to float32
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
