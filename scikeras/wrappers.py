"""Wrapper for using the Scikit-Learn API with Keras models.
"""
import inspect
import os
import warnings

from collections import defaultdict
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_X_y,
)
from tensorflow.keras.models import Model
from tensorflow.python.keras.losses import is_categorical_crossentropy
from tensorflow.python.keras.utils.generic_utils import (
    has_arg,
    register_keras_serializable,
)

from ._utils import (
    LabelDimensionTransformer,
    TFRandomState,
    _windows_upcast_ints,
    accepts_kwargs,
    get_metric_full_name,
    has_param,
    make_model_picklable,
    route_params,
)


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

    _fit_params = {
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

    _predict_params = {
        # parameters destined to keras.Model.predict
        "batch_size",
        "verbose",
        "callbacks",
        "steps",
    }

    _compile_params = {
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

    _meta_params = {
        # parameters created by wrappers within `fit`
        "_random_state",
        "n_features_in_",
        "X_dtype_",
        "y_dtype_",
        "X_shape_",
        "y_shape_",
        "model_",
        "history_",
        "is_fitted_",
        "n_outputs_",
        "keras_expected_n_ouputs_",
    }

    _routing_prefixes = {"model", "fit", "compile", "predict"}

    def __init__(
        self,
        build_fn=None,
        *,
        model=None,  # replaces build_fn
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

        # Check for deprecated APIs
        if self.model is None and build_fn is not None:
            self.model = build_fn
            warnings.warn(
                "`build_fn` will be renamed to `model` in a future release,"
                " at which point use of `build_fn` will raise an Error instead."
            )
        for kwd in kwargs.keys():
            if "__" not in kwd:
                raise ValueError(
                    "All kwargs must be routed parameters (ex: `model__hidden_layer_sizes`)"
                    f" but {kwd} is not a routed parameter!"
                )

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def _model_params(self):
        return {
            p.strip("model__")
            for p in self.get_params().keys()
            if p.startswith("model__")
        }

    def _check_model_param(self, model):
        """Checks `model`.

        Arguments:
            model : model param from __init__

        Raises:
            ValueError: if `build_fn` is not valid.
        """
        if model is None:
            # no model, use this class' _keras_build_fn
            if not hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "If not using the `build_fn` param, "
                    "you must implement `_keras_build_fn`"
                )
            final_build_fn = getattr(self, "_keras_build_fn")
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
            raise TypeError("`model` must be a callable or None")

        return final_build_fn

    def _build_keras_model(self, X, y):
        """Build the Keras model.

        This method will process all arguments and call the model building
        function with appropriate arguments.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValueError : In case sample_weight != None and the Keras model's
                `fit` method does not support that parameter.
        """
        # dynamically build model, i.e. final_build_fn builds a Keras model

        # determine what type of build_fn to use
        final_build_fn = self._check_model_param(getattr(self, "model", None))

        # collect parameters
        params = self.get_params()
        build_params = route_params(
            params, destination="model__", pass_filter=set()
        )
        if has_param(final_build_fn, "meta_params") or accepts_kwargs(
            final_build_fn
        ):
            # build_fn accepts `meta_params`, add it
            meta_params = route_params(
                self.get_meta_params(),
                destination=None,
                pass_filter=self._meta_params,
            )
            build_params["meta_params"] = meta_params
        if has_param(final_build_fn, "compile_params") or accepts_kwargs(
            final_build_fn
        ):
            # build_fn accepts `compile_params`, add it
            compile_params = route_params(
                params, destination="compile", pass_filter=self._compile_params
            )
            build_params["compile_params"] = compile_params

        # build model
        if self._random_state is not None:
            with TFRandomState(self._random_state):
                model = final_build_fn(**build_params)
        else:
            model = final_build_fn(**build_params)

        # make serializable
        make_model_picklable(model)

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
        fit_args = route_params(
            params, destination="fit", pass_filter=self._fit_params
        )
        fit_args["sample_weight"] = sample_weight

        if self._random_state is not None:
            with TFRandomState(self._random_state):
                hist = self.model_.fit(x=X, y=y, **fit_args)
        else:
            hist = self.model_.fit(x=X, y=y, **fit_args)

        if warm_start:
            if not hasattr(self, "history_"):
                self.history_ = defaultdict(list)
            self.history_ = {
                get_metric_full_name(k): self.history_[get_metric_full_name(k)]
                + hist.history[k]
                for k in hist.history.keys()
            }
        else:
            self.history_ = hist.history
        self.is_fitted_ = True

        # return self to allow fit_transform and such to work
        return self

    def _check_output_model_compatibility(self, y):
        """Checks that the model output number and y shape match, reshape as needed.

        This is mainly in place to avoid cryptic TF errors.
        """
        # check if this is a multi-output model
        if self.keras_expected_n_ouputs_ != len(self.model_.outputs):
            raise RuntimeError(
                "Detected an input of size "
                "{}, but {} has {} outputs".format(
                    (y[0].shape[0], len(y)),
                    self.model_,
                    len(self.model_.outputs),
                )
            )

        # tf v1 does not accept single item lists
        # tf v2 does
        # so go with what tf v1 accepts
        if len(y) == 1:
            y = y[0]
        else:
            y = tuple(np.squeeze(y_) for y_ in y)
        return y

    def _validate_data(self, X, y=None, reset=True):
        """Validate input data and set or check the `n_features_in_` attribute.
        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

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

        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
        else:
            if not hasattr(self, "n_features_in_"):
                raise RuntimeError(
                    "The reset parameter is False but there is no "
                    "n_features_in_ attribute. Is this estimator fitted?"
                )
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but this {self.__name__} "
                    f"is expecting {self.n_features_in_} features as input."
                )
        if y is None:
            return X
        return X, y

    @staticmethod
    def preprocess_y(y):
        """Handles manipulation of y inputs to fit or score.

        By default, this just makes sure y is 2D.

        Arguments:
            y : 1D or 2D numpy array

        Returns:
            y : numpy array of shape (n_samples, n_ouputs)
            extra_args : dictionary of output attributes, ex: n_outputs_
        """

        extra_args = {
            "y_dtype_": y.dtype,
            "y_shape_": y.shape,
        }

        return y, extra_args

    @staticmethod
    def postprocess_y(y):
        """Handles manipulation of predicted `y` values.

        By default, it joins lists of predictions for multi-ouput models
        into a single numpy array.
        Subclass and override this method to customize processing.

        Arguments:
            y : 2D numpy array or list of numpy arrays
                (the latter is for multi-ouput models)

        Returns:
            y : 2D numpy array with singular dimensions stripped
                or 1D numpy array
            extra_args : attributes of output `y`.
        """
        y = np.column_stack(y)

        extra_args = dict()
        return np.squeeze(y), extra_args

    @staticmethod
    def preprocess_X(X):
        """Handles manipulation of X before fitting.

        Subclass and override this method to process X, for example
        accommodate a multi-input model.

        Arguments:
            X : 2D numpy array

        Returns:
            X : unchanged 2D numpy array
            extra_args : attributes of output `y`.
        """
        extra_args = {
            "X_dtype_": X.dtype,
            "X_shape_": X.shape,
        }
        return X, extra_args

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
            X=X,
            y=y,
            sample_weight=sample_weight,
            warm_start=getattr(self, "warm_start", False),
        )

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
        # Handle random state
        if hasattr(self, "random_state"):
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
        else:
            self._random_state = None

        # Data checks
        if warm_start and not hasattr(self, "n_features_in_"):
            # Warm start requested but not fitted yet
            reset = True
        elif warm_start:
            # Already fitted and warm start requested
            reset = False
        else:
            # No warm start requested
            reset = True
        X, y = self._validate_data(X=X, y=y, reset=reset)

        # Save input dtype
        self.y_dtype_ = y.dtype

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
            if np.any(zeros):
                X = X[~zeros]
                y = y[~zeros]
                sample_weight = sample_weight[~zeros]
                if sample_weight.size == 0:
                    # could check any of the arrays here, arbitrary choice
                    # there will be no samples left! warn users
                    raise RuntimeError(
                        "Cannot train because there are no samples"
                        " left after deleting points with zero sample weight!"
                    )

        # pre process X, y
        X, extra_args = self.preprocess_X(X)
        # update self.X_dtype_, self.X_shape_
        for attr_name, attr_val in extra_args.items():
            setattr(self, attr_name, attr_val)
        y, extra_args = self.preprocess_y(y)
        # update self.classes_, self.n_outputs_, self.n_classes_ and
        #  self.target_type_
        for attr_name, attr_val in extra_args.items():
            setattr(self, attr_name, attr_val)

        # build model
        if (not warm_start) or (not hasattr(self, "model_")):
            self.model_ = self._build_keras_model(X, y)

        y = self._check_output_model_compatibility(y)

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
        X = self._validate_data(X=X, y=None, reset=False)

        # pre process X
        X, _ = self.preprocess_X(X)

        # filter kwargs and get attributes for predict
        params = self.get_params()
        pred_args = route_params(
            params, destination="predict", pass_filter=self._predict_params
        )

        # predict with Keras model
        y_pred = self.model_.predict(X, **pred_args)

        # post process y
        y, _ = self.postprocess_y(y_pred)
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

        Raises:
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
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
        pred_args = route_params(
            params, destination="score", pass_filter=set()
        )

        return self.scorer(y, y_pred, sample_weight=sample_weight)

    def get_meta_params(self) -> Dict[str, Any]:
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
                param.startswith(prefix + "__")
                for prefix in self._routing_prefixes
            ):
                # routed param
                setattr(self, param, value)
            else:
                passthrough[param] = value
        return super().set_params(**passthrough)

    def _get_param_names(self):
        """Get parameter names for the estimator"""
        return (
            k
            for k in self.__dict__
            if not k.endswith("_") and not k.startswith("_")
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
    _tags = BaseWrapper._tags.copy()
    _tags.update(
        {
            "multilabel": True,
            "_xfail_checks": {
                "check_classifiers_classes": "can't meet \
                performance target",
                "check_fit_idempotent": "tf does not use \
                sparse tensors",
                "check_no_attributes_set_in_init": "can only \
                pass if all params are hardcoded in __init__",
            },
        }
    )

    _meta_params = (
        BaseWrapper._meta_params.copy()
    )  # parameters created by wrappers within `fit`
    _meta_params.update(
        {
            "n_classes_",
            "target_type_",
            "classes_",
            "encoders_",
            "n_outputs_",
            "keras_expected_n_ouputs_",
        }
    )

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

    @staticmethod
    def preprocess_y(y):
        """Handles manipulation of y inputs to fit or score.

        For KerasClassifier, this handles interpreting classes from `y`.

        Arguments:
            y : 1D or 2D numpy array

        Returns:
            y : modified 2D numpy array with 0 indexed integer class labels.
            extra_args : dictionary of output attributes, ex `n_outputs_`
        """
        y, extra_args = super(KerasClassifier, KerasClassifier).preprocess_y(y)

        target_type_ = type_of_target(y)

        if len(y.shape) == 1:
            n_outputs_ = 1
        else:
            n_outputs_ = y.shape[1]

        if target_type_ == "binary":
            # y = array([1, 0, 1, 0])
            # single task, single label, binary classification
            keras_expected_n_ouputs_ = 1  # single sigmoid output expected
            # encode
            encoder = LabelEncoder()
            # No need to reshape to 1D here,
            # binary targets are always 1D already
            y = encoder.fit_transform(y)
            classes_ = encoder.classes_
            # make lists
            encoders_ = [encoder]
            classes_ = [classes_]
            y = [y]
        elif target_type_ == "multiclass":
            # y = array([1, 5, 2])
            keras_expected_n_ouputs_ = 1  # single softmax output expected
            # encode
            encoder = LabelEncoder()
            if len(y.shape) > 1 and y.shape[1] == 1:
                # Make 1D just so LabelEncoder is happy
                y = y.reshape(-1,)
            y = encoder.fit_transform(y)
            classes_ = encoder.classes_
            # make lists
            encoders_ = [encoder]
            classes_ = [classes_]
            y = [y]
        elif target_type_ == "multilabel-indicator":
            # y = array([1, 1, 1, 0], [0, 0, 1, 1])
            # split into lists for multi-output Keras
            # will be processed as multiple binary classifications
            classes_ = [np.array([0, 1])] * y.shape[1]
            y = np.split(y, y.shape[1], axis=1)
            keras_expected_n_ouputs_ = len(y)
            # encode
            encoders_ = [LabelEncoder() for _ in range(len(y))]
            y = [
                encoder.fit_transform(
                    y_.reshape(-1,) if y_.shape[1] == 1 else y_
                )
                for encoder, y_ in zip(encoders_, y)
            ]
            classes_ = [encoder.classes_ for encoder in encoders_]
        elif target_type_ == "multiclass-multioutput":
            # y = array([1, 0, 5], [2, 1, 3])
            # split into lists for multi-output Keras
            # each will be processesed as a seperate multiclass problem
            y = np.split(y, y.shape[1], axis=1)
            keras_expected_n_ouputs_ = len(y)
            # encode
            encoders_ = [LabelEncoder() for _ in range(len(y))]
            y = [
                encoder.fit_transform(
                    y_.reshape(-1,) if y_.shape[1] == 1 else y_
                )
                for encoder, y_ in zip(encoders_, y)
            ]
            classes_ = [encoder.classes_ for encoder in encoders_]
        else:
            raise ValueError("Unknown label type: {}".format(target_type_))

        # self.classes_ is kept as an array when n_outputs>1 for compatibility
        # with ensembles and other meta estimators
        # which do not support multioutput
        if len(classes_) == 1:
            n_classes_ = classes_[0].size
            classes_ = classes_[0]
            n_outputs_ = 1
        else:
            n_classes_ = [class_.shape[0] for class_ in classes_]
            n_outputs_ = len(n_classes_)

        extra_args.update(
            {
                "classes_": classes_,
                "encoders_": encoders_,
                "n_outputs_": n_outputs_,
                "keras_expected_n_ouputs_": keras_expected_n_ouputs_,
                "n_classes_": n_classes_,
                "target_type_": target_type_,
            }
        )

        return y, extra_args

    def postprocess_y(self, y):
        """Reverts _pre_process_inputs to return predicted probabilites
             in formats sklearn likes as well as retrieving the original
             classes.
        """
        if not isinstance(y, list):
            # convert single-target y to a list for easier processing
            y = [y]

        target_type_ = self.target_type_

        class_predictions = []

        for i in range(self.n_outputs_):

            if target_type_ == "binary":
                # array([0.9, 0.1], [.2, .8]) -> array(['yes', 'no'])
                if (
                    isinstance(self.encoders_[i], LabelEncoder)
                    and len(self.encoders_[i].classes_) == 1
                ):
                    # special case: single input label for sigmoid output
                    # may give more predicted classes than inputs for
                    # small sample sizes!
                    # don't even bother inverse transforming, just fill.
                    class_predictions.append(
                        np.full(
                            shape=(y[i].shape[0], 1),
                            fill_value=self.encoders_[i].classes_[0],
                        )
                    )
                else:
                    y_ = y[i].round().astype(int)
                    if y_.shape[1] == 1:
                        # Appease the demands of sklearn transformers
                        y_ = np.squeeze(y_, axis=1)
                    class_predictions.append(
                        self.encoders_[i].inverse_transform(y_)
                    )
                if (
                    len(y[i].shape) == 1
                    or y[i].shape[1] == 1
                    and len(self.encoders_[i].classes_) == 2
                ):
                    # result from a single sigmoid output
                    # reformat so that we have 2 columns
                    y[i] = np.column_stack([1 - y[i], y[i]])
            elif target_type_ in ("multiclass", "multiclass-multioutput"):
                # array([0.8, 0.1, 0.1], [.1, .8, .1]) ->
                # array(['apple', 'orange'])
                idx = np.argmax(y[i], axis=-1)
                y_ = np.zeros(y[i].shape, dtype=int)
                y_[np.arange(y[i].shape[0]), idx] = 1
                if y_.shape[1] == 1:
                    # Appease the demands of sklearn transformers
                    y_ = np.squeeze(y_, axis=1)
                class_predictions.append(
                    self.encoders_[i].inverse_transform(y_)
                )
            elif target_type_ == "multilabel-indicator":
                class_predictions.append(
                    self.encoders_[i].inverse_transform(
                        np.argmax(y[i], axis=1)
                    )
                )

        class_probabilities = np.squeeze(np.column_stack(y))

        y = np.squeeze(np.column_stack(class_predictions))

        # type cast back to input dtype
        y = y.astype(self.y_dtype_, copy=False)

        extra_args = {"class_probabilities": class_probabilities}

        return y, extra_args

    def _check_output_model_compatibility(self, y):
        """Checks that the model output number and loss functions match y.
        """
        # check loss function to adjust the encoding of the input
        # we need to do this to mimick scikit-learn behavior
        if isinstance(self.model_.loss, list):
            losses = self.model_.loss
        else:
            losses = [self.model_.loss] * self.n_outputs_
        for i, loss in enumerate(losses):
            if is_categorical_crossentropy(loss) and (
                y[i].ndim == 1 or y[i].shape[1] == 1
            ):
                encoder = OneHotEncoder(sparse=False, dtype=np.uint8)
                tf1dto2d = LabelDimensionTransformer()
                y[i] = tf1dto2d.fit_transform(y[i])
                y[i] = encoder.fit_transform(y[i])
                self.encoders_[i] = make_pipeline(
                    self.encoders_[i], tf1dto2d, encoder, "passthrough",
                )

        return super()._check_output_model_compatibility(y)

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
        self.classes_ = classes  # TODO: don't swallow this param
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
        X = self._validate_data(X=X, y=None, reset=False)

        # pre process X
        X, _ = self.preprocess_X(X)

        # collect arguments
        predict_args = route_params(
            self.get_params(),
            destination="predict",
            pass_filter=self._predict_params,
        )

        # call the Keras model's predict
        outputs = self.model_.predict(X, **predict_args)

        # join list of outputs into single output array
        _, extra_args = self.postprocess_y(outputs)

        # get class probabilities from postprocess_y's output
        class_probabilities = extra_args["class_probabilities"]

        return class_probabilities


class KerasRegressor(BaseWrapper):
    """Implementation of the scikit-learn regressor API for Keras.
    """

    _estimator_type = "regressor"
    _tags = BaseWrapper._tags.copy()
    _tags.update(
        {
            "multilabel": True,
            "_xfail_checks": {
                "check_fit_idempotent": "tf does not use sparse tensors",
                "check_methods_subset_invariance": "can't meet tol",
                "check_no_attributes_set_in_init": "can only pass if all \
                params are hardcoded in __init__",
            },
        }
    )

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
            return np.squeeze(y.astype(np.float32, copy=False)), dict()
        else:
            return np.squeeze(y.astype(np.float64, copy=False)), dict()

    def preprocess_y(self, y):
        """Split y for multi-output tasks.
        """
        y, extra_args = super().preprocess_y(y)

        if len(y.shape) == 1:
            n_outputs_ = 1
        else:
            n_outputs_ = y.shape[1]

        # for regression, multi-output is handled by single Keras output
        keras_expected_n_ouputs_ = 1

        extra_args.update(
            {
                "n_outputs_": n_outputs_,
                "keras_expected_n_ouputs_": keras_expected_n_ouputs_,
            }
        )

        y = [y]  # pack into single output list

        return y, extra_args

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
        if self.model_.loss not in ("mean_squared_error", self.r_squared,):
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
        ss_res = tf.math.reduce_sum(
            tf.math.squared_difference(y_true, y_pred), axis=0
        )
        ss_tot = tf.math.reduce_sum(
            tf.math.squared_difference(
                y_true, tf.math.reduce_mean(y_true, axis=0)
            ),
            axis=0,
        )
        return tf.math.reduce_mean(
            1 - ss_res / (ss_tot + tf.keras.backend.epsilon()), axis=-1
        )
