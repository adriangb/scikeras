"""Wrapper for using the Scikit-Learn API with Keras models.
"""
import inspect
import os
import warnings

from collections import defaultdict

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y
from tensorflow.keras.models import Model
from tensorflow.python.keras.losses import is_categorical_crossentropy
from tensorflow.python.keras.utils.generic_utils import has_arg
from tensorflow.python.keras.utils.generic_utils import (
    register_keras_serializable,
)

from ._utils import LabelDimensionTransformer
from ._utils import TFRandomState
from ._utils import _get_default_args
from ._utils import get_metric_full_name
from ._utils import make_model_picklable


OS_IS_WINDOWS = os.name == "nt"  # see tensorflow/probability#886


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

    def __init__(
        self,
        build_fn=None,
        *,
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
        **kwargs,
    ):
        # Get defaults from `build_fn`
        if inspect.isfunction(build_fn):
            vars(self).update(_get_default_args(build_fn))

        if isinstance(build_fn, Model):
            # ensure prebuilt model can be serialized
            make_model_picklable(build_fn)

        # Parse hardcoded params
        self.build_fn = build_fn
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

        # Unpack kwargs
        vars(self).update(**kwargs)

    @property
    def __name__(self):
        return self.__class__.__name__

    def _check_build_fn(self, build_fn):
        """Checks `build_fn`.

        Arguments:
            build_fn : method or callable class as defined in __init__

        Raises:
            ValueError: if `build_fn` is not valid.
        """
        if build_fn is None:
            # no build_fn, use this class' __call__method
            if not hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "If not using the `build_fn` param, "
                    "you must implement `_keras_build_fn`"
                )
            final_build_fn = getattr(self, "_keras_build_fn")
        elif isinstance(build_fn, Model):
            # pre-built Keras Model
            def final_build_fn():
                return build_fn

        elif inspect.isfunction(build_fn):
            if hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "This class cannot implement `_keras_build_fn` if"
                    " using the `build_fn` parameter"
                )
            # a callable method/function
            final_build_fn = build_fn
        elif (
            callable(build_fn)
            and hasattr(build_fn, "__class__")
            and hasattr(build_fn.__class__, "__call__")
        ):
            if hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "This class cannot implement `_keras_build_fn` if"
                    " using the `build_fn` parameter"
                )
            # an instance of a class implementing __call__
            final_build_fn = build_fn.__call__
        else:
            raise TypeError("`build_fn` must be a callable or None")

        return final_build_fn

    def _fit_build_keras_model(self, X, y, **kwargs):
        """Build the Keras model.

        This method will process all arguments and call the model building
        function with appropriate arguments.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments `build_fn`.
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
        final_build_fn = self._check_build_fn(getattr(self, "build_fn", None))

        # get model arguments
        model_args = self._filter_params(final_build_fn)

        # check if the model building function requires X and/or y to be passed
        X_y_args = self._filter_params(
            final_build_fn, params_to_check={"X": X, "y": y}
        )

        # filter kwargs
        kwargs = self._filter_params(final_build_fn, params_to_check=kwargs)

        # combine all arguments
        build_args = {
            **model_args,  # noqa: E999
            **X_y_args,  # noqa: E999
            **kwargs,  # noqa: E999
        }

        # build model
        if self._random_state is not None:
            with TFRandomState(self._random_state):
                model = final_build_fn(**build_args)
        else:
            model = final_build_fn(**build_args)

        # make serializable
        make_model_picklable(model)

        return model

    def _fit_keras_model(self, X, y, sample_weight, warm_start, **kwargs):
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
            **kwargs: dictionary arguments
                Legal arguments are the arguments of the keras model's
                `fit` method.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValueError : In case sample_weight != None and the Keras model's
                        `fit` method does not support that parameter.
        """
        # add `sample_weight` param, required to be explicit by some sklearn
        # functions that use inspect.signature on the `score` method
        if sample_weight is not None:
            # avoid pesky Keras warnings if sample_weight is not used
            kwargs.update({"sample_weight": sample_weight})

        # filter kwargs down to those accepted by self.model_.fit
        kwargs = self._filter_params(self.model_.fit, params_to_check=kwargs)

        # get model.fit's arguments (allows arbitrary model use)
        fit_args = self._filter_params(self.model_.fit)

        # fit model and save history
        # order implies kwargs overwrites fit_args
        fit_args = {**fit_args, **kwargs}

        if OS_IS_WINDOWS:
            # see tensorflow/probability#886
            if not isinstance(X, np.ndarray):  # list, tuple, etc.
                X = [
                    X_.astype(np.int64) if X_.dtype == np.int32 else X_
                    for X_ in X
                ]
            else:
                X = X.astype(np.int64) if X.dtype == np.int32 else X
            if not isinstance(y, np.ndarray):  # list, tuple, etc.
                y = [
                    y_.astype(np.int64) if y_.dtype == np.int32 else y_
                    for y_ in y
                ]
            else:
                y = y.astype(np.int64) if y.dtype == np.int32 else y

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
        if y is not None:
            X, y = check_X_y(
                X,
                y,
                allow_nd=True,  # allow X to have more than 2 dimensions
                multi_output=True,  # allow y to be 2D
            )
        X = check_array(X, allow_nd=True)

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

        extra_args = dict()

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
        extra_args = dict()
        return X, extra_args

    def fit(self, X, y, sample_weight=None, warm_start=False, **kwargs):
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
            **kwargs: dictionary arguments
                Legal arguments are the arguments of the keras model's `fit`
                method.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValueError : In case of invalid shape for `y` argument.
            ValueError : In case sample_weight != None and the Keras model's
                `fit` method does not support that parameter.
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

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            # Scikit-Learn expects a 0 in sample_weight to mean
            # "ignore the sample", but because of how Keras applies
            # sample_weight to the loss function, this doesn't
            # exacly work out (as in, sklearn estimator checks fail
            # because the predictions differ by a small margin).
            # To get around this, we manually delete these samples here
            zeros = sample_weight == 0
            if np.any(zeros):
                X = X[~zeros]
                y = y[~zeros]
                sample_weight = sample_weight[~zeros]

        # pre process X, y
        X, _ = self.preprocess_X(X)
        y, extra_args = self.preprocess_y(y)
        # update self.classes_, self.n_outputs_, self.n_classes_ and
        #  self.cls_type_
        for attr_name, attr_val in extra_args.items():
            setattr(self, attr_name, attr_val)

        # build model
        if (not warm_start) or (not hasattr(self, "model_")):
            self.model_ = self._fit_build_keras_model(X, y, **kwargs)

        y = self._check_output_model_compatibility(y)

        # fit model
        return self._fit_keras_model(
            X, y, sample_weight=sample_weight, warm_start=warm_start, **kwargs
        )

    def partial_fit(self, X, y, sample_weight=None, **kwargs):
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
            **kwargs: dictionary arguments
                Legal arguments are the arguments of the keras model's `fit`
                method.

        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.partial_fit(X, y).transform(X) )
        Raises:
            ValueError : In case of invalid shape for `y` argument.
            ValuError : In case sample_weight != None and the Keras model's
                `fit` method does not support that parameter.
        """
        return self.fit(
            X, y, sample_weight=sample_weight, warm_start=True, **kwargs
        )

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `self.model_.predict`.

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
        kwargs = self._filter_params(
            self.model_.predict, params_to_check=kwargs
        )
        predict_args = self._filter_params(self.model_.predict)

        # predict with Keras model
        pred_args = {**predict_args, **kwargs}
        y_pred = self.model_.predict(X, **pred_args)

        # post process y
        y, _ = self.postprocess_y(y_pred)
        return y

    def score(self, X, y, sample_weight=None, **kwargs):
        """Returns the mean accuracy on the given test data and labels.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. The Keras Model must support this.
            **kwargs: dictionary arguments
                Legal arguments are those of `self.model_.evaluate`.

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

        # compute Keras model score
        y_pred = self.predict(X, **kwargs)

        return self._scorer(y, y_pred, sample_weight=sample_weight)

    def _filter_params(self, fn, params_to_check=None):
        """Filters all instance attributes (parameters) and
             returns those in `fn`'s arguments.

        Arguments:
            fn : arbitrary function
            params_to_check : dictionary, parameters to check.
                Defaults to checking all attributes of this estimator.

        Returns:
            res : dictionary containing variables
                in both self and `fn`'s arguments.
        """
        res = {}
        for name, value in (params_to_check or self.__dict__).items():
            if has_arg(fn, name):
                res.update({name: value})
        return res

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
    _scorer = staticmethod(sklearn_accuracy_score)
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
        y, _ = super(KerasClassifier, KerasClassifier).preprocess_y(y)

        cls_type_ = type_of_target(y)

        input_dtype_ = y.dtype

        if len(y.shape) == 1:
            n_outputs_ = 1
        else:
            n_outputs_ = y.shape[1]

        if cls_type_ == "binary":
            # y = array([1, 0, 1, 0])
            # single task, single label, binary classification
            keras_expected_n_ouputs_ = 1  # single sigmoid output expected
            # encode
            encoder = LabelEncoder()
            # No need to reshape to 1D here,
            # binary targets are always 1D already
            y = encoder.fit_transform(y).astype(y.dtype)
            classes_ = encoder.classes_
            # make lists
            encoders_ = [encoder]
            classes_ = [classes_]
            y = [y]
        elif cls_type_ == "multiclass":
            # y = array([1, 5, 2])
            keras_expected_n_ouputs_ = 1  # single softmax output expected
            # encode
            encoder = LabelEncoder()
            if len(y.shape) > 1 and y.shape[1] == 1:
                # Make 1D just so LabelEncoder is happy
                y = y.reshape(-1,)
            y = encoder.fit_transform(y).astype(y.dtype)
            classes_ = encoder.classes_
            # make lists
            encoders_ = [encoder]
            classes_ = [classes_]
            y = [y]
        elif cls_type_ == "multilabel-indicator":
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
                ).astype(y_.dtype)
                for encoder, y_ in zip(encoders_, y)
            ]
            classes_ = [encoder.classes_ for encoder in encoders_]
        elif cls_type_ == "multiclass-multioutput":
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
                ).astype(y_.dtype)
                for encoder, y_ in zip(encoders_, y)
            ]
            classes_ = [encoder.classes_ for encoder in encoders_]
        else:
            raise ValueError("Unknown label type: {}".format(cls_type_))

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

        extra_args = {
            "classes_": classes_,
            "encoders_": encoders_,
            "n_outputs_": n_outputs_,
            "keras_expected_n_ouputs_": keras_expected_n_ouputs_,
            "n_classes_": n_classes_,
            "cls_type_": cls_type_,
            "input_dtype_": input_dtype_,
        }

        return y, extra_args

    def postprocess_y(self, y):
        """Reverts _pre_process_inputs to return predicted probabilites
             in formats sklearn likes as well as retrieving the original
             classes.
        """
        if not isinstance(y, list):
            # convert single-target y to a list for easier processing
            y = [y]

        cls_type_ = self.cls_type_

        class_predictions = []

        for i in range(self.n_outputs_):

            if cls_type_ == "binary":
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
            elif cls_type_ in ("multiclass", "multiclass-multioutput"):
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
            elif cls_type_ == "multilabel-indicator":
                class_predictions.append(
                    self.encoders_[i].inverse_transform(
                        np.argmax(y[i], axis=1)
                    )
                )

        class_probabilities = np.squeeze(np.column_stack(y))

        y = np.squeeze(np.column_stack(class_predictions))

        # type cast back to input dtype
        y = y.astype(self.input_dtype_)

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
                encoder = OneHotEncoder(sparse=False, dtype=y[i].dtype)
                tf1dto2d = LabelDimensionTransformer()
                y[i] = tf1dto2d.fit_transform(y[i])
                y[i] = encoder.fit_transform(y[i])
                self.encoders_[i] = make_pipeline(
                    self.encoders_[i], tf1dto2d, encoder, "passthrough",
                )

        return super()._check_output_model_compatibility(y)

    def predict_proba(self, X, **kwargs):
        """Returns class probability estimates for the given test data.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

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

        # filter kwargs and get attributes that are inputs to model.predict
        kwargs = self._filter_params(
            self.model_.predict, params_to_check=kwargs
        )
        predict_args = self._filter_params(self.model_.predict)

        # call the Keras model
        predict_args = {**predict_args, **kwargs}
        outputs = self.model_.predict(X, **predict_args)

        # join list of outputs into single output array
        _, extra_args = self.postprocess_y(outputs)

        class_probabilities = extra_args["class_probabilities"]

        return class_probabilities


class KerasRegressor(BaseWrapper):
    """Implementation of the scikit-learn regressor API for Keras.
    """

    _estimator_type = "regressor"
    _scorer = staticmethod(sklearn_r2_score)
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

    def _validate_data(self, X, y=None, reset=True):
        """Convert y to float, regressors cannot accept int."""
        if y is not None:
            y = check_array(y, ensure_2d=False)
        return super()._validate_data(X=X, y=y, reset=reset)

    def postprocess_y(self, y):
        """Ensures output is float64 and squeeze."""
        return np.squeeze(y.astype("float64")), dict()

    def preprocess_y(self, y):
        """Split y for multi-output tasks.
        """
        y, _ = super().preprocess_y(y)

        if len(y.shape) == 1:
            n_outputs_ = 1
        else:
            n_outputs_ = y.shape[1]

        # for regression, multi-output is handled by single Keras output
        keras_expected_n_ouputs_ = 1

        extra_args = {
            "n_outputs_": n_outputs_,
            "keras_expected_n_ouputs_": keras_expected_n_ouputs_,
        }

        y = [y]  # pack into single output list

        return y, extra_args

    def score(self, X, y, sample_weight=None, **kwargs):
        """Returns the mean loss on the given test data and labels.

        Arguments:
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)`
                True labels for `X`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        Returns:
            score: float
                Mean accuracy of predictions on `X` wrt. `y`.
        """
        res = super(KerasRegressor, self).score(X, y, sample_weight, **kwargs)

        # check loss function and warn if it is not the same as score function
        if self.model_.loss not in ("mean_squared_error", self.r_squared,):
            warnings.warn(
                "Since ScikitLearn's `score` uses R^2 by default, it is "
                "advisable to use the same loss/metric when optimizing the "
                "model.This class provides an R^2 implementation in "
                "`KerasRegressor.r_squared`."
            )

        return res

    @staticmethod
    @register_keras_serializable()
    def r_squared(y_true, y_pred):
        """A simple Keras implementation of R^2 that can be used as a Keras
        loss function.

        Since ScikitLearn's `score` uses R^2 by default, it is
        advisable to use the same loss/metric when optimizing the model.
        """
        # Ensure input dytpes match
        dtype_y_true = np.dtype(y_true.dtype.as_numpy_dtype())
        dtype_y_pred = np.dtype(y_pred.dtype.as_numpy_dtype())
        dest_dtype = np.promote_types(dtype_y_pred, dtype_y_true)
        y_true = tf.cast(y_true, dtype=dest_dtype)
        y_pred = tf.cast(y_pred, dtype=dest_dtype)
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
