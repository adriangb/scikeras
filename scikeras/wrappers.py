"""Wrapper for using the Scikit-Learn API with Keras models.
"""
import inspect
import warnings
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    _check_sample_weight,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.losses import is_categorical_crossentropy
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.utils.generic_utils import (
    has_arg,
    register_keras_serializable,
)


# known keras function names that will be added to _legal_params_fns if they
# exist in the generated model
KNOWN_KERAS_FN_NAMES = (
    "fit",
    "evaluate",
    "predict",
)

# used by inspect to resolve parameters of parent classes
ARGS_KWARGS_IDENTIFIERS = (
    inspect.Parameter.VAR_KEYWORD,
    inspect.Parameter.VAR_POSITIONAL,
)


class LabelDimensionTransformer(TransformerMixin, BaseEstimator):
    """Transforms from 1D -> 2D and back.

    Used when applying LabelTransformer -> OneHotEncoder.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X):
        if X.shape[1] == 1:
            X = np.squeeze(X, axis=1)
        return X


def unpack_keras_model(model, training_config, weights):
    """Creates a new Keras model object using the input
    parameters.

    Returns
    -------
    Model
        A copy of the input Keras Model,
        compiled if the original was compiled.
    """
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(training_config)
        )
    restored_model.set_weights(weights)
    restored_model.__reduce_ex__ = pack_keras_model.__get__(restored_model)
    return restored_model


def pack_keras_model(model_obj, protocol):
    """Pickle a Keras Model.

    Arguments:
        model_obj: an instance of a Keras Model.
        protocol: pickle protocol version, ignored.

    Returns
    -------
    Pickled model
        A tuple following the pickle protocol.
    """
    if not isinstance(model_obj, Model):
        raise TypeError("`model_obj` must be an instance of a Keras Model")
    # pack up model
    model_metadata = saving_utils.model_metadata(model_obj)
    training_config = model_metadata.get("training_config", None)
    model = serialize(model_obj)
    weights = model_obj.get_weights()
    return (unpack_keras_model, (model, training_config, weights))


def make_model_picklable(model_obj):
    """Makes a Keras Model object picklable without cloning.

    Arguments:
        model_obj: an instance of a Keras Model.

    Returns
    -------
    Model
        The input model, but directly picklable.
    """
    if not isinstance(model_obj, Model):
        raise TypeError("`model_obj` must be an instance of a Keras Model")
    model_obj.__reduce_ex__ = pack_keras_model.__get__(model_obj)


class BaseWrapper(BaseEstimator):
    """Base class for the Keras scikit-learn wrapper.

    Warning: This class should not be used directly.
    Use descendant classes instead.

    Arguments:
        build_fn: callable function or class instance
        **sk_params: model parameters & fitting parameters

    The `build_fn` should construct, compile and return a Keras model, which
    will then be used to fit/predict. One of the following
    three values could be passed to `build_fn`:
    1. A function
    2. An instance of a class that implements the `__call__` method
    3. An instance of a Keras Model. A copy of this instance will be made
    4. None. This means you implement a class that inherits from `BaseWrapper`,
    `KerasClassifier` or `KerasRegressor`. The `__call__` method of the
    present class will then be treated as the default `build_fn`.
    If `build_fn` has parameters X or y, these will be passed automatically.

    `sk_params` takes both model parameters and fitting parameters. Legal model
    parameters are the arguments of `build_fn`. Note that like all other
    estimators in scikit-learn, `build_fn` or your child class should provide
    default values for its arguments, so that you could create the estimator
    without passing any values to `sk_params`.

    `sk_params` could also accept parameters for calling `fit`, `predict`,
    `predict_proba`, and `score` methods (e.g., `epochs`, `batch_size`).
    fitting (predicting) parameters are selected in the following order:

    1. Values passed to the dictionary arguments of
    `fit`, `predict`, `predict_proba`, and `score` methods
    2. Values passed to `sk_params`
    3. The default values of the `keras.models.Sequential`
    `fit`, `predict`, `predict_proba` and `score` methods

    When using scikit-learn's `grid_search` API, legal tunable parameters are
    those you could pass to `sk_params`, including fitting parameters.
    In other words, you could use `grid_search` to search for the best
    `batch_size` or `epochs` as well as the model parameters.
    """

    # basic legal parameter set, based on functions that will normally be
    # called the model building function will be dynamically added
    _legal_params_fns = [
        Sequential.evaluate,
        Sequential.fit,
        Sequential.predict,
        Model.evaluate,
        Model.fit,
        Model.predict,
    ]

    _sk_params = None
    is_fitted_ = False

    _tags = {
        "non_deterministic": True,  # can't easily set random_state
        "poor_score": True,
        "multioutput": True,
    }

    def __init__(self, build_fn=None, **sk_params):

        if isinstance(build_fn, Model):
            # ensure prebuilt model can be serialized
            make_model_picklable(build_fn)

        self.build_fn = build_fn

        if sk_params:

            # for backwards compatibility

            # the sklearn API requires that all __init__ parameters be saved
            # as an instance attribute of the same name
            for name, val in sk_params.items():
                setattr(self, name, val)

            # save keys so that we can count these as __init__ params
            self._sk_params = list(sk_params.keys())

        # check that all __init__ parameters were assigned (as per sklearn API)
        try:
            params = self.get_params(deep=False)
            for key in params.keys():
                try:
                    getattr(self, key)
                except AttributeError:
                    raise RuntimeError(
                        "Unasigned input parameter: {}".format(key)
                    )
        except AttributeError as e:
            raise RuntimeError("Unasigned input parameter: {}".format(e))

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
            if not hasattr(self, "__call__"):
                raise ValueError(
                    "If not using the `build_fn` param, "
                    "you must implement `__call__`"
                )
            final_build_fn = self.__call__
        elif isinstance(build_fn, Model):
            # pre-built Keras Model
            def final_build_fn():
                return build_fn

        elif inspect.isfunction(build_fn):
            if hasattr(self, "__call__"):
                raise ValueError(
                    "This class cannot implement `__call__` if"
                    " using the `build_fn` parameter"
                )
            # a callable method/function
            final_build_fn = build_fn
        elif (
            callable(build_fn)
            and hasattr(build_fn, "__class__")
            and hasattr(build_fn.__class__, "__call__")
        ):
            if hasattr(self, "__call__"):
                raise ValueError(
                    "This class cannot implement `__call__` if"
                    " using the `build_fn` parameter"
                )
            # an instance of a class implementing __call__
            final_build_fn = build_fn.__call__
        else:
            raise TypeError("`build_fn` must be a callable or None")
        # append legal parameters
        self._legal_params_fns.append(final_build_fn)

        return final_build_fn

    def _build_keras_model(self, X, y, sample_weight, **kwargs):
        """Build the Keras model.

        This method will process all arguments and call the model building
        function with appropriate arguments.

        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `X`.
            sample_weight : array-like of shape (n_samples,)
                Sample weights. The Keras Model must support this.
            **kwargs: dictionary arguments
                Legal arguments are the arguments `build_fn`.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValuError : In case sample_weight != None and the Keras model's
                `fit` method does not support that parameter.
        """
        # dynamically build model, i.e. final_build_fn builds a Keras model

        # determine what type of build_fn to use
        final_build_fn = self._check_build_fn(self.build_fn)

        # get model arguments
        model_args = self._filter_params(final_build_fn)

        # add `sample_weight` param
        # while it is not usually needed to build the model, some Keras models
        # require knowledge of the type of sample_weight to be built.
        sample_weight_arg = self._filter_params(
            final_build_fn, params_to_check={"sample_weight": sample_weight}
        )

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
            **sample_weight_arg,  # noqa: E999
            **kwargs,  # noqa: E999
        }

        # build model
        model = final_build_fn(**build_args)

        # append legal parameter names from model
        for known_keras_fn in KNOWN_KERAS_FN_NAMES:
            if hasattr(model, known_keras_fn):
                self._legal_params_fns.append(getattr(model, known_keras_fn))

        # make serializable
        make_model_picklable(model)

        return model

    def _fit_keras_model(self, X, y, sample_weight, **kwargs):
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
            **kwargs: dictionary arguments
                Legal arguments are the arguments of the keras model's
                `fit` method.
        Returns:
            self : object
                a reference to the instance that can be chain called
                (ex: instance.fit(X,y).transform(X) )
        Raises:
            ValuError : In case sample_weight != None and the Keras model's
                        `fit` method does not support that parameter.
        """
        # add `sample_weight` param, required to be explicit by some sklearn
        # functions that use inspect.signature on the `score` method
        if sample_weight is not None:
            # avoid pesky Keras warnings if sample_weight is not used
            kwargs.update({"sample_weight": sample_weight})

        # filter kwargs down to those accepted by self.model_.fit
        kwargs = self._filter_params(self.model_.fit, params_to_check=kwargs)

        if sample_weight is not None and "sample_weight" not in kwargs:
            raise ValueError(
                "Parameter `sample_weight` is unsupported by Keras model "
                + str(self.model_)
            )

        # get model.fit's arguments (allows arbitrary model use)
        fit_args = self._filter_params(self.model_.fit)

        # fit model and save history
        # order implies kwargs overwrites fit_args
        fit_args = {**fit_args, **kwargs}

        hist = self.model_.fit(x=X, y=y, **fit_args)

        if not hasattr(self, "history_"):
            self.history_ = defaultdict(list)
        keys = set(hist.history).union(self.history_.keys())
        self.history_ = {k: self.history_[k] + hist.history[k] for k in keys}
        self.is_fitted_ = True

        # return self to allow fit_transform and such to work
        return self

    def _check_output_model_compatibility(self, y):
        """Checks that the model output number and y shape match, reshape as needed.
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
        if y is not None:
            X, y = check_X_y(
                X,
                y,
                allow_nd=True,  # allow X to have more than 2 dimensions
                multi_output=True,  # allow y to be 2D
            )
        X = check_array(X, allow_nd=True, dtype=["float64", "int"])

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
    def _pre_process_y(y):
        """Handles manipulation of y inputs to fit or score.

        By default, this just makes sure y is 2D.

        Arguments:
            y : 1D or 2D numpy array

        Returns:
            y : numpy array of shape (n_samples, n_ouputs)
            extra_args : dictionary of output attributes, ex: n_outputs_
                    These parameters are added to `self` by `fit` and
                    consumed (but not reset) by `score`.
        """

        extra_args = dict()

        return y, extra_args

    @staticmethod
    def _post_process_y(y):
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
            extra_args : attributes of output `y` such as probabilites.
                Currently unused by KerasRegressor but kept for flexibility.
        """
        y = np.column_stack(y)

        extra_args = dict()
        return np.squeeze(y), extra_args

    @staticmethod
    def _pre_process_X(X):
        """Handles manipulation of X before fitting.

        Subclass and override this method to process X, for example
        accommodate a multi-input model.

        Arguments:
            X : 2D numpy array

        Returns:
            X : unchanged 2D numpy array
            extra_args : attributes of output `y` such as probabilites.
                    Currently unused by KerasRegressor but kept for
                    flexibility.
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
            ValuError : In case sample_weight != None and the Keras model's
                `fit` method does not support that parameter.
        """
        X, y = self._validate_data(X=X, y=y, reset=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=["float64", "int"]
            )

        # pre process X, y
        X, _ = self._pre_process_X(X)
        y, extra_args = self._pre_process_y(y)
        # update self.classes_, self.n_outputs_, self.n_classes_ and
        #  self.cls_type_
        for attr_name, attr_val in extra_args.items():
            setattr(self, attr_name, attr_val)

        # build model
        if (not warm_start) or (not hasattr(self, "model_")):
            self.model_ = self._build_keras_model(
                X, y, sample_weight=sample_weight, **kwargs
            )

        y = self._check_output_model_compatibility(y)

        # fit model
        return self._fit_keras_model(
            X, y, sample_weight=sample_weight, **kwargs
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
        X, _ = self._pre_process_X(X)

        # filter kwargs and get attributes for predict
        kwargs = self._filter_params(
            self.model_.predict, params_to_check=kwargs
        )
        predict_args = self._filter_params(self.model_.predict)

        # predict with Keras model
        pred_args = {**predict_args, **kwargs}
        y_pred = self.model_.predict(X, **pred_args)

        # post process y
        y, _ = self._post_process_y(y_pred)
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
                Legal arguments are those of self.model_.evaluate.

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
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=["float64", "int"]
            )

        # pre process X, y
        _, extra_args = self._pre_process_y(y)

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
        parameters = super()._get_param_names()
        # add kwargs/sk_params if the user gave those as input
        if self._sk_params:
            return sorted(parameters + self._sk_params)
        return parameters

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
                "check_sample_weights_invariance": "can't easily \
                set Keras random seed",
                "check_classifiers_multilabel_representation_invariance": "can't \
                easily set Keras random seed",
                "check_estimators_data_not_an_array": "can't easily \
                set Keras random seed",
                "check_classifier_data_not_an_array": "can't set \
                Keras random seed",
                "check_classifiers_classes": "can't meet \
                performance target",
                "check_supervised_y_2d": "can't easily set \
                Keras random seed",
                "check_fit_idempotent": "tf does not use \
                sparse tensors",
                "check_methods_subset_invariance": "can't easily set \
                Keras random seed",
            },
        }
    )

    @staticmethod
    def _pre_process_y(y):
        """Handles manipulation of y inputs to fit or score.

             For KerasClassifier, this handles interpreting classes from `y`.

        Arguments:
            y : 1D or 2D numpy array

        Returns:
            y : modified 2D numpy array with 0 indexed integer class labels.
            classes_ : list of original class labels.
            n_classes_ : number of classes.
            one_hot_encoded : True if input y was one-hot-encoded.
        """
        y, _ = super(KerasClassifier, KerasClassifier)._pre_process_y(y)

        cls_type_ = type_of_target(y)

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
            if len(y.shape) > 1 and y.shape[1] == 1:
                # Make 1D just so LabelEncoder is happy
                y = y.reshape(-1,)
            y = encoder.fit_transform(y)
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
            y = encoder.fit_transform(y)
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
                )
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
                )
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
        }

        return y, extra_args

    def _post_process_y(self, y):
        """Reverts _pre_process_inputs to return predicted probabilites
             in formats sklearn likes as well as retrieving the original
             classes.
        """
        if not isinstance(y, list):
            # convert single-target y to a list for easier processing
            y = [y]

        cls_type_ = self.cls_type_

        class_predictions = []

        def to_target(y_i):
            idx = np.argmax(y_i, axis=-1)
            y_ = np.zeros(y_i.shape)
            y_[np.arange(y_i.shape[0]), idx] = 1
            return y_

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
                encoder = OneHotEncoder(sparse=False)
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
        X, _ = self._pre_process_X(X)

        # filter kwargs and get attributes that are inputs to model.predict
        kwargs = self._filter_params(
            self.model_.predict, params_to_check=kwargs
        )
        predict_args = self._filter_params(self.model_.predict)

        # call the Keras model
        predict_args = {**predict_args, **kwargs}
        outputs = self.model_.predict(X, **predict_args)

        # join list of outputs into single output array
        _, extra_args = self._post_process_y(outputs)

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
                "check_sample_weights_invariance": "can't easily set \
                Keras random seed",
                "check_estimators_data_not_an_array": "can't easily set \
                Keras random seed",
                "check_regressor_data_not_an_array": "can't set Keras \
                    random seed",
                "check_supervised_y_2d": "can't easily set Keras random seed",
                "check_fit_idempotent": "tf does not use sparse tensors",
                "check_regressors_int": "can't easily set Keras \
                random seed",
                "poor_score": True,
                "check_methods_subset_invariance": "can't easily set \
                Keras random seed",
            },
        }
    )

    def _validate_data(self, X, y=None, reset=True):
        """Convert y to float, regressors cannot accept int."""
        if y is not None:
            y = check_array(y, dtype="float64", ensure_2d=False)
        return super()._validate_data(X=X, y=y, reset=reset)

    def _post_process_y(self, y):
        """Ensures output is float64 and squeeze."""
        return np.squeeze(y.astype("float64")), dict()

    def _pre_process_y(self, y):
        """Split y for multi-output tasks.
        """
        y, _ = super()._pre_process_y(y)

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
        if self.model_.loss not in (
            "mean_squared_error",
            self.root_mean_squared_error,
        ):
            warnings.warn(
                "R^2 is used to compute the score, it is advisable to use"
                " a compatible loss function. This class provides an R^2"
                " implementation in `KerasRegressor"
                ".root_mean_squared_error`."
            )

        return res

    @staticmethod
    @register_keras_serializable()
    def root_mean_squared_error(y_true, y_pred):
        """A simple Keras implementation of R^2 that can be used as a Keras
             loss function.

             Since `score` uses R^2, it is
             advisable to use the same loss/metric when optimizing the model.
        """
        ss_res = K.sum(K.square(y_true - y_pred), axis=0)
        ss_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
        return K.mean(1 - ss_res / (ss_tot + K.epsilon()), axis=-1)
