## Background

One of the primary functions of the Scikit-Learn wrappers for Keras is fielding conversion from Scikit-Learn's data types
to TensorFlow's data types. An example of these conversions is integer-encoding or one-hot encoding categorical targets.

Originally, these conversions were hardcoded into the wrappers in an ad-hoc manner (see here).
SciKeras introduced the concept of `TargetTransformer` and `InputTransformer`,
two Scikit-Learn style transformers that formalize this data conversion framework and allow users to insert their own custom pipeline.

Currently, user-customization requires subclassing the wrappers,
and composability is only provided via meta-transformers (i.e. a Scikit-Learn pipeline) of transformers.

Seperately, SciKeras also implements data validations that mirrors what most Scikit-Learn estimators implement,
for example to assert that `X` and `y` are of the same length or that `y` is purely numeric for regressors.
SciKeras also validates and inspects the model, for example to make sure that the output shape matches the
target's shape.
These checks are helpful for simple models, but may be too restrictive for more complex scenarios, like
multi-input/output models.

This RFC proposes a unifified interface for composable data transformations and validations.
The goal is to provide a pipeline of default transformations and validations that cover the simple use cases,
while allowing users to easily remove checks or add transformation steps for more advanced use cases.

Some of the speicific functional requirements are:
1. Able to implement the default (i.e. current) data transformations and validations. This includes:
 - Integer encoding targets for classifiers.
 - One-hot encoding targets for classifiers using the categorical crossentropy loss.
 - Converting class probability predictions into class predictions for classifiers.
2. Able to implement user-defined transformations, including:
 - Splitting the input and/or target into multiple inputs/outputs.
 - Reshaping 2D inputs into 3D.
3. Able to operate on array-like data (lists, Numpy arrays, Pandas DataFrames, etc.) as well as `tf.data.Dataset`s.
4. Composable and modifiable without subclassing.

## Proposal

This proposal consists of 2 pipelines:
1. A pipeline for preparing array-like data for a `tf.data.Dataset`. This would include, for example, integer-encoding object-dtype arrays (`tf.Tensors` can't hold objects).
2. A pipeline for applying transformations to `tf.data.Dataset`s. For example, one-hot encoding targets for classifiers using the categorical crossentropy loss.

If the data comes in as a `tf.data.Dataset`, the first pipeline is skipped. If not, it is run and the output is converted to a `Dataset`.
The second pipeline is then run in all cases.

These pipelines will consist of chained transformers implementing a Scikit-Learn-like interface, but without being restricted to the exact Scikit-Learn API.

```python
from typing import Any, Dict, Optional, Tuple, Sequence, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from scikeras.wrappers import BaseWrapper


Data = Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]
ArrayLike = Union[Sequence, np.ndarray]


class NotInitializedError(Exception):
    ...


class ArrayTransformer:

    def set_model(self, model: "BaseWrapper") -> None:
        self.model = model

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool = True) -> Data:
        return X, y, sample_weight

    def transform_output(self, y_pred_proba: np.ndarray, y: Union[np.ndarray, None]) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        return y_pred_proba, None

    def get_meta(self) -> Dict[str, Any]:
        return {}


class DatasetTransformer:

    def set_model(self, model: "BaseWrapper") -> None:
        self.model = model

    def transform_input(self, data: tf.data.Dataset, *, initialize: bool = True) -> tf.data.Dataset:
        return data

    def transform_output(self, y_pred_proba: np.ndarray, y: Union[np.ndarray, None]) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        return y_pred_proba, None

    def get_meta(self) -> Dict[str, Any]:
        return {}

```

SciKeras will initialize the pipeline of transformers by calling `set_model` with a reference to the current estimator.
This is exactly how Keras handles callbacks ([`set_model`](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/callbacks.py#L302), [in-place modification of `model`](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/callbacks.py#L1153) in History).

Then SciKeras will then iterate through them, similar to a Scikit-Learn pipeline

```python
import itertools
from typing import Optional, Protocol, Sequence, Tuple, Type, Union

from numpy import ndarray
from numpy.typing import ArrayLike
import tensorflow as tf


Input = Union[tf.data.Dataset, ArrayLike]


class BaseWrapper:
    
    def __init__(
        self,
        array_transformers: Sequence[Callable[["BaseWrapper", ArrayTransformer]]] = tuple(),  # a tuple with default transformers
        dataset_transformers: Sequence[Callable[["BaseWrapper", ArrayTransformer]]] = tuple(),
    ) -> None:
        self.array_transformers = array_transformers
        self.dataset_transformers = dataset_transformers

    def _transform_input(self, X: Union[tf.data.Dataset, ArrayLike], y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool) -> tf.data.Dataset:
        if initialize:
            for tf in itertools.chain(self.array_pipeline, self.dataset_pipeline):
                tf.set_model(self)
        if isinstance(X, tf.data.Dataset):
            self._numpy_input = False
            data = X
        else:
            self._numpy_input = True
            for t in self.array_pipeline:
                X, y, sample_weight = t.transform_input(X, y, sample_weight, initialize=initialize)
            data = tf.data.Dataset.from_tensors((X, y, sample_weight))
        for t in self.dataset_pipeline_:
            data = t.transform_input(data, initialize=initialize)
        if self._numpy_input:
            # keep as numpy arrays to allow validation_split and such to work
            X, y, sample_weight = next(iter(data))
            return X, y, sample_weight
        else:
            return data, None, None

    def initialize(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> "BaseWrapper":
        self._transform_input(X, y, sample_weight, initialize=True)
        return self

    def fit(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> "BaseWrapper":
        data = self._transform_input(X, y, sample_weight, initialize=True)
        ...
        return self

    def partial_fit(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> "BaseWrapper":
        initialize = hasattr(self, "model_")  # or other check
        data = self._transform_input(X, y, sample_weight, initialize=initialize)
        ...
        return self

    def predict(self, X: Input) -> ndarray:
        data = self._transform_input(X, y, sample_weight, initialize=False)
        y_pred = self.model_.predict(data)
        for t in itertools.chain(
            reversed(self.dataset_transformers_),
            reversed(self.array_transformers_)
        ):
            y_proba = t.transform_output(y_proba, None)
        return y_proba
```

Classifiers (as well as LTR or other learning problems where `y` is not the raw prediction probabilties) can use this interface
to convert probabilities to class predictions, or to modify the probabilites themselves. Two small examples:

```python
class BinaryPredictionReshaper(DatasetTransformer):

    def transform_input(self, data: tf.data.Dataset, *, initialize: bool) -> tf.data.Dataset:
        if initialize:
            self._is_binary = ... # call type_of_target or other check
        return data

    def transform_output(self, y_pred_proba: np.ndarray, y: Union[np.ndarray, None]) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        shp = y_pred_proba.shape
        if self._is_binary and len(shp) == 1 or len(shp) == 2 and shp[1] == 1:
            # single sigmoid output, reshape to a 2D array of predicitons, which is what sklearn expects
            y_pred_proba = np.column_stack([1-y_pred_proba, y_pred_proba])
        return y_pred_proba, y

class ClassifierPredictionDecoder(DatasetTransformer):

    def transform_output(self, y_pred_proba: np.ndarray, y: Union[np.ndarray, None]) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if y is None:
            y = np.argmax(y_pred_proba, axis=1)
        return y_pred_proba, y
```


## Example Implementations

### One-hot encode targets

This moves one-hot encoding out of `ClassifierLabelEncoder`.
This means the transformation can be applied to any input, including tf.data.Datasets.
Performance should also be better because TensorFlow lazily applies and optimizes `map` operations on `Dataset`.

```python
def _is_ohe_dataset(data: tf.data.Dataset) -> bool:
    target_shape = data.element_spec[1].shape
    if len(target_shape) != 2 or target_shape[1] == 1:
        return False  # needs to be 2D with >=2 columns to be one-hot encoded
    y = next(iter(data))[1]
    return tf.math.reduce_all(tf.math.reduce_sum(y, axis=1) == 1, axis=0).numpy()  # all rows add up to 1


class ClassifierOneHotEncoder(DatasetTransformer):
    """One-hot encode the target if the loss function is categorical crossentropy.
    """

    def transform_input(self, data: tf.data.Dataset, *, initialize: bool) -> tf.data.Dataset:
        if initialize:
            loss = getattr(self.model, "loss", None)
            loss_requires_ohe = False if loss is None else is_categorical_crossentropy(loss)
            self._needs_ohe = loss_requires_ohe and not _is_ohe_dataset(data)
            if self._needs_ohe:
                user_supplied_classes = getattr(self.model, "classes_", None)
                self.classes_ = user_supplied_classes if user_supplied_classes is not None else tf.unique(next(iter(data))[1])[1]
        if self._needs_ohe:
            data = data.map(lambda X, y, sample_weight: (X, tf.one_hot(y, indices=self.classes_, depth=len(self.classes_)), sample_weight))
        return data

    def transform_output(self, y_pred_proba: np.ndarray, y: Union[np.ndarray, None]) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if y is None and self._needs_ohe:
            y = np.argmax(y_pred_proba, axis=1)
        return y_pred_proba, y
```

The we add this to the default list of transformers for classifiers:

```python
class KerasClassifier:
    
    def __init__(
        self,
        dataset_transformers = (DatasetOneHotEncoder,)
        ...
    ):
        ...
```

## Validate array-like data

This mirrors the current implementation of `BaseWrapper._validate_data`.

Moving that check to this interface would:
1. Only apply these checks array-like inputs.
2. Move the implementation from a hardcoded private method to be stand-alone (making it easier to test, etc.).
3. Make usage of these checks both composable and optional.

This implementation could also be split up:
1. Transform X & y into arrays.
2. Check shapes, styles, etc. as tf.data.Dataset

```python
def _check_array_dtype(arr: ArrayLike, force_numeric: bool):
    if not isinstance(arr, np.ndarray):
        return _check_array_dtype(np.asarray(arr), force_numeric=force_numeric)
    elif (
         arr.dtype.kind in ("O", "U", "S" or not force_numeric
    ):
        return None  # check_array won't do any casting with dtype=None
    else:
        # default to TFs backend float type
        # instead of float64 (sklearn's default)
        return tf.keras.backend.floatx()


class ValidateFeaturesArray(ArrayTransformer):

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool) -> Data:
        X = check_array(
            X,
            allow_nd=True,
            ensure_2d=True,
            dtype=_check_array_dtype(X, force_numeric=True)
        )
        n_feautres_in_ = X.shape[1]
        if initialize:
            self.n_feautres_in_ = n_feautres_in_
        else:
            if self.n_feautres_in_ != n_feautres_in_:
                raise ValueError(
                    f"Expected X to have {self.n_feautres_in_} features, but got {n_feautres_in_} features"
                )
        return X, y, sample_weight
    
    def get_meta(self) -> Dict[str, Any]:
        return {"n_features_in_": self.n_feautres_in_}


class ValidateClassifierTargetArray(ArrayTransformer):

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool) -> Data:
        if y is not None:
            y = check_array(
                y,
                ensure_2d=False,
                allow_nd=False,
                dtype=_check_array_dtype(y, force_numeric=False),
            )
            classes_ = np.unique(y)
            if initialize:
                self.classes_ = classes_
            else:
                if self.classes_ != classes_:
                    raise ValueError(
                        f"Expected y to have {self.classes_} classes, but got {classes_} classes"
                    )
        return X, y, sample_weight

    def get_meta(self) -> Dict[str, Any]:
        return {"classes_": self.classes_}


class ValidateRegressorTargetArray(ArrayTransformer):

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool) -> Data:
        if y is not None:
            y = check_array(
                y,
                ensure_2d=False,
                allow_nd=False,
                dtype=_check_array_dtype(y, force_numeric=True),
            )
        return X, y, sample_weight


class ValidateSampleWeight(ArrayTransformer):

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool) -> Data:
        if isinstance(sample_weight, numbers.Number):
            sample_weight = np.full(shape=(len(X),), fill_value=sample_weight)
        if sample_weight is not None:
            sample_weight = check_array(
                sample_weight,
                accept_sparse=False,
                ensure_2d=False,
                dtype=tf.keras.backend.floatx(),
                copy=False,
            )
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")
            if np.all(sample_weight == 0):
                raise ValueError(
                    "No training samples had any weight; only zeros were passed in sample_weight."
                    " That means there's nothing to train on by definition, so training can not be completed."
                )
        return X, y, sample_weight

```

Now we can add this to the default array transformers:

```python
class KerasClassifier:
    def __init__(
        self,
        array_transformers = (ClassifierInputValidator,)
    ):
        ...


class KerasRegressor:
    def __init__(
        self,
        array_transformers = (RegressorInputValidator,)
    ):
        ...
```

## Issues this can potentially resolve

- [#160](https://github.com/adriangb/scikeras/pull/160), "Dealing with variable length inputs"
- [#106](https://github.com/adriangb/scikeras/pull/106), "Validate inputs to Keras model"
- [#209](https://github.com/adriangb/scikeras/pull/209), "ENH: programmatic validations and error handling"
- #148 (by allowing users to implement it)
- [#111](https://github.com/adriangb/scikeras/pull/111), "move data validation to a modular interface"
- #167

## Outstanding questions

Some outstanding issues:

1. Validations that require a model to be built. For example, checking the model's output shape (#106, #143).
2. Transformations involving not just the data but other parameters passed to Keras' `fit`/`predict` (#167).
