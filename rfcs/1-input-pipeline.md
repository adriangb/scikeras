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
from typing import Optional, Protocol, Sequence, Tuple, Type, Union

from numpy import ndarray
from numpy.typing import ArrayLike


class ArrayTransformer:

    def __init__(self, model: BaseWrapper) -> None:
        self.model = model

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool = True) -> Tuple[ndarray, Union[ndarray, None], Union[ndarray, None]]:
        return X, y, sample_weight

    def transform_output_proba(self, y_pred_proba: ndarray) -> ndarray:
        return y_pred_proba


class DatasetTransformer:

    def __init__(self, model: BaseWrapper) -> None:
        self.model = model

    def transform_input(self, data: tf.data.Dataset, initialize: bool = True) -> tf.data.Dataset:
        return data

    def transform_output_proba(self, y_pred_proba: ndarray) -> ndarray:
        return y_pred_proba
```

SciKeras will initialize the pipeline of transformers by passing a reference to the current estimator,
and will then iterate through them as described above:

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

    def _transform_input(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool) -> tf.data.Dataset:
        # sample implementation
        if initialize:
            self.array_transformers_ = [t(self) for t in self.array_transformers]
            self.dataset_transformers_ = [t(self) for t in self.dataset_transformers]
        if not isinstance(X, tf.data.Dataset):
            for t in self.array_transformers_:
                X, y, sample_weight = t.transform_input(X, y, sample_weight, initialize=initialize)
            data = tf.data.Dataset.from_tensors(X, y, sample_weight)
        else:
            data = X
        for t in self.dataset_transformers_:
            data = t.transform_input(data, initialize=initialize)
        # build self.model_, etc.
        return data

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
            y_pred = t.transform_output_proba(y_pred)
        return y_pred
```


## Example Implementations

### One-hot encode targets

This moves one-hot encoding out of `ClassifierLabelEncoder`.
This means the transformation can be applied to any input, including tf.data.Datasets.
Performance should also be better because TensorFlow lazily applies and optimizes `map` operations on `Dataset`.

```python

def is_ohe_dataset(data: tf.data.Dataset) -> bool:
    target_shape = data.element_spec[1].shape
    if len(target_shape) != 2 or target_shape[1] == 1:
        return False
    y = next(iter(data))[1]
    return tf.math.reduce_all(tf.math.reduce_sum(y, axis=1) == 1, axis=0).numpy()


class DatasetOneHotEncoder(DatasetTransformer):

    def transform_input(self, data: tf.data.Dataset, initialize: bool = True) -> tf.data.Dataset:
        if initialize:
            loss_requires_ohe = is_categorical_crossentropy(getattr(self.model, "loss"))
            self.needs_ohe_ = loss_requires_ohe and not is_ohe_dataset(data)
            if self.needs_ohe_:
                user_supplied_classes = getattr(self.model, "classes_", None)
                self.classes_ = user_supplied_classes if user_supplied_classes is not None else tf.unique(next(iter(data))[1])[1]
        if self.needs_ohe_:
            data = data.map(lambda X, y, sample_weight: (X, tf.one_hot(y, indices=self.classes_, depth=len(self.classes_))))
        return data

    def transform_output_proba(self, y_pred_proba: ndarray) -> ndarray:
        return np.argmax(y_pred_proba, axis=1)
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

```python
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


class _InputValidator(ArrayTransformer):

    def __init__(self, model: BaseWrapper, y_numeric: bool):
        super().__init__(model)
        self.y_numeric = y_numeric

    def transform_input(self, X: ArrayLike, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike], *, initialize: bool = True) -> Tuple[ndarray, Union[ndarray, None], Union[ndarray, None]]:            
        if y is not None:
            assert len(X) == len(y), "X and y must be of the same lenght"
            y = check_array(
                y,
                ensure_2d=False,
                allow_nd=False,
                dtype=_check_array_dtype(y, force_numeric=self.y_numeric),
            )
            y_dtype_ = y.dtype
            y_ndim_ = y.ndim
            if reset:
                self.model.target_type_ = self._type_of_target(y)
                self.model.y_dtype_ = y_dtype_
                self.model.y_ndim_ = y_ndim_
            else:
                ...  # raise errors
        if X is not None:
            X = check_array(
                X, allow_nd=True, dtype=_check_array_dtype(X, force_numeric=True)
            )
            X_dtype_ = X.dtype
            X_shape_ = X.shape
            n_features_in_ = X.shape[1]
            if reset:
                self.model.X_dtype_ = X_dtype_
                self.model.X_shape_ = X_shape_
                self.model.n_features_in_ = n_features_in_
            else:
                ...  # raise errors
        return X, y


RegressorInputValidator = functools.partial(_InputValidator, y_numeric=True)
ClassifierInputValidator = functools.partial(_InputValidator, y_numeric=False)
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

- #167
- #106 / #143
- #209
- #148 (by allowing users to implement it)
- #111
- #167

## Outstanding questions

Some outstanding issues:

1. API for transforming prediciton probabilties into class predictions.
This only applies to classifiers, but can it be generalized and included into the base interface?
2. Validations that require a model to be built. For example, checking the model's output shape (#106, #143).
3. Transformations involving not just the data but other parameters passed to Keras' `fit`/`predict` (#167).
