```python
import itertools
from typing import Optional, Protocol, Sequence, Tuple, Union

from numpy import ndarray
from numpy.typing import ArrayLike
import tensorflow as tf


class ArrayTransformer(Protocol):

    model: "BaseWrapper"

    def __init__(self, model: "BaseWrapper") -> None:
        self.model = model

    def initialize(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike) -> None:
        ...

    def transform_in(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike) -> Tuple[ndarray, Union[ndarray, None], Union[ndarray, None]]:
        ...
    
    def initilize_transform_in(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike) -> Tuple[ndarray, Union[ndarray, None], Union[ndarray, None]]:
        self.initialize(X, y, sample_weight)
        return self.transform_in(X, y, sample_weight)

    def transform_out_pred_proba(self, y_pred_proba: ndarray) -> ndarray:
        ...


class DatasetTransformer:

    def initialize(self, data: tf.data.Dataset) -> None:
        ...

    def transform_in(self, data: tf.data.Dataset) -> tf.data.Dataset:
        ...

    def initilize_transform_in(self, data: tf.data.Dataset) -> tf.data.Dataset:
        self.initialize(data)
        return self.transform_in(data)

    def transform_out_pred_proba(self, y_pred_proba: ndarray) -> ndarray:
        ...


Input = Union[tf.data.Dataset, ArrayLike]


class BaseWrapper:
    
    def __init__(
        self,
        array_transformers: Sequence[ArrayTransformer] = tuple(),  # a tuple with default transformers
        dataset_transformers: Sequence[DatasetTransformer] = tuple(),
    ) -> None:
        self.array_transformers = array_transformers
        self.dataset_transformers = dataset_transformers
    
    def _transform_data(self, X: Input, y: Union[ArrayLike, None], sample_weight: Union[ArrayLike, None]):
        # sample implementation
        if not isinstance(X, tf.data.Dataset):
            for at in self.array_transformers:
                X, y, sample_weight = at.transform_in(X, y, sample_weight)
            data = tf.data.Dataset.from_tensors(X, y, sample_weight)
        else:
            data = X
        for dt in self.dataset_transformers:
            data = dt.transform_in(data)
        return data

    def _initialize(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> tf.data.Dataset:
        # sample implementation
        if not isinstance(X, tf.data.Dataset):
            for at in self.array_transformers:
                X, y, sample_weight = at.initilize_transform_in(X, y, sample_weight)
            data = tf.data.Dataset.from_tensors(X, y, sample_weight)
        else:
            data = X
        for dt in self.dataset_transformers:
            data = dt.initilize_transform_in(data)
        # build model, etc.
        return data

    def initialize(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> "BaseWrapper":
        self._initialize(X, y, sample_weight)
        return self

    def fit(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> "BaseWrapper":
        data = self.initialize(X, y, sample_weight)
        ...
        return self

    def partial_fit(self, X: Input, y: Optional[ArrayLike], sample_weight: Optional[ArrayLike]) -> "BaseWrapper":
        initialize = hasattr(self, "model_")  # or other check
        pipe = self.initialize if initialize else self._transform_data
        data = pipe(X, y, sample_weight)
        ...
        return self

    def predict(self, X: Input) -> ndarray:
        data = self._transform_data(X, None, None, reset=False)
        y_pred = self.model_.predict(data)
        for dt in itertools.chain(
            reversed(self.dataset_transformers),
            reversed(self.array_transformers)
        ):
            y_pred = dt.transform_out_preds(y_pred)
        return y_pred
```
