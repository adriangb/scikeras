==========
Quickstart
==========

Training a model
----------------

Below, we define our own Keras :class:`~tensorflow.keras.Sequential` and train
it on a toy classification dataset using SciKeras
:class:`.KerasClassifier`:

.. code:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from tensorflow import keras

    from scikeras.wrappers import KerasClassifier


    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    def get_model(hidden_layer_dim, meta):
        # note that meta is a special argument that will be
        # handed a dict containing input metadata
        n_features_in_ = meta["n_features_in_"]
        X_shape_ = meta["X_shape_"]
        n_classes_ = meta["n_classes_"]

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(hidden_layer_dim))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(n_classes_))
        model.add(keras.layers.Activation("softmax"))
        return model

    clf = KerasClassifier(get_model, hidden_layer_dim=100)

    clf.fit(X, y)
    y_proba = clf.predict_proba(X)


Note that SciKeras even chooses a loss function and compiles your model.
To override the default loss, simply specify a loss function:

.. code-block:: diff

    -KerasClassifier(get_model, hidden_layer_dim=100)
    +KerasClassifier(get_model, loss="categorical_crossentropy")

In this case, you would need to specify the loss since SciKeras
will not default to categorical crossentropy, even for one-hot
encoded targets.
See :ref:`loss-selection` for more details.

In an sklearn Pipeline
----------------------

Since :class:`.KerasClassifier` provides an sklearn-compatible
interface, it is possible to put it into an sklearn
:class:`~sklearn.pipeline.Pipeline`:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', clf),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)


Grid search
-----------

Another advantage of SciKeras is that you can perform an sklearn
:class:`~sklearn.model_selection.GridSearchCV` or
:class:`~sklearn.model_selection.RandomizedSearchCV`:

.. code:: python

    from sklearn.model_selection import GridSearchCV


    params = {
        "hidden_layer_dim": [50, 100, 200],
        "loss": ["sparse_categorical_crossentropy"],
        "optimizer": ["adam", "sgd"],
        "optimizer__learning_rate": [0.0001, 0.001, 0.1],
    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

    gs.fit(X, y)
    print(gs.best_score_, gs.best_params_)


What's next?
------------

Please visit the :ref:`tutorials` page to explore additional examples on using SciKeras!
