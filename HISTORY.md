# History

## 0.1.8 (2020-06-30)

* Add support for partial fitting and warm starts (#17, thank you @stsievert!).
* Add support for random states via the `random_state` parameter (#27).
* Scikit-Learn SLEP10 compliance (#26).
* Fix unnecessary data reshaping warnings (#23).

## 0.1.7 (2020-05-18)

* Versioning fix.

## 0.1.6 (2020-05-18)

* Rename repo.
* Derive BaseWrapper from BaseEstimator.
* Python 3.8 support.

## 0.1.5.post1 (2020-05-13)

* Deprecate Python 3.5 (#11).
* Fix prebuilt model bug (#5).
* Clean up serialization (#7).
* Implement scikit-learn estimator tests (#10).

## 0.1.4 (2020-04-12)

* Offload output type detection for classification to `sklearn.utils.multiclass.type_of_target`.
* Add documentation.
* Some file cleanup.

## 0.1.3 (2020-04-11)

* First release on PyPI.
