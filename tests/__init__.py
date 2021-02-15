"""Unit test package for scikeras."""
import pytest

from sklearn.exceptions import DataConversionWarning  # noqa


# Force data conversion warnings to be come errors
pytestmark = pytest.mark.filterwarnings("error::sklearn.exceptions.DataConversionWarning")

# @pytest.mark.filterwarnings("ignore:api v1")
# def test_one():
#     assert api_v1() == 1
