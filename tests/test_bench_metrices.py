import pytest
import numpy as np
from geostatbench import geostatbench as gsb


__author__ = "NilsWildt"
__copyright__ = "NilsWildt"
__license__ = "MIT"

def test_MAE():
    """Test MAE"""
    x1 = np.array([0,0,0])
    x2 = np.array([1,1,1])
    assert gsb.MAE(x1,x2) == 1
    # with pytest.raises(AssertionError):
    #     MAE(-10)

def test_RMSE():
    """Test MAE"""
    x1 = np.array([0,0,0])
    x2 = np.array([2,1,1])
    assert np.isclose(gsb.RMSE(x1,x2),np.sqrt(2))
    # with pytest.raises(AssertionError):
    #     MAE(-10)

def test_AESD():
    """Test MAE"""
    x1 = np.array([0,0,0])
    x2 = np.array([1,1,1])
    assert gsb.RMSE(x1,x2) == 1
    # with pytest.raises(AssertionError):
    #     MAE(-10)


# def test_PSRF():
#     """Test MAE"""
#     x1 = np.array([0,0,0])
#     x2 = np.array([1,1,1])
#     assert gsb.PSRF(x1,x2) == 1
#     # with pytest.raises(AssertionError):
#     #     MAE(-10)


def test_KS_distance():
    """Test MAE"""
    x1 = np.array([0,0,0])
    x2 = np.array([1,1,1])
    assert gsb.KS_distance(x1,x2) == 1
    # with pytest.raises(AssertionError):
    #     MAE(-10)
