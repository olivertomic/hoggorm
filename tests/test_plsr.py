'''
FIXME: PCA testing ideas:
 * Well known datasets (iris)
 * Combinations of input parameters
 * Edge case datasets
 * Big matrix for performance testing / profiling
 * Illegale data and error handling (zero variance)
 * Integer and float type matrix
'''
import os.path as osp

import numpy as np

import pytest

from hoggorm import nipalsPLS2 as PLSR


# If the following equation is element-wise True, then allclose returns True.
# absolute(a - b) <= (atol + rtol * absolute(b))
# default: rtol=1e-05, atol=1e-08
rtol = 1e-05
atol = 1e-08


ATTRS = [
    'modelSettings',
]


def test_api_verify(ldat, sdat):
    print(ldat.shape, sdat.shape)
    plsr1 = PLSR(arrX=ldat, arrY=sdat, numComp=3, Xstand=False, Ystand=False, cvType=["loo"])
    print('plsr1', plsr1)
    plsr2 = PLSR(ldat, sdat)
    print('plsr2', plsr2)
    plsr3 = PLSR(ldat, sdat, numComp=300, cvType=["loo"])
    print('plsr3', plsr3)
    plsr4 = PLSR(arrX=ldat, arrY=sdat, cvType=["loo"], numComp=5, Xstand=False, Ystand=False)
    print('plsr4', plsr4)
    plsr5 = PLSR(ldat, sdat, Xstand=True, Ystand=True)
    print('plsr5', plsr5)
    assert True
