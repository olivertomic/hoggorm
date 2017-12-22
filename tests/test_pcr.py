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

from hoggorm import nipalsPCR as PCR


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
    pcr1 = PCR(arrX=ldat, arrY=sdat, numComp=3, Xstand=False, Ystand=False, cvType=["loo"])
    print('pcr1', pcr1)
    pcr2 = PCR(ldat, sdat)
    print('pcr2', pcr2)
    pcr3 = PCR(ldat, sdat, numComp=300, cvType=["loo"])
    print('pcr3', pcr3)
    pcr4 = PCR(arrX=ldat, arrY=sdat, cvType=["loo"], numComp=5, Xstand=False, Ystand=False)
    print('pcr4', pcr4)
    # pcr5 = PCR(ldat, sdat, Xstand=True, Ystand=True)
    # print('pcr5', pcr5)
    assert True
