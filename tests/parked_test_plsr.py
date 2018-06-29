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

from hoggorm import nipalsPLS1 as PLSR1
from hoggorm import nipalsPLS2 as PLSR2


# If the following equation is element-wise True, then allclose returns True.
# absolute(a - b) <= (atol + rtol * absolute(b))
# default: rtol=1e-05, atol=1e-08
rtol = 1e-05
atol = 1e-08


ATTRS = [
    'modelSettings',
]


def test_api2_verify(ldat, sdat):
    print(ldat.shape, sdat.shape)
    plsr1 = PLSR2(arrX=ldat, arrY=sdat, numComp=3, Xstand=False, Ystand=False, cvType=["loo"])
    print('plsr1', plsr1)
    plsr2 = PLSR2(ldat, sdat)
    print('plsr2', plsr2)
    plsr3 = PLSR2(ldat, sdat, numComp=300, cvType=["loo"])
    print('plsr3', plsr3)
    plsr4 = PLSR2(arrX=ldat, arrY=sdat, cvType=["loo"], numComp=5, Xstand=False, Ystand=False)
    print('plsr4', plsr4)
    plsr5 = PLSR2(ldat, sdat, Xstand=True, Ystand=True)
    print('plsr5', plsr5)
    assert True





def test_api1_verify(ldat, sdat):
    svec = np.column_stack(sdat[:,0]).T
    print('\n')
    print('Shapes:', ldat.shape, svec.shape)
    print(ldat)
    print(svec)
    plsr1 = PLSR1(arrX=ldat, vecy=svec, numComp=3, Xstand=False, Ystand=False, cvType=["loo"])
    print('plsr1', plsr1)
    plsr2 = PLSR1(ldat, svec)
    print('plsr2', plsr2)
    plsr3 = PLSR1(ldat, svec, numComp=300, cvType=["loo"])
    print('plsr3', plsr3)
    plsr4 = PLSR1(arrX=ldat, vecy=svec, cvType=["loo"], numComp=5, Xstand=False, Ystand=False)
    print('plsr4', plsr4)
    plsr5 = PLSR1(ldat, svec, Xstand=True, Ystand=True)
    print('plsr5', plsr5)
    assert True
