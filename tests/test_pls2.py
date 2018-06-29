# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:22:58 2018

@author: olive
"""

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

from hoggorm import nipalsPLS2 as PLS2


# If the following equation is element-wise True, then allclose returns True.
# absolute(a - b) <= (atol + rtol * absolute(b))
# default: rtol=1e-05, atol=1e-08
rtol = 1e-05
atol = 1e-08


ATTRS = [
    'modelSettings',
    'X_means',
    'X_scores',
    'X_loadings',
    'X_corrLoadings',
    'X_residuals',
    'X_calExplVar',
    'X_cumCalExplVar_indVar',
    'X_cumCalExplVar',
    'X_predCal',
    'X_PRESSE_indVar',
    'X_PRESSE',
    'X_MSEE_indVar',
    'X_MSEE',
    'X_RMSEE_indVar',
    'X_RMSEE',
    'X_valExplVar',
    'X_cumValExplVar_indVar',
    'X_cumValExplVar',
    'X_predVal',
    'X_PRESSCV_indVar',
    'X_PRESSCV',
    'X_MSECV_indVar',
    'X_MSECV',
    'X_RMSECV_indVar',
    'X_RMSECV',
    #'X_scores_predict',
    'Y_means',
    'Y_loadings',
    'Y_corrLoadings',
    'Y_residuals',
    'Y_calExplVar',
    'Y_cumCalExplVar_indVar',
    'Y_cumCalExplVar',
    'Y_predCal',
    'Y_PRESSE_indVar',
    'Y_PRESSE',
    'Y_MSEE_indVar',
    'Y_MSEE',
    'Y_RMSEE_indVar',
    'Y_RMSEE',
    'Y_valExplVar',
    'Y_cumValExplVar_indVar',
    'Y_cumValExplVar',
    'Y_predVal',
    'Y_PRESSCV_indVar',
    'Y_PRESSCV',
    'Y_MSECV_indVar',
    'Y_MSECV',
    'Y_RMSECV_indVar',
    'Y_RMSECV',
    'cvTrainAndTestData',
    'corrLoadingsEllipses',
]


def test_api_verify(pls2cached, cfldat):
    """
    Check if all methods in list ATTR are also available in nipalsPCA class.
    """
    # First check those in list ATTR above. These don't have input parameters.
    for fn in ATTRS:
        res = getattr(pls2cached, fn)()
        print(fn, type(res), '\n')
        if isinstance(res, np.ndarray):
            print(res.shape, '\n')
    
    # Now check those with input paramters
    res = pls2cached.X_scores_predict(Xnew=cfldat)
    print('X_scores_predict', type(res), '\n')
    print(res.shape)


def test_constructor_api_variants(cfldat, csedat):
    print(cfldat.shape, csedat.shape)
    pls2_1 = PLS2(arrX=cfldat, arrY=csedat, numComp=3, Xstand=False, Ystand=False, cvType=["loo"])
    print('pls2_1', pls2_1)
    pls2_2 = PLS2(cfldat, csedat)
    print('pls2_2', pls2_2)
    pls2_3 = PLS2(cfldat, csedat, numComp=300, cvType=["loo"])
    print('pls2_3', pls2_3)
    pls2_4 = PLS2(arrX=cfldat, arrY=csedat, cvType=["loo"], numComp=5, Xstand=False, Ystand=False)
    print('pls2_4', pls2_4)
    pls2_5 = PLS2(arrX=cfldat, arrY=csedat, Xstand=True, Ystand=True)
    print('pls2_5', pls2_5)
    pls2_6 = PLS2(arrX=cfldat, arrY=csedat, numComp=2, Xstand=False, cvType=["KFold", 3])
    print('pls2_6', pls2_6)
    pls2_7 = PLS2(arrX=cfldat, arrY=csedat, numComp=2, Xstand=False, cvType=["lolo", [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]])
    print('pls2_7', pls2_7)
    assert True


def test_compare_reference(pls2ref, pls2cached):
    rname, refdat = pls2ref
    res = getattr(pls2cached, rname)()
    if refdat is None:
        dump_res(rname, res)
        assert False, "Missing reference data for {}, data is dumped".format(rname)
    elif not np.allclose(res[:, :2], refdat[:, :2], rtol=rtol, atol=atol):
        dump_res(rname, res)
        assert False, "Difference in {}, data is dumped".format(rname)
    else:
        assert True


testMethods = ["X_scores", "X_loadings", "X_corrLoadings", "X_cumCalExplVar_indVar",
               "Y_loadings", "Y_corrLoadings", "Y_cumCalExplVar_indVar"]
@pytest.fixture(params=testMethods)
def pls2ref(request, datafolder):
    rname = request.param
    refn = "ref_PLS2_{}.tsv".format(rname.lower())
    try:
        refdat = np.loadtxt(osp.join(datafolder, refn))
    except FileNotFoundError:
        refdat = None

    return (rname, refdat)


@pytest.fixture(scope="module")
def pls2cached(cfldat, csedat):
    return PLS2(arrX=cfldat, arrY=csedat, cvType=["loo"])


def dump_res(rname, dat):
    dumpfolder = osp.realpath(osp.dirname(__file__))
    dumpfn = "dump_PLS2_{}.tsv".format(rname.lower())
    np.savetxt(osp.join(dumpfolder, dumpfn), dat, fmt='%.9e', delimiter='\t')
