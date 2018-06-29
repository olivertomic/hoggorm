# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:45:30 2018

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

from hoggorm import nipalsPLS1 as PLS1


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
    'Y_cumCalExplVar',
    'Y_predCal',
    'Y_PRESSE',
    'Y_MSEE',
    'Y_RMSEE',
    'Y_valExplVar',
    'Y_cumValExplVar',
    'Y_predVal',
    'Y_PRESSCV',
    'Y_MSECV',
    'Y_RMSECV',
    'cvTrainAndTestData',
    'corrLoadingsEllipses',
]


def test_api_verify(pls1cached, cfldat):
    """
    Check if all methods in list ATTR are also available in nipalsPCA class.
    """
    # First check those in list ATTR above. These don't have input parameters.
    for fn in ATTRS:
        res = getattr(pls1cached, fn)()
        print(fn, type(res), '\n')
        if isinstance(res, np.ndarray):
            print(res.shape, '\n')
    
    # Now check those with input paramters
    res = pls1cached.X_scores_predict(Xnew=cfldat)
    print('X_scores_predict', type(res), '\n')
    print(res.shape)


def test_constructor_api_variants(cfldat, csecol2dat):
    print(cfldat.shape, csecol2dat.shape)
    pls1_1 = PLS1(arrX=cfldat, vecy=csecol2dat, numComp=3, Xstand=False, cvType=["loo"])
    print('pls1_1', pls1_1)
    pls1_2 = PLS1(cfldat, csecol2dat)
    print('pls1_2', pls1_2)
    pls1_3 = PLS1(cfldat, csecol2dat, numComp=300, cvType=["loo"])
    print('pls1_3', pls1_3)
    pls1_4 = PLS1(arrX=cfldat, vecy=csecol2dat, cvType=["loo"], numComp=5, Xstand=False)
    print('pls1_4', pls1_4)
    pls1_5 = PLS1(arrX=cfldat, vecy=csecol2dat, Xstand=True)
    print('pls1_5', pls1_5)
    pls1_6 = PLS1(arrX=cfldat, vecy=csecol2dat, numComp=2, Xstand=False, cvType=["KFold", 3])
    print('pls1_6', pls1_6)
    pls1_7 = PLS1(arrX=cfldat, vecy=csecol2dat, numComp=2, Xstand=False, cvType=["lolo", [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]])
    print('pls1_7', pls1_7)
    assert True


def test_compare_reference(pls1ref, pls1cached):
    rname, refdat = pls1ref
    res = getattr(pls1cached, rname)()
    if refdat is None:
        dump_res(rname, res)
        assert False, "Missing reference data for {}, data is dumped".format(rname)
    elif not np.allclose(res[:, :2], refdat[:, :2], rtol=rtol, atol=atol):
        dump_res(rname, res)
        assert False, "Difference in {}, data is dumped".format(rname)
    else:
        assert True


testMethods = ["X_scores", "X_loadings", "X_corrLoadings", "X_cumCalExplVar_indVar"]
@pytest.fixture(params=testMethods)
def pls1ref(request, datafolder):
    rname = request.param
    refn = "ref_PLS1_{}.tsv".format(rname.lower())
    try:
        refdat = np.loadtxt(osp.join(datafolder, refn))
    except FileNotFoundError:
        refdat = None

    return (rname, refdat)


@pytest.fixture(scope="module")
def pls1cached(cfldat, csecol2dat):
    return PLS1(arrX=cfldat, vecy=csecol2dat, cvType=["loo"])


def dump_res(rname, dat):
    dumpfolder = osp.realpath(osp.dirname(__file__))
    dumpfn = "dump_PLS1_{}.tsv".format(rname.lower())
    np.savetxt(osp.join(dumpfolder, dumpfn), dat, fmt='%.9e', delimiter='\t')