'''
PCA testing ideas:
 * Check max_pc
 * calc_n_pc to max_pc
 * Results sets and shapes
 * Check zero variance test
 * Check that res is a new copy each time (unique id?)
 * Calculation with missing data
'''
import os.path as osp

import numpy as np

import pytest

from hoggorm import nipalsPCA as PCA


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
    # 'X_scores_predict',
    'cvTrainAndTestData',
    'corrLoadingsEllipses',
]


def test_api_verify(pcacached, ldat):

    for fn in ATTRS:
        res = getattr(pcacached, fn)()
        print(fn, type(res))
        if isinstance(res, np.ndarray):
            print(res.shape)
    res = pcacached.X_scores_predict(ldat)
    print('X_scores_predict', type(res))
    print(res.shape)


def test_constructor_api_variants(ldat):
    print("\n")
    pca1 = PCA(ldat)
    print("pca1")
    pca2 = PCA(ldat, numComp=200, Xstand=False)
    print("pca2")
    # FIXME: This will hang for given dataset
    # pca3 = PCA(ldat, Xstand=True, cvType=["loo"])
    # print("pca3")
    pca4 = PCA(ldat, numComp=2, cvType=["loo"])
    print("pca4")
    pca5 = PCA(ldat, numComp=2, Xstand=False, cvType=["loo"])
    print("pca5")


def test_compare_reference(pcaref, pcacached):
    rname, refdat = pcaref
    res = getattr(pcacached, rname)()
    if refdat is None:
        dump_res(rname, res)
        assert False, "Missing reference data for {}, data is dumpet".format(rname)
    elif not np.allclose(res, refdat, rtol=rtol, atol=atol):
        dump_res(rname, res)
        assert False, "Difference in {}, data is dumpet".format(rname)
    else:
        assert True


@pytest.fixture(params=["X_scores", "X_loadings"])
def pcaref(request, datafolder):
    rname = request.param
    refn = "ref_{}.tsv".format(rname.lower())
    try:
        refdat = np.loadtxt(osp.join(datafolder, refn))
    except FileNotFoundError:
        refdat = None

    return (rname, refdat)


@pytest.fixture(scope="module")
def pcacached(ldat):
    return PCA(ldat, cvType=["loo"])


def dump_res(rname, dat):
    dumpfolder = osp.realpath(osp.dirname(__file__))
    dumpfn = "dump_{}.tsv".format(rname.lower())
    np.savetxt(osp.join(dumpfolder, dumpfn), dat, fmt='%.9e', delimiter='\t')
