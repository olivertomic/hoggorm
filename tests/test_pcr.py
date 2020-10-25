'''
Test whether PCR results are as expected.
'''

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
    'X_scores_predict',
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


@pytest.fixture(scope="module")
def pcrcached(cfldat, csedat):
    """
    Run PCR from current hoggorm installation and compare results against reference results.
    """
    return PCR(arrX=cfldat, arrY=csedat, cvType=["loo"])


testMethods = [
    "X_scores", "X_loadings", "X_corrLoadings", "X_cumCalExplVar_indVar",
    "X_cumCalExplVar", "Y_loadings", "Y_corrLoadings",
    "Y_cumCalExplVar_indVar", "Y_cumCalExplVar"
]


@pytest.fixture(params=testMethods)
def pcrref(request, datafolder):
    """
    Load reference numerical results from file.
    """
    rname = request.param
    refn = "ref_PCR_{}.tsv".format(rname[0].lower() + rname[1:])
    try:
        refdat = np.loadtxt(datafolder.joinpath(refn))
    except FileNotFoundError:
        refdat = None

    return (rname, refdat)


def test_compare_reference(pcrref, pcrcached, dump_res):
    """
    Check whether numerical outputs are the same (or close enough).
    """
    rname, refdat = pcrref
    res = getattr(pcrcached, rname)()

    # print('res:')
    # print(res[:10, :3], '\n\n')
    # print('ref:')
    # print(refdat[:10, :3], '\n\n')
    # 1 / 0

    if refdat is None:
        dump_res(rname, res)
        assert False, "Missing reference data for {}, data is dumped".format(
            rname)
    elif rname == 'X_cumCalExplVar' or rname == 'Y_cumCalExplVar':
        if not np.allclose(np.array(res[:3]), refdat[:3], rtol=rtol,
                           atol=atol):
            dump_res(rname, res)
            assert False, "Difference in {}, data is dumped".format(rname)
    elif not np.allclose(res[:, :3], refdat[:, :3], rtol=rtol, atol=atol):
        dump_res(rname, res)
        assert False, "Difference in {}, data is dumped".format(rname)
    else:
        assert True


def test_api_verify(pcrcached, cfldat):
    """
    Check whether all methods in list ATTR are also available in nipalsPCR class.
    """
    # Loop through all methods in ATTR
    for fn in ATTRS:
        if fn == 'X_scores_predict':
            res = pcrcached.X_scores_predict(Xnew=cfldat)
            print('fn:', 'X_scores_predict')
            print('type(res):', type(res))
            print('shape:', res.shape, '\n\n')
        else:
            res = getattr(pcrcached, fn)()
            print('fn:', fn)
            print('type(res):', type(res))
            if isinstance(res, np.ndarray):
                print('shape:', res.shape, '\n\n')
            else:
                print('\n')


def test_constructor_api_variants(cfldat, csedat):
    """
    Check whether various combinations of keyword arguments work.
    """
    pcr1 = PCR(arrX=cfldat,
               arrY=csedat,
               numComp=3,
               Xstand=False,
               Ystand=False,
               cvType=["loo"])
    print('pcr1', pcr1)
    pcr2 = PCR(cfldat, csedat)
    print('pcr2', pcr2)
    pcr3 = PCR(cfldat, csedat, numComp=200, cvType=["loo"])
    print('pcr3', pcr3)
    pcr4 = PCR(arrX=cfldat,
               arrY=csedat,
               cvType=["loo"],
               numComp=5,
               Xstand=False,
               Ystand=False)
    print('pcr4', pcr4)
    pcr5 = PCR(arrX=cfldat, arrY=csedat, Xstand=True, Ystand=True)
    print('pcr5', pcr5)
    pcr6 = PCR(arrX=cfldat,
               arrY=csedat,
               numComp=2,
               Xstand=False,
               cvType=["KFold", 3])
    print('pcr6', pcr6)
    pcr7 = PCR(arrX=cfldat,
               arrY=csedat,
               numComp=2,
               Xstand=False,
               cvType=["lolo", [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]])
    print('pcr7', pcr7)
    assert True
