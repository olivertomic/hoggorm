'''
Test whether PCA results are as expected.
'''
import numpy as np
import pytest
from hoggorm import nipalsPCA as PCA

# If the following equation is element-wise True, then allclose returns True.
# absolute(a - b) <= (atol + rtol * absolute(b))
# default: rtol=1e-05, atol=1e-08
rtol = 1e-05
atol = 1e-08

ATTRS = [
    'modelSettings', 'X_means', 'X_scores', 'X_loadings', 'X_corrLoadings',
    'X_residuals', 'X_calExplVar', 'X_cumCalExplVar_indVar', 'X_cumCalExplVar',
    'X_predCal', 'X_PRESSE_indVar', 'X_PRESSE', 'X_MSEE_indVar', 'X_MSEE',
    'X_RMSEE_indVar', 'X_RMSEE', 'X_valExplVar', 'X_cumValExplVar_indVar',
    'X_cumValExplVar', 'X_predVal', 'X_PRESSCV_indVar', 'X_PRESSCV',
    'X_MSECV_indVar', 'X_MSECV', 'X_RMSECV_indVar', 'X_RMSECV',
    'X_scores_predict', 'cvTrainAndTestData', 'corrLoadingsEllipses'
]


@pytest.fixture(scope="module")
def pcacached(cfldat):
    """
    Run PCA from current hoggorm installation and compare results against reference results.
    """
    return PCA(cfldat, cvType=["loo"])


testMethods = [
    "X_scores", "X_loadings", "X_corrLoadings", "X_cumCalExplVar_indVar",
    "X_cumCalExplVar"
]


@pytest.fixture(params=testMethods)
def pcaref(request, datafolder):
    """
    Load reference numerical results from file.
    """
    rname = request.param
    refn = "ref_PCA_{}.tsv".format(rname[0].lower() + rname[1:])
    print(refn)
    try:
        refdat = np.loadtxt(datafolder.joinpath(refn))
    except FileNotFoundError:
        refdat = None

    return (rname, refdat)


def test_compare_reference(pcaref, pcacached, dump_res):
    """
    Check whether numerical outputs are the same (or close enough).
    """
    rname, refdat = pcaref
    res = getattr(pcacached, rname)()

    if refdat is None:
        dump_res(rname, res)
        assert False, "Missing reference data for {}, data is dumped".format(
            rname)
    elif rname == 'X_cumCalExplVar':
        if not np.allclose(np.array(res[:3]), refdat[:3], rtol=rtol,
                           atol=atol):
            dump_res(rname, res)
            assert False, "Difference in {}, data is dumped".format(rname)
    elif not np.allclose(res[:, :3], refdat[:, :3], rtol=rtol, atol=atol):
        dump_res(rname, res)
        assert False, "Difference in {}, data is dumped".format(rname)
    else:
        assert True


def test_api_verify(pcacached, cfldat):
    """
    Check whether all methods in list ATTR are also available in nipalsPCA class.
    """
    # Loop through all methods in ATTR
    for fn in ATTRS:
        if fn == 'X_scores_predict':
            res = pcacached.X_scores_predict(Xnew=cfldat)
            print('fn:', 'X_scores_predict')
            print('type(res):', type(res))
            print('shape:', res.shape, '\n\n')
        else:
            res = getattr(pcacached, fn)()
            print('fn:', fn)
            print('type(res):', type(res))
            if isinstance(res, np.ndarray):
                print('shape:', res.shape, '\n\n')
            else:
                print('\n')


def test_constructor_api_variants(cfldat):
    """
    Check whether various combinations of keyword arguments work.
    """
    print("\n")
    pca1 = PCA(cfldat)
    print("pca1")
    pca2 = PCA(cfldat, numComp=200, Xstand=False)
    print("pca2")
    pca3 = PCA(cfldat, Xstand=True, cvType=["loo"])
    print("pca3")
    pca4 = PCA(cfldat, numComp=2, cvType=["loo"])
    print("pca4")
    pca5 = PCA(cfldat, numComp=2, Xstand=False, cvType=["loo"])
    print("pca5")
    pca6 = PCA(cfldat, numComp=2, Xstand=False, cvType=["KFold", 3])
    print("pca6")
    pca7 = PCA(cfldat,
               numComp=2,
               Xstand=False,
               cvType=["lolo", [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]])
    print("pca7")
