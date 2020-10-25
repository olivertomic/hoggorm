'''
To be able to run test you have to install the hoggorm package.

You can either do a normal install
pip install hoggorm

or you can install in developer mode
pip install -e .
or
python setup.py develop
'''
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture(scope="session")
def datafolder() -> Path:
    return Path(__file__).parent.joinpath("test_data")


@pytest.fixture(scope="session")
def dump_res(datafolder):
    """
    Dumps information to file if reference data is missing or difference is larger than tolerance.
    """

    def dumb_it(rname, dat):
        dumpfn = "dump_PLS1_{}.tsv".format(rname.lower())
        np.savetxt(datafolder.joinpath(dumpfn), dat, fmt='%.9e', delimiter='\t')

    return dumb_it


@pytest.fixture(scope="session")
def get_test_data(datafolder):
    """Helper fixture to load data files with appropriate dtype and shape.

    """

    def load_data(name, dtype=np.float64, reshape=None):
        mat = np.loadtxt(datafolder.joinpath(name),
                         dtype=dtype,
                         skiprows=1)
        if reshape:
            mat = mat.reshape(*reshape)

        return mat

    return load_data


@pytest.fixture(scope="module")
def ldat(get_test_data):
    '''Read liking data and return as numpy array'''
    return get_test_data('source_l_dat.tsv', dtype=np.uint8)


@pytest.fixture(scope="module")
def sdat(get_test_data):
    '''Read sensory data and return as numpy array'''
    return get_test_data('source_s_dat.tsv')


@pytest.fixture(scope="module")
def cfldat(get_test_data):
    '''Read fluorescence spectra on cheese samples and return as numpy array'''
    return get_test_data('data_cheese_fluo.tsv')


@pytest.fixture(scope="module")
def cflnewdat(get_test_data):
    '''Read fluorescence spectra on cheese samples and return as numpy array'''
    return get_test_data('data_cheese_fluo_newRand.tsv')


@pytest.fixture(scope="module")
def csedat(get_test_data):
    '''Read sensory data on cheese samples and return as numpy array'''
    return get_test_data('data_cheese_sensory.tsv')


@pytest.fixture(scope="module")
def csecol2dat(get_test_data):
    '''Read sensory data (ATT02, i.e. one columns) on cheese samples and return as numpy array'''
    return get_test_data('data_cheese_sensory_col2.tsv', reshape=(-1, 1))
