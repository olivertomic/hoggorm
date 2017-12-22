'''
To be able to run test you have to install the hoggorm package.

You can either do a normal install
pip install hoggorm

or you can install in developer mode
pip install -e .
or
python setup.py develop
'''
import os.path as osp

import numpy as np

import pytest


@pytest.fixture(scope="session")
def datafolder():
    return osp.realpath(osp.dirname(__file__))


@pytest.fixture(scope="module")
def ldat(datafolder):
    '''Read liking data and return as numpy array'''
    return np.loadtxt(osp.join(datafolder, 'source_l_dat.tsv'), dtype=np.uint8, skiprows=1)


@pytest.fixture(scope="module")
def sdat(datafolder):
    '''Read sensory data and return as numpy array'''
    return np.loadtxt(osp.join(datafolder, 'source_s_dat.tsv'), dtype=np.float64, skiprows=1)
