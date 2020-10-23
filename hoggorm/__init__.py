# -*- coding: utf-8 -*-

""" Hoggorm is a Python package for explorative multivariate statistics in Python.

It contains PCA (principal component analysis), PCR (principal component regression), PLSR (partial least squares regression) and the matrix corrlation coefficients RV and RV2.

"""

# Import built-in modules first, followed by third-party modules,
# followed by any changes to the path and your own modules.

from .statTools import ortho, center, standardise, matrixRank
from .mat_corr_coeff import RVcoeff, RV2coeff, SMI
from .pca import nipalsPCA
from .pcr import nipalsPCR
from .plsr1 import nipalsPLS1
from .plsr2 import nipalsPLS2

__version__ = "0.13.3"
