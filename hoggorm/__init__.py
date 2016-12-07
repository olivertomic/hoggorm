# -*- coding: utf-8 -*-

""" Hoggorm is a Python package for explorative multivariate statistics in Python.

It contains PCA (principal component analysis), PCR (principal component regression), PLSR (partial least squares regression) and the matrix corrlation coefficients RV and RV2.

"""

# Import built-in modules first, followed by third-party modules,
# followed by any changes to the path and your own modules.

from .version import __version__

from .statTools import (RVcoeff, RV2coeff, ortho, centre, standardise, matrixRank)
from .pca import nipalsPCA
from .pcr import nipalsPCR
from .plsr import (nipalsPLS1, nipalsPLS2)
