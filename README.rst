Hoggorm
=======

Hoggorm is a Python package for explorative multivariate statistics in Python. It contains 

* PCA (principal component analysis)
* PCR (principal component regression)
* PLSR (partial least squares regression)
  
  - PLSR1 for single variable responses
  - PLSR2 for multivariate responses
* matrix corrlation coefficients RV and RV2.

Unlike `scikit-learn`_, whis is an excellent python machine learning package focusing on classification and predicition, Hoggorm rather aims at understanding and interpretation of the variance in the data. Hoggorm also also contains tools for prediction.

.. _scikit-learn: http://scikit-learn.org/stable/

Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/download/

- numpy


Installation
------------

Install Hoggorm easily from the command line from the `PyPI - the Python Packaging Index`_. 

.. _PyPI - the Python Packaging Index: https://pypi.python.org/pypi

.. code-block:: bash

	pip install hoggorm


Documentation
-------------

- Documentation at `Read the Docs`_
- Jupyter notebooks with examples of how to use Hoggorm
  
  - for `PCA`_
  - for PCR (coming soon)
  - for PLSR1 (coming soon)
  - for PLSR2 (coming soon)
  - for matrix correlation ceoefficitents RV and RV2 (coming soon)
  

.. _Read the Docs: http://hoggorm.readthedocs.io/en/latest
.. _PCA: https://github.com/olivertomic/hoggorm/blob/master/docs/PCA%20with%20hoggorm.ipynb
