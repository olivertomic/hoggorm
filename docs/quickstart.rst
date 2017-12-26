Quickstart
==========

Hoggorm is a Python package for explorative multivariate statistics in Python. It contains 

* PCA (principal component analysis)
* PCR (principal component regression)
* PLSR (partial least squares regression)
  
  - PLSR1 for single variable responses
  - PLSR2 for multivariate responses
* matrix corrlation coefficients RV and RV2.

Unlike `scikit-learn`_, whis is an excellent Python machine learning package focusing on classification and predicition, Hoggorm rather aims at understanding and interpretation of the variance in the data. Hoggorm also contains tools for prediction.

.. _scikit-learn: http://scikit-learn.org/stable/

Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/download/

- numpy


Installation and updates
------------------------

Installation
++++++++++++

Install Hoggorm easily from the command line from the `PyPI - the Python Packaging Index`_. 

.. _PyPI - the Python Packaging Index: https://pypi.python.org/pypi

.. code-block:: bash

	pip install hoggorm

Upgrading
+++++++++

To upgrade Hoggorm from a previously installed older version execute the following from the command line:

.. code-block:: bash
        
        pip install --upgrade hoggorm


If you need more information on how to install Python packages using pip, please see the `pip documentation`_.

.. _pip documentation: https://pip.pypa.io/en/stable/#


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


Example
-------

.. code-block:: bash

	import hoggorm as ho
	
	# Compute PCA model with
	# - 5 components
	# - standardised/scaled variables
	# - KFold cross validation with 4 folds
	model = ho.nipalsPCA(arrX=myData, numComp=5, Xstand=True, cvType=["Kfold", 4])
	
	# Extract results from PCA model
	scores = model.X_scores()
	loadings = model.X_loadings()
	cumulativeCalibratedExplainedVariance_allVariables = model.X_cumCalExplVar_indVar()
	cumulativeValidatedExplainedVariance_total = model.X_cumValExplVar()




