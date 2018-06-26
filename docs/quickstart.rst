Quickstart
==========

hoggorm is a Python package for explorative multivariate statistics in Python. It contains 

* PCA (principal component analysis)
* PCR (principal component regression)
* PLSR (partial least squares regression)
  
  - PLSR1 for single variable responses
  - PLSR2 for multivariate responses
* matrix corrlation coefficients RV and RV2.

Unlike `scikit-learn`_, whis is an excellent Python machine learning package focusing on classification and predicition, hoggorm rather aims at understanding and interpretation of the variance in the data. hoggorm also contains tools for prediction.

.. _scikit-learn: http://scikit-learn.org/stable/

.. note:: Results computed with the hoggorm package can be visualised using plotting functions implemented in the complementary `hoggormplot`_ package.

.. _hoggormplot: http://hoggormplot.readthedocs.io/en/latest/index.html


Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/download/

- numpy


Installation and updates
------------------------

Installation
++++++++++++

Install hoggorm easily from the command line from the `PyPI - the Python Packaging Index`_. 

.. _PyPI - the Python Packaging Index: https://pypi.python.org/pypi

.. code-block:: bash

	pip install hoggorm

Upgrading
+++++++++

To upgrade hoggorm from a previously installed older version execute the following from the command line:

.. code-block:: bash
        
        pip install --upgrade hoggorm


If you need more information on how to install Python packages using pip, please see the `pip documentation`_.

.. _pip documentation: https://pip.pypa.io/en/stable/#


Documentation
-------------

- Documentation at `Read the Docs`_
- Jupyter notebooks with examples of how to use hoggorm
  
  - for `PCA`_
  - for PCR (coming soon)
  - for PLSR1 (coming soon)
  - for PLSR2 (coming soon)
  - for matrix correlation ceoefficitents RV and RV2 (coming soon)
  

.. _Read the Docs: http://hoggorm.readthedocs.io/en/latest
.. _PCA: https://github.com/olivertomic/hoggorm/blob/master/docs/PCA%20with%20hoggorm.ipynb

More examples in Jupyter notebooks are provided at `hoggormExamples GitHub repository`_.

.. _hoggormExamples GitHub repository: https://github.com/khliland/hoggormExamples


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


hoggorm repository on GitHub
----------------------------
The source code is available at the `hoggorm GitHub repository`_.

.. _hoggorm GitHub repository: https://github.com/olivertomic/hoggorm









