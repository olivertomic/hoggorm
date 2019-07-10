Quickstart
==========

hoggorm is a Python package for explorative multivariate statistics in Python. It contains 

* PCA (principal component analysis)
* PCR (principal component regression)
* PLSR (partial least squares regression)
  
  - PLSR1 for univariate responses
  - PLSR2 for multivariate responses
* matrix correlation coefficients RV and RV2.

Unlike `scikit-learn`_, whis is an excellent Python machine learning package focusing on classification and predicition, hoggorm rather aims at understanding and interpretation of the variance in the data. hoggorm also contains tools for prediction.

.. _scikit-learn: http://scikit-learn.org/stable/

.. note:: Results computed with the hoggorm package can be visualised using plotting functions implemented in the complementary `hoggormplot`_ package.

.. _hoggormplot: http://hoggormplot.readthedocs.io/en/latest/index.html


Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/download/

- numpy >= 1.11.3


Installation and upgrades
-------------------------

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
		- `PCA on cancer data`_ on men in OECD countries
		- `PCA on NIR spectroscopy data`_ measured on gasoline	
		- `PCA on sensory data`_ measured on cheese
  - for `PCR`_
		- PCR on NIR spectroscopy and octane data measured on gasoline (coming soon)
		- `PCR on sensory and fluorescence spectroscopy data`_ measured on cheese
  - for `PLSR1`_ for univariate response (one response variable)
    	- `PLSR1 on NIR spectroscopy and octane data`_ measured on gasoline
  - for `PLSR2`_ for multivariate response (multiple response variables)
    	- `PLSR2 on sensory and fluorescence spectroscopy data`_ measured on cheese
  - for matrix correlation ceoefficitents `RV and RV2`_ 
		- `RV and RV2 coefficient on sensory and fluorescence spectroscopy data`_ measured on cheese
  

.. _Read the Docs: http://hoggorm.readthedocs.io/en/latest
.. _PCA: https://github.com/olivertomic/hoggorm/tree/master/examples/PCA
.. _PCR: https://github.com/olivertomic/hoggorm/tree/master/examples/PCR
.. _PLSR1: https://github.com/olivertomic/hoggorm/tree/master/examples/PLSR
.. _PLSR2: https://github.com/olivertomic/hoggorm/tree/master/examples/PLSR
.. _RV and RV2: https://github.com/olivertomic/hoggorm/tree/master/examples/RV_%26_RV2
.. _PCA on cancer data: https://github.com/olivertomic/hoggorm/blob/master/examples/PCA/PCA_on_cancer_data.ipynb
.. _PCA on NIR spectroscopy data: https://github.com/olivertomic/hoggorm/blob/master/examples/PCA/PCA_on_spectroscopy_data.ipynb
.. _PCA on sensory data: https://github.com/olivertomic/hoggorm/blob/master/examples/PCA/PCA_on_descriptive_sensory_analysis_data.ipynb
.. _PCR on sensory and fluorescence spectroscopy data: https://github.com/olivertomic/hoggorm/blob/master/examples/PCR/PCR_on_sensory_and_fluorescence_data.ipynb
.. _PLSR1 on NIR spectroscopy and octane data: https://github.com/olivertomic/hoggorm/blob/master/examples/PLSR/PLSR_on_NIR_and_octane_data.ipynb
.. _PLSR2 on sensory and fluorescence spectroscopy data: https://github.com/olivertomic/hoggorm/blob/master/examples/PLSR/PLSR_on_sensory_and_fluorescence_data.ipynb
.. _RV and RV2 coefficient on sensory and fluorescence spectroscopy data: https://github.com/olivertomic/hoggorm/blob/master/examples/RV_%26_RV2/RV_and_RV2_on_sensory_and_fluorescence_data.ipynb

More examples in Jupyter notebooks are provided at `hoggormExamples GitHub repository`_.

.. _hoggormExamples GitHub repository: https://github.com/khliland/hoggormExamples


Example
-------

.. code-block:: bash

	# Import hoggorm
	>>> import hoggorm as ho

	# Consumer liking data of 5 consumers stored in a numpy array
	>>> print(my_data)
	[[2 4 2 7 6]
     [4 7 4 3 6]
     [3 3 2 5 2]
     [5 9 6 4 4]
     [1 2 1 3 4]]
	
	# Compute PCA model with
	# - 3 components
	# - standardised/scaled variables (features or columns)
	# - Leave-one-out (LOO) cross validation
	>>> model = ho.nipalsPCA(arrX=my_data, numComp=3, Xstand=True, cvType=["loo"])
	
	# Extract results from PCA model
	# Get PCA scores
	>>> scores = model.X_scores()
	>>> print(scores)
	[[-0.97535198 -1.71827581  0.43672952]
	 [ 1.28340424 -0.24453505 -0.98250731]
	 [-0.9127492   0.97132275  1.04708189]
	 [ 2.34954599  0.30633998  0.43178679]
	 [-1.74484905  0.68514813 -0.93309089]]
	
	# Get PCA loadings
	>>> loadings = model.X_loadings()
	>>> print(loadings)
	[[ 0.55080115  0.10025801  0.25045298]
	 [ 0.57184198 -0.11712858  0.00316316]
	 [ 0.57141459  0.00568809  0.10503941]
	 [-0.1682551  -0.61149788  0.77153937]
	 [ 0.12161589 -0.77605877 -0.57528864]]
	
	# Get cumulative explained variance for each variable
	>>> cumCalExplVar_allVariables = model.X_cumCalExplVar_indVar()
	>>> print(cumCalExplVar_allVariables)
	[[ 0.          0.          0.          0.          0.        ]
	 [90.98654597 98.07234952 97.92497156  8.48956314  4.43690992]
	 [92.12195756 99.62227118 97.92862256 50.73769558 72.47502242]
	 [97.31181824 99.62309922 98.84150821 99.98958248 99.85786661]]
	
	# Get cumulative explained variance for all variables
	>>> cumCalExplVar_total = model.X_cumValExplVar()
	>>> print(cumCalExplVar_total)
	[0.0, 35.43333631454735, 32.12929746015379, 71.32495809880507]

hoggorm repository on GitHub
----------------------------
The source code is available at the `hoggorm GitHub repository`_.

.. _hoggorm GitHub repository: https://github.com/olivertomic/hoggorm


Testing
-------
The correctness of the results provided PCA, PCR and PLSR may be checked using the tests provided in the `tests`_ folder.

.. _tests: https://github.com/olivertomic/hoggorm/tree/master/tests


After cloning the repository to your disk, at the command line navigate to the test folder. The code below shows an example of how to run the test for PCA.

.. code-block:: bash
        
        python test_pca.py 

After testing is finished, pytest should report that none of tests failed.


