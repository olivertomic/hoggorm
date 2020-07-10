hoggorm
=======

.. image:: https://pepy.tech/badge/hoggorm
    :target: https://pepy.tech/project/hoggorm
    :alt: PyPI Downloads

.. image:: https://pepy.tech/badge/hoggorm/month
    :target: https://pepy.tech/project/hoggorm/month
    :alt: PyPI Downloads
    
.. image:: https://pepy.tech/badge/hoggorm/week
    :target: https://pepy.tech/project/hoggorm/week
    :alt: PyPI Downloads

.. image:: https://readthedocs.org/projects/hoggorm/badge/?version=latest
    :target: https://hoggorm.readthedocs.io/en/latest/?badge=latest

.. image:: http://joss.theoj.org/papers/10.21105/joss.00980/status.svg
   :target: https://doi.org/10.21105/joss.00980

hoggorm is a Python package for explorative multivariate statistics in Python. It contains the following methods: 

* PCA (principal component analysis)
* PCR (principal component regression)
* PLSR (partial least squares regression)
  
  - PLSR1 for single variable responses
  - PLSR2 for multivariate responses
* matrix correlation coefficients RV, RV2 and SMI.

Unlike `scikit-learn`_, which is an excellent python machine learning package focusing on classification, regression, clustering and predicition, hoggorm rather aims at understanding and interpretation of the variance in the data. hoggorm also contains tools for prediction.
The complementary package `hoggormplot`_ can be used for visualisation of results of models trained with hoggorm. 

.. _scikit-learn: http://scikit-learn.org/stable/
.. _hoggormplot: https://github.com/olivertomic/hoggormPlot

Examples
--------
Below are links to some Jupyter notebooks that illustrate how to use hoggorm and hoggormplot with the methods mentioned above. All examples are also found in the `examples`_ folder.

- Jupyter notebooks with examples of how to use hoggorm
  
  - for `PCA`_
		- `PCA on cancer data`_ on men in OECD countries
		- `PCA on NIR spectroscopy data`_ measured on gasoline	
		- `PCA on sensory data`_ measured on cheese
  - for `PCR`_
		- `PCR on sensory and fluorescence spectroscopy data`_ measured on cheese
  - for `PLSR1`_ for univariate response (one response variable)
    	- `PLSR1 on NIR spectroscopy and octane data`_ measured on gasoline
  - for `PLSR2`_ for multivariate response (multiple response variables)
    	- `PLSR2 on sensory and fluorescence spectroscopy data`_ measured on cheese
  - for matrix correlation ceoefficitents `RV and RV2`_ 
		- `RV and RV2 coefficient on sensory and fluorescence spectroscopy data`_ measured on cheese
  - for the `SMI`_ (similarity of matrix index)
		- `SMI on sensory data and fluorescense data`_
		- `SMI on pseudo-random numbers`_
  
.. _examples: https://github.com/olivertomic/hoggorm/tree/master/examples
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
.. _SMI: https://github.com/olivertomic/hoggorm/tree/master/examples/SMI
.. _SMI on sensory data and fluorescense data: https://github.com/olivertomic/hoggorm/blob/master/examples/SMI/SMI_on_sensory_and_fluorescence.ipynb
.. _SMI on pseudo-random numbers: https://github.com/olivertomic/hoggorm/blob/master/examples/SMI/SMI_pseudo-random_numbers.ipynb

Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/download/

- numpy >= 1.11.3

Installation
------------

Install hoggorm easily from the command line from the `PyPI - the Python Packaging Index`_. 

.. _PyPI - the Python Packaging Index: https://pypi.python.org/pypi

.. code-block:: bash

	pip install hoggorm


Documentation
-------------
.. image:: https://readthedocs.org/projects/hoggorm/badge/?version=latest

- Documentation at `Read the Docs`_
- Jupyter notebooks with `examples`_ of how to use Hoggorm together with the complementary plotting package `hoggormplot`_.
  
  
.. _Read the Docs: http://hoggorm.readthedocs.io/en/latest
.. _examples: https://github.com/olivertomic/hoggorm/tree/master/examples
.. _hoggormplot: https://github.com/olivertomic/hoggormPlot


Citing hoggorm
--------------

If you use hoggorm in a report or scientific publication, we would appreciate citations to the following paper:

.. image:: http://joss.theoj.org/papers/10.21105/joss.00980/status.svg
   :target: https://doi.org/10.21105/joss.00980

Tomic et al., (2019). hoggorm: a python library for explorative multivariate statistics. Journal of Open Source Software, 4(39), 980, https://doi.org/10.21105/joss.00980 

Bibtex entry:

.. code-block:: bash

    @article{hoggorm,
      title={hoggorm: a python library for explorative multivariate statistics},
      author={Tomic, Oliver and Graff, Thomas and Liland, Kristian Hovde and N{\ae}s, Tormod},
      journal={The Journal of Open Source Software},
      volume={4},
      number={39},
      year={2019}
      doi={10.21105/joss.00980},
      url={http://joss.theoj.org/papers/10.21105/joss.00980}
    }


