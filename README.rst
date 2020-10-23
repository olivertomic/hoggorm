hoggorm
=======

.. image:: https://img.shields.io/pypi/l/hoggorm.svg
    :target: https://github.com/olivertomic/hoggorm/blob/master/LICENSE

.. image:: https://readthedocs.org/projects/hoggorm/badge/?version=latest
    :target: https://hoggorm.readthedocs.io/en/latest/?badge=latest

.. image:: http://joss.theoj.org/papers/10.21105/joss.00980/status.svg
   :target: https://doi.org/10.21105/joss.00980

.. image:: https://codecov.io/gh/andife/hoggorm/branch/hogCI/graph/badge.svg?token=IWQHXZQY4F
   :target: https://codecov.io/gh/andife/hoggorm/branch/hogCI

.. image:: https://github.com/mansenfranzen/hoggorm/workflows/ci-build/badge.svg?branch=ci_github_actions&event=push
   :target: https://github.com/mansenfranzen/hoggorm/actions?query=workflow%3Aci-build

.. image:: https://travis-ci.com/andife/hoggorm.svg?branch=hogCI
   :target: https://travis-ci.com/andife/hoggorm

.. image:: https://app.codacy.com/project/badge/Grade/16c4487ca1b945a28af18f44f04be0d5    
    :target: https://www.codacy.com/gh/andife/hoggorm/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=andife/hoggorm&amp;utm_campaign=Badge_Grade
   
.. image:: https://bestpractices.coreinfrastructure.org/projects/4359/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/4359
   
hoggorm is a Python package for explorative multivariate statistics in Python. It contains the following methods:

* PCA (principal component analysis)
* PCR (principal component regression)
* PLSR (partial least squares regression)
  
  - PLSR1 for single variable responses
  - PLSR2 for multivariate responses
* matrix correlation coefficients RV, RV2 and SMI.

Unlike `scikit-learn`_, which is an excellent python machine learning package focusing on classification, regression, clustering and predicition, hoggorm rather aims at understanding and interpretation of the variance in the data. hoggorm also contains tools for prediction.
The complementary package `hoggormplot`_ can be used for visualisation of results of models trained with hoggorm. 

.. _scikit-learn: https://scikit-learn.org/stable/
.. _hoggormplot: https://github.com/olivertomic/hoggormPlot

Examples
--------

.. |ColabCancer| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/olivertomic/hoggorm/blob/master/examples/PCA/PCA_on_cancer_data.ipynb
    :alt: Open in Colab

.. |BinderCancer| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/olivertomic/hoggorm/master?filepath=examples/PCA/PCA_on_cancer_data.ipynb
    :alt: Open in Binder

.. |BinderSensory| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/olivertomic/hoggorm/master?filepath=examples%2FPCR%2FPCR_on_sensory_and_fluorescence_data.ipynb
    :alt: Open in Binder

.. |ColabSensory| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/olivertomic/hoggorm/blob/master/examples/RV_%26_RV2/RV_and_RV2_on_sensory_and_fluorescence_data.ipynb
    :alt: Open In Colab

.. |ColabPCRCheese| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://github.com/olivertomic/hoggorm/blob/master/examples/PCR/PCR_on_sensory_and_fluorescence_data.ipynb
    :alt: Open In Colab

.. |ColabPLSR2Cheese| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://github.com/olivertomic/hoggorm/blob/master/examples/PLSR/PLSR_on_sensory_and_fluorescence_data.ipynb
    :alt: Open In Colab

Below are links to some Jupyter notebooks that illustrate how to use hoggorm and hoggormplot with the methods mentioned above. All examples are also found in the `examples`_ folder.

- Jupyter notebooks with examples of how to use hoggorm
  
  - for `PCA`_
		- `PCA on cancer data`_ on men in OECD countries |ColabCancer| |BinderCancer|
		- `PCA on NIR spectroscopy data`_ measured on gasoline
		- `PCA on sensory data`_ measured on cheese
  - for `PCR`_
		- `PCR on sensory and fluorescence spectroscopy data`_ measured on cheese |ColabPCRCheese|
  - for `PLSR1`_ for univariate response (one response variable)
    	- `PLSR1 on NIR spectroscopy and octane data`_ measured on gasoline
  - for `PLSR2`_ for multivariate response (multiple response variables)
    	- `PLSR2 on sensory and fluorescence spectroscopy data`_ measured on cheese |ColabPLSR2Cheese|
  - for matrix correlation coefficients `RV and RV2`_
		- `RV and RV2 coefficient on sensory and fluorescence spectroscopy data`_ measured on cheese |ColabSensory| |BinderSensory|
  - for the `SMI`_ (similarity of matrix index)
		- `SMI on sensory data and fluorescence data`_
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
.. _SMI on sensory data and fluorescence data: https://github.com/olivertomic/hoggorm/blob/master/examples/SMI/SMI_on_sensory_and_fluorescence.ipynb
.. _SMI on pseudo-random numbers: https://github.com/olivertomic/hoggorm/blob/master/examples/SMI/SMI_pseudo-random_numbers.ipynb




Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/download/

- numpy >= 1.9

Installation
------------

Using pip
*********

.. image:: https://pepy.tech/badge/hoggorm
    :target: https://pepy.tech/project/hoggorm
    :alt: PyPI Downloads

.. image:: https://pepy.tech/badge/hoggorm/month
    :target: https://pepy.tech/project/hoggorm/month
    :alt: PyPI Downloads

.. image:: https://pepy.tech/badge/hoggorm/week
    :target: https://pepy.tech/project/hoggorm/week
    :alt: PyPI Downloads

Install hoggorm easily from the command line from the `PyPI - the Python Packaging Index`_.

.. _PyPI - the Python Packaging Index: https://pypi.python.org/pypi

.. code-block:: bash

	pip install hoggorm

Using conda
***********

.. image:: https://img.shields.io/conda/dn/conda-forge/hoggorm.svg
    :target: https://anaconda.org/conda-forge/hoggorm
    :alt: Conda Downloads

.. image:: https://img.shields.io/conda/vn/conda-forge/hoggorm.svg
    :target: https://anaconda.org/conda-forge/hoggorm
    :alt: Conda Version
 
You can install using the conda package manager by running

.. code-block:: bash

    conda install -c conda-forge hoggorm


Documentation
-------------
.. image:: https://readthedocs.org/projects/hoggorm/badge/?version=latest

- Documentation at `Read the Docs`_
- Jupyter notebooks with `examples`_ of how to use Hoggorm together with the complementary plotting package `hoggormplot`_.
  
  
.. _Read the Docs: https://hoggorm.readthedocs.io/en/latest/
.. _examples: https://github.com/olivertomic/hoggorm/tree/master/examples
.. _hoggormplot: https://github.com/olivertomic/hoggormPlot


Citing hoggorm
--------------

If you use hoggorm in a report or scientific publication, we would appreciate citations to the following paper:

.. image:: https://joss.theoj.org/papers/10.21105/joss.00980/status.svg
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


