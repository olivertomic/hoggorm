
Utility classes and functions
==============================

There are number of functions and classes that might be useful for working with data outside the hoggorm package. They are provided here
for convenience.

Functions in hoggorm.statTools module
-------------------------------------

The hoggorm.statTools module provides some functions that can be useful when working with multivariate data sets. 

.. automodule:: hoggorm.statTools
   :members:

Cross validation classes in hoggorm.cross_val module
----------------------------------------------------

hoggorm classes PCA, PLSR and PCR use a number classes for computation of the models which are found in the hoggorm.cross_val module.

The cross validation classes in this module are used inside the multivariate statistical methods and may be called upon using the ``cvType`` 
input parameter for these methods. They are not intended to be used outside the multivariate statistical methods, even though it is possible. 
They are shown here to illustrate how the different cross validation options work. 

.. automodule:: hoggorm.cross_val
   :members:
