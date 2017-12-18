
Utililty functions
==================

Hoggorm classes PCA, PLSR and PCR use a number of functions and classes for computation of the models. These are found in the hoggorm.statTools module and hoggorm.cross_val module. 

Functions in hoggorm.statTools module
-------------------------------------
.. automodule:: hoggorm.statTools
   :members:

Cross validation classes in hoggorm.cross_val module
----------------------------------------------------

The cross validation classes in this module are used inside the multivariate statistical methods and may be called upon using the ``cvType`` 
input parameter for these methods. They are not intended to be used outside the multivariate statistical methods, even though it is possible. 
They are shown here to illustrate how the different cross validation options work. 

.. automodule:: hoggorm.cross_val
   :members:
