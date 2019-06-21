#!/usr/bin/env python
# coding: utf-8

# # RV and RV2 coefficient on Sensory and Fluorescence data

# This notebook illustrates how to use the **hoggorm** package to carry out partial least squares regression (PLSR) on multivariate data. Furthermore, we will learn how to visualise the results of the PLSR using the **hoggormPlot** package.

# ---

# ### Import packages and prepare data

# First import **hoggorm** for analysis of the data and **hoggormPlot** for plotting of the analysis results. We'll also import **pandas** such that we can read the data into a data frame. **numpy** is needed for checking dimensions of the data.

# In[2]:


import hoggorm as ho
import hoggormplot as hop
import pandas as pd
import numpy as np


# Next, load the data that we are going to analyse using **hoggorm**. After the data has been loaded into the pandas data frame, we'll display it in the notebook.

# In[3]:


# Load fluorescence data
X_df = pd.read_csv('cheese_fluorescence.txt', index_col=0, sep='\t')
X_df


# In[4]:


# Load sensory data
Y_df = pd.read_csv('cheese_sensory.txt', index_col=0, sep='\t')
Y_df


# The ``RVcoeff`` and ``RV2coeff`` methods in hoggorm accept only **numpy** arrays with numerical values and not pandas data frames. Therefore, the pandas data frames holding the imported data need to be "taken apart" into three parts: 
# * two numpy array holding the numeric values
# * two Python list holding variable (column) names
# * two Python list holding object (row) names. 

# In[5]:


# Get the values from the data frame
X = X_df.values
Y = Y_df.values

# Get the variable or columns names
X_varNames = list(X_df.columns)
Y_varNames = list(Y_df.columns)

# Get the object or row names
X_objNames = list(X_df.index)
Y_objNames = list(Y_df.index)


# ---

# ### Apply RV and RV2 to our data

# Now, let's apply the RV and RV2 matrix correlation coefficient methods on the data [description of the input parameters](https://hoggorm.readthedocs.io/en/latest/matrix_corr_coeff.html). The functions take python lists as input which may contain two or more arrays measured on the same objects and compute RV and RV2 matrix correlation coefficients between pairs of arrays. The number and order of objects (rows) for the two arrays must match. The number of variables in each array may vary. The RV coefficient results in values 0 <= RV <= 1. The RV2 coefficient is a modified version of the RV coefficient with values -1 <= RV2 <= 1. RV2 is independent of object and variable size.

# ### Preprocessing the data

# Arrays need to be preprocessed before computing RV and RV2. More precisely, the arrays need to be either centred or standardised/scaled.

# In[9]:


# Center data first
X_cent = ho.center(X_df.values, axis=0)
Y_cent = ho.center(Y_df.values, axis=0)


# In[8]:


X_cent


# In[10]:


Y_cent


# After both arrays were centered, we store them in a list and submit them to the RV or RV2 matrix correlation coefficient function, as described below. Note that the list can contain two or more arrays. The function then returns an array holding RV coefficient for all pair-wise combinations of arrays.

# In[19]:


rv_results_cent = ho.RVcoeff([X_cent, Y_cent])


# In[23]:


rv_results_cent


# The RV computation results are stored in a new array as seen above. At the diagonal the RV is 1, since the we compute $RV(X_{cent}, X_{cent}) = 1$ and $RV(Y_{cent}, Y_{cent}) = 1$, in each case indicating that the information across the two matrices is identical. Correspondingly, $RV(X_{cent}, Y_{cent}) = 0.24142324$ at index ``[0, 1]`` and $RV(Y_{cent}, X_{cent}) = 0.24142324$ at index ``[1, 0]``.

# Now the corresponding computation using the RV2 coefficient.

# In[24]:


rv2_results_cent = ho.RV2coeff([X_cent, Y_cent])


# In[25]:


rv2_results_cent


# Do the same computations, however with standardised arrays where each feature has the same weight.

# In[30]:


# Standardise data first
X_stand = ho.standardise(X_df.values, mode=0)
Y_stand = ho.standardise(Y_df.values, mode=0)


# In[26]:


rv_results_stand = ho.RVcoeff([X_stand, Y_stand])


# In[27]:


rv_results_stand


# In[28]:


rv2_results_stand = ho.RV2coeff([X_stand, Y_stand])


# In[29]:


rv2_results_stand

