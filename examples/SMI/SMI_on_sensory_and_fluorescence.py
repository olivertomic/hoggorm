#!/usr/bin/env python
# coding: utf-8

# # SMI - Similarity of Matrices Index
# SMI is a measure of the similarity between the dominant subspaces of two matrices. It comes in two flavours (projections): 
# - OP - Orthogonal Projections
# - PR - Procrustes Rotations.  
# 
# The former (default) compares subspaces using ordinary least squares and can be formulated as the explained variance  when predicting one matrix subspace using the other matrix subspace. PR is a restriction where only rotation and scaling is allowed in the similarity calculations.  
#   
# Subspaces are by default computed using Principal Component Analysis (PCA). When the number of components extracted from one of the matrices is smaller than the other, the explained variance is calculated predicting the smaller subspace by using the larger subspace.

# ## Example: Sensory and Fluorescence data
# ---
# ### Import packages and prepare data

# First import **hoggorm** for analysis of the data and **hoggormPlot** for plotting of the analysis results. We'll also import **pandas** such that we can read the data into a data frame. **numpy** is needed for checking dimensions of the data.

# In[18]:


import hoggorm as ho
import hoggormplot as hop
import pandas as pd
import numpy as np


# Next, load the data that we are going to analyse using **hoggorm**. After the data has been loaded into the pandas data frame, we'll display it in the notebook.

# In[19]:


# Load fluorescence data
X1_df = pd.read_csv('cheese_fluorescence.txt', index_col=0, sep='\t')
X1_df


# In[20]:


# Load sensory data
X2_df = pd.read_csv('cheese_sensory.txt', index_col=0, sep='\t')
X2_df


# ### Orthogonal Projections
# The default comparison between two matrices with SMI is using Orthogonal Projections, i.e. ordinary least squares regression is used to relate the dominant subspaces in the two matrices.
#   
# In contrast to PLSR, SMI is not performing av prediction of sensory properties from fluorescence measurements, but rather treats the two sets of measurements symmetrically, focusing on the major variation in each of them.

# More details regarding the use of the SMI are found in the [documentation](https://hoggorm.readthedocs.io/en/latest/matrix_corr_coeff.html).

# In[21]:


# Get the values from the data frame
X1 = X1_df.values
X2 = X2_df.values

smiOP = ho.SMI(X1, X2, ncomp1=10, ncomp2=10)
print(np.round(smiOP.smi, 2))


# A hypothesis can be made regarding the similarity of two subspaces where the null hypothesis is that they are equal and the alternative is that they are not. Permutation testing yields the following P-values (probabilities that the observed difference could be larger given the null hypothesis is true).

# In[22]:


print(np.round(smiOP.significance(), 2))


# Finally we visualize the SMI values and their corresponding P-values.

# In[23]:


# Plot similarities
hop.plotSMI(smiOP, [10, 10], X1name='fluorescence', X2name='sensory')


# The significance symbols in the diamond plot above indicate if a chosen subspace from one matrix can be found inside the subspace from the other matrix ($\supset$, $\subset$, =), or if there is signficant difference (P-values <0.001\*\*\* <0.01 \*\* <0.05 \* <0.1 . >=0.1).  
# 
# From the P-values and plot we can observe that the there is a significant difference between the sensory data and the fluorescence data in the first of the dominant subspaces of the matrices. Looking only at the diagonal, we see that 6 components are needed before we loose the significance completely. Looking at the one-dimensional subspaces, we can observe that four sensory components are needed before there is no significant difference to the first fluorescence component.
#   
# This can be interpreted as some fundamental difference in the information spanned by flurescence measurements and sensory perceptions that is only masked if large proportions of the subspaces are included.

# ### Procrustes Rotations
# The similarities using PR <= OP, and in this simple case OP$^2$ = PR. Otherwise the pattern stays the same.

# In[4]:


smiPR = ho.SMI(X1, X2, ncomp1=10, ncomp2=10, projection="Procrustes")
print(np.round(smiPR.smi, 2))


# The number of permutations can be controlled for quick (100) or accurate (>10000) computations of significance.

# In[14]:


print(np.round(smiPR.significance(B = 100),2))


# In[17]:


hop.plotSMI(smiPR, X1name='fluorescence', X2name='sensory')


# The SMI values in the Procrustes Rotations case are mostly very similar to the Orthogonal Projections case. This means that the differences between the two matrices can be attributed to rotation and scaling to a large degree. With a few execpetions, we therefore see the same patterns in the significances too.

# 

# _Reference:_   
# Ulf Geir Indahl, Kristian Hovde Liland, Tormod Næs,  
# [A similarity index for comparing coupled matrices](https://www.onlinelibrary.wiley.com/doi/10.1002/cem.3049),
# Journal of Chemometrics 32(e3049), (2018).

# In[ ]:




