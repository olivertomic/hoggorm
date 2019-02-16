
# coding: utf-8

# # Principal Component Regression (PCR) on Sensory and Fluorescence data

# This notebook illustrates how to use the **hoggorm** package to carry out principal component regression (PCR) on multivariate data. Furthermore, we will learn how to visualise the results of the PCR using the **hoggormPlot** package.

# ---

# ### Import packages and prepare data

# First import **hoggorm** for analysis of the data and **hoggormPlot** for plotting of the analysis results. We'll also import **pandas** such that we can read the data into a data frame. **numpy** is needed for checking dimensions of the data.

# In[1]:


import hoggorm as ho
import hoggormplot as hop
import pandas as pd
import numpy as np


# Next, load the data that we are going to analyse using **hoggorm**. After the data has been loaded into the pandas data frame, we'll display it in the notebook.

# In[2]:


# Load fluorescence data
X_df = pd.read_csv('cheese_fluorescence.txt', index_col=0, sep='\t')
X_df


# In[3]:


# Load sensory data
Y_df = pd.read_csv('cheese_sensory.txt', index_col=0, sep='\t')
Y_df


# The ``nipalsPCR`` class in hoggorm accepts only **numpy** arrays with numerical values and not pandas data frames. Therefore, the pandas data frames holding the imported data need to be "taken apart" into three parts: 
# * two numpy array holding the numeric values
# * two Python list holding variable (column) names
# * two Python list holding object (row) names. 
# 
# The numpy arrays with values will be used as input for the ``nipalsPCR`` class for analysis. The Python lists holding the variable and row names will be used later in the plotting function from the **hoggormPlot** package when visualising the results of the analysis. Below is the code needed to access both data, variable names and object names.

# In[4]:


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

# ### Apply PCR to our data

# Now, let's run PCR on the data using the ``nipalsPCR`` class. The documentation provides a [description of the input parameters](https://hoggorm.readthedocs.io/en/latest/pcr.html). Using input paramter ``arrX`` and ``arrY`` we define which numpy array we would like to analyse. ``arrY`` is what typically is considered to be the response matrix, while the measurements are typically defined as ``arrX``. By setting input parameter ``Xstand=False`` and ``Ystand=False`` we make sure that the variables are only mean centered, not scaled to unit variance, if this is what you want. This is the default setting and actually doesn't need to expressed explicitly. Setting paramter ``cvType=["loo"]`` we make sure that we compute the PCR model using full cross validation. ``"loo"`` means "Leave One Out". By setting paramter ``numpComp=4`` we ask for four components to be computed.

# In[5]:


model = ho.nipalsPCR(arrX=X, Xstand=False, 
                     arrY=Y, Ystand=False,
                     cvType=["loo"], 
                     numComp=4)


# That's it, the PCR model has been computed. Now we would like to inspect the results by visualising them. We can do this using plotting functions of the separate [**hoggormPlot** package](https://hoggormplot.readthedocs.io/en/latest/). If we wish to plot the results for component 1 and component 2, we can do this by setting the input argument ``comp=[1, 2]``. The input argument ``plots=[1, 2, 3, 4, 6]`` lets the user define which plots are to be plotted. If this list for example contains value ``1``, the function will generate the scores plot for the model. If the list contains value ``2``, then the loadings plot will be plotted. Value ``3`` stands for correlation loadings plot and value ``4`` stands for bi-plot and ``6`` stands for explained variance plot. The hoggormPlot documentation provides a [description of input paramters](https://hoggormplot.readthedocs.io/en/latest/mainPlot.html).

# In[6]:


hop.plot(model, comp=[1, 2], 
         plots=[1, 2, 3, 4, 6], 
         objNames=X_objNames, 
         XvarNames=X_varNames,
         YvarNames=Y_varNames)


# Plots can also be called separately.

# In[7]:


# Plot cumulative explained variance (both calibrated and validated) using a specific function for that.
hop.explainedVariance(model)


# In[8]:


# Plot cumulative validated explained variance for each variable in Y
hop.explainedVariance(model, individual = True)


# In[9]:


# Plot cumulative validated explained variance in X.
hop.explainedVariance(model, which='X')


# In[10]:


hop.scores(model)


# In[11]:


hop.correlationLoadings(model)


# In[12]:


# Plot regression coefficients
hop.coefficients(model, comp=3)


# ---

# ### Accessing numerical results

# Now that we have visualised the PCR results, we may also want to access the numerical results. Below are some examples. For a complete list of accessible results, please see this part of the documentation.  

# In[13]:


# Get X scores and store in numpy array
X_scores = model.X_scores()

# Get scores and store in pandas dataframe with row and column names
X_scores_df = pd.DataFrame(model.X_scores())
X_scores_df.index = X_objNames
X_scores_df.columns = ['PC{0}'.format(x+1) for x in range(model.X_scores().shape[1])]
X_scores_df


# In[14]:


help(ho.nipalsPLS2.X_scores)


# In[15]:


# Dimension of the X_scores
np.shape(model.X_scores())


# We see that the numpy array holds the scores for all countries and OECD (35 in total) for four components as required when computing the PCA model.

# In[16]:


# Get X loadings and store in numpy array
X_loadings = model.X_loadings()

# Get X loadings and store in pandas dataframe with row and column names
X_loadings_df = pd.DataFrame(model.X_loadings())
X_loadings_df.index = X_varNames
X_loadings_df.columns = ['PC{0}'.format(x+1) for x in range(model.X_loadings().shape[1])]
X_loadings_df


# In[17]:


help(ho.nipalsPLS2.X_loadings)


# In[18]:


np.shape(model.X_loadings())


# Here we see that the array holds the loadings for the 10 variables in the data across four components.

# In[19]:


# Get Y loadings and store in numpy array
Y_loadings = model.Y_loadings()

# Get Y loadings and store in pandas dataframe with row and column names
Y_loadings_df = pd.DataFrame(model.Y_loadings())
Y_loadings_df.index = Y_varNames
Y_loadings_df.columns = ['PC{0}'.format(x+1) for x in range(model.Y_loadings().shape[1])]
Y_loadings_df


# In[20]:


# Get X correlation loadings and store in numpy array
X_corrloadings = model.X_corrLoadings()

# Get X correlation loadings and store in pandas dataframe with row and column names
X_corrloadings_df = pd.DataFrame(model.X_corrLoadings())
X_corrloadings_df.index = X_varNames
X_corrloadings_df.columns = ['PC{0}'.format(x+1) for x in range(model.X_corrLoadings().shape[1])]
X_corrloadings_df


# In[21]:


help(ho.nipalsPLS2.X_corrLoadings)


# In[22]:


# Get Y loadings and store in numpy array
Y_corrloadings = model.X_corrLoadings()

# Get Y loadings and store in pandas dataframe with row and column names
Y_corrloadings_df = pd.DataFrame(model.Y_corrLoadings())
Y_corrloadings_df.index = Y_varNames
Y_corrloadings_df.columns = ['PC{0}'.format(x+1) for x in range(model.Y_corrLoadings().shape[1])]
Y_corrloadings_df


# In[23]:


help(ho.nipalsPLS2.Y_corrLoadings)


# In[24]:


# Get calibrated explained variance of each component in X
X_calExplVar = model.X_calExplVar()

# Get calibrated explained variance in X and store in pandas dataframe with row and column names
X_calExplVar_df = pd.DataFrame(model.X_calExplVar())
X_calExplVar_df.columns = ['calibrated explained variance in X']
X_calExplVar_df.index = ['PC{0}'.format(x+1) for x in range(model.X_loadings().shape[1])]
X_calExplVar_df


# In[25]:


help(ho.nipalsPLS2.X_calExplVar)


# In[26]:


# Get calibrated explained variance of each component in Y
Y_calExplVar = model.Y_calExplVar()

# Get calibrated explained variance in Y and store in pandas dataframe with row and column names
Y_calExplVar_df = pd.DataFrame(model.Y_calExplVar())
Y_calExplVar_df.columns = ['calibrated explained variance in Y']
Y_calExplVar_df.index = ['PC{0}'.format(x+1) for x in range(model.Y_loadings().shape[1])]
Y_calExplVar_df


# In[27]:


help(ho.nipalsPLS2.Y_calExplVar)


# In[28]:


# Get cumulative calibrated explained variance in X
X_cumCalExplVar = model.X_cumCalExplVar()

# Get cumulative calibrated explained variance in X and store in pandas dataframe with row and column names
X_cumCalExplVar_df = pd.DataFrame(model.X_cumCalExplVar())
X_cumCalExplVar_df.columns = ['cumulative calibrated explained variance in X']
X_cumCalExplVar_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
X_cumCalExplVar_df


# In[29]:


help(ho.nipalsPLS2.X_cumCalExplVar)


# In[30]:


# Get cumulative calibrated explained variance in Y
Y_cumCalExplVar = model.Y_cumCalExplVar()

# Get cumulative calibrated explained variance in Y and store in pandas dataframe with row and column names
Y_cumCalExplVar_df = pd.DataFrame(model.Y_cumCalExplVar())
Y_cumCalExplVar_df.columns = ['cumulative calibrated explained variance in Y']
Y_cumCalExplVar_df.index = ['PC{0}'.format(x) for x in range(model.Y_loadings().shape[1] + 1)]
Y_cumCalExplVar_df


# In[31]:


help(ho.nipalsPLS2.Y_cumCalExplVar)


# In[32]:


# Get cumulative calibrated explained variance for each variable in X
X_cumCalExplVar_ind = model.X_cumCalExplVar_indVar()

# Get cumulative calibrated explained variance for each variable in X and store in pandas dataframe with row and column names
X_cumCalExplVar_ind_df = pd.DataFrame(model.X_cumCalExplVar_indVar())
X_cumCalExplVar_ind_df.columns = X_varNames
X_cumCalExplVar_ind_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
X_cumCalExplVar_ind_df


# In[33]:


help(ho.nipalsPLS2.X_cumCalExplVar_indVar)


# In[34]:


# Get cumulative calibrated explained variance for each variable in Y
Y_cumCalExplVar_ind = model.Y_cumCalExplVar_indVar()

# Get cumulative calibrated explained variance for each variable in Y and store in pandas dataframe with row and column names
Y_cumCalExplVar_ind_df = pd.DataFrame(model.Y_cumCalExplVar_indVar())
Y_cumCalExplVar_ind_df.columns = Y_varNames
Y_cumCalExplVar_ind_df.index = ['PC{0}'.format(x) for x in range(model.Y_loadings().shape[1] + 1)]
Y_cumCalExplVar_ind_df


# In[35]:


help(ho.nipalsPLS2.Y_cumCalExplVar_indVar)


# In[36]:


# Get calibrated predicted Y for a given number of components

# Predicted Y from calibration using 1 component
Y_from_1_component = model.Y_predCal()[1]

# Predicted Y from calibration using 1 component stored in pandas data frame with row and columns names
Y_from_1_component_df = pd.DataFrame(model.Y_predCal()[1])
Y_from_1_component_df.index = Y_objNames
Y_from_1_component_df.columns = Y_varNames
Y_from_1_component_df


# In[37]:


# Get calibrated predicted Y for a given number of components

# Predicted Y from calibration using 4 component
Y_from_4_component = model.Y_predCal()[4]

# Predicted Y from calibration using 1 component stored in pandas data frame with row and columns names
Y_from_4_component_df = pd.DataFrame(model.Y_predCal()[4])
Y_from_4_component_df.index = Y_objNames
Y_from_4_component_df.columns = Y_varNames
Y_from_4_component_df


# In[38]:


help(ho.nipalsPLS2.X_predCal)


# In[39]:


# Get validated explained variance of each component X
X_valExplVar = model.X_valExplVar()

# Get calibrated explained variance in X and store in pandas dataframe with row and column names
X_valExplVar_df = pd.DataFrame(model.X_valExplVar())
X_valExplVar_df.columns = ['validated explained variance in X']
X_valExplVar_df.index = ['PC{0}'.format(x+1) for x in range(model.X_loadings().shape[1])]
X_valExplVar_df


# In[40]:


help(ho.nipalsPLS2.X_valExplVar)


# In[41]:


# Get validated explained variance of each component Y
Y_valExplVar = model.Y_valExplVar()

# Get calibrated explained variance in X and store in pandas dataframe with row and column names
Y_valExplVar_df = pd.DataFrame(model.Y_valExplVar())
Y_valExplVar_df.columns = ['validated explained variance in Y']
Y_valExplVar_df.index = ['PC{0}'.format(x+1) for x in range(model.Y_loadings().shape[1])]
Y_valExplVar_df


# In[42]:


help(ho.nipalsPLS2.Y_valExplVar)


# In[43]:


# Get cumulative validated explained variance in X
X_cumValExplVar = model.X_cumValExplVar()

# Get cumulative validated explained variance in X and store in pandas dataframe with row and column names
X_cumValExplVar_df = pd.DataFrame(model.X_cumValExplVar())
X_cumValExplVar_df.columns = ['cumulative validated explained variance in X']
X_cumValExplVar_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
X_cumValExplVar_df


# In[44]:


help(ho.nipalsPLS2.X_cumValExplVar)


# In[45]:


# Get cumulative validated explained variance in Y
Y_cumValExplVar = model.Y_cumValExplVar()

# Get cumulative validated explained variance in Y and store in pandas dataframe with row and column names
Y_cumValExplVar_df = pd.DataFrame(model.Y_cumValExplVar())
Y_cumValExplVar_df.columns = ['cumulative validated explained variance in Y']
Y_cumValExplVar_df.index = ['PC{0}'.format(x) for x in range(model.Y_loadings().shape[1] + 1)]
Y_cumValExplVar_df


# In[46]:


help(ho.nipalsPLS2.Y_cumValExplVar)


# In[47]:


# Get cumulative validated explained variance for each variable in Y
Y_cumCalExplVar_ind = model.Y_cumCalExplVar_indVar()

# Get cumulative validated explained variance for each variable in Y and store in pandas dataframe with row and column names
Y_cumValExplVar_ind_df = pd.DataFrame(model.Y_cumValExplVar_indVar())
Y_cumValExplVar_ind_df.columns = Y_varNames
Y_cumValExplVar_ind_df.index = ['PC{0}'.format(x) for x in range(model.Y_loadings().shape[1] + 1)]
Y_cumValExplVar_ind_df


# In[48]:


help(ho.nipalsPLS2.X_cumValExplVar_indVar)


# In[49]:


# Get validated predicted Y for a given number of components

# Predicted Y from validation using 1 component
Y_from_1_component_val = model.Y_predVal()[1]

# Predicted Y from calibration using 1 component stored in pandas data frame with row and columns names
Y_from_1_component_val_df = pd.DataFrame(model.Y_predVal()[1])
Y_from_1_component_val_df.index = Y_objNames
Y_from_1_component_val_df.columns = Y_varNames
Y_from_1_component_val_df


# In[50]:


# Get validated predicted Y for a given number of components

# Predicted Y from validation using 3 components
Y_from_3_component_val = model.Y_predVal()[3]

# Predicted Y from calibration using 3 components stored in pandas data frame with row and columns names
Y_from_3_component_val_df = pd.DataFrame(model.Y_predVal()[3])
Y_from_3_component_val_df.index = Y_objNames
Y_from_3_component_val_df.columns = Y_varNames
Y_from_3_component_val_df


# In[51]:


help(ho.nipalsPLS2.Y_predVal)


# In[52]:


# Get predicted scores for new measurements (objects) of X

# First pretend that we acquired new X data by using part of the existing data and overlaying some noise
import numpy.random as npr
new_X = X[0:4, :] + npr.rand(4, np.shape(X)[1])
np.shape(X)

# Now insert the new data into the existing model and compute scores for two components (numComp=2)
pred_X_scores = model.X_scores_predict(new_X, numComp=2)

# Same as above, but results stored in a pandas dataframe with row names and column names
pred_X_scores_df = pd.DataFrame(model.X_scores_predict(new_X, numComp=2))
pred_X_scores_df.columns = ['PC{0}'.format(x+1) for x in range(2)]
pred_X_scores_df.index = ['new object {0}'.format(x+1) for x in range(np.shape(new_X)[0])]
pred_X_scores_df


# In[53]:


help(ho.nipalsPLS2.X_scores_predict)


# In[54]:


# Predict Y from new X data
pred_Y = model.Y_predict(new_X, numComp=2)

# Predict Y from nex X data and store results in a pandas dataframe with row names and column names
pred_Y_df = pd.DataFrame(model.Y_predict(new_X, numComp=2))
pred_Y_df.columns = Y_varNames
pred_Y_df.index = ['new object {0}'.format(x+1) for x in range(np.shape(new_X)[0])]
pred_Y_df

