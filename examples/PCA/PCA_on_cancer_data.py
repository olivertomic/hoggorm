
# coding: utf-8

# # Principal component analysis (PCA) on cancer data

# This notebook illustrates how to use the **hoggorm** package to carry out principal component analysis (PCA) on a multivariate data set on cancer in men across OECD countries. Furthermore, we will learn how to visualise the results of the PCA using the **hoggormPlot** package.

# ---

# ### Import packages and prepare data

# First import **hoggorm** for analysis of the data and **hoggormPlot** for plotting of the analysis results. We'll also import **pandas** such that we can read the data into a data frame. **numpy** is needed for checking dimensions of the data.

# In[1]:


import hoggorm as ho
import hoggormplot as hop
import pandas as pd
import numpy as np


# Next, load the cancer data that we are going to analyse using **hoggorm**. The data can be acquired from the OECD (The Organisation for Economic Co-operation and Development) and holds the percentages of various cacner types in men. After the data has been loaded into the pandas data frame, we'll display it in the notebook.

# In[2]:


# Load OECD data for cancer in men

# Insert code for reading data from other folder in repository instead of directly from same repository.
data_df = pd.read_csv('Cancer_men_perc.txt', index_col=0, sep='\t')
data_df


# Let's have a look at the dimensions of the data frame.

# In[3]:


np.shape(data_df)


# There are observations for 34 countries as well as all OECD countries together, which results in 35 rows. Furthermore, there are 10 columns where each column represents one type of cancer in men.

# The ``nipalsPCA`` class in hoggorm accepts only **numpy** arrays with numerical values and not pandas data frames. Therefore, the pandas data frame holding the imported data needs to be "taken apart" into three parts: 
# * a numpy array holding the numeric values
# * a Python list holding variable (column) names
# * a Python list holding object (row) names. 
# 
# The array with values will be used as input for the ``nipalsPCA`` class for analysis. The Python lists holding the variable and row names will be used later in the plotting function from the **hoggormPlot** package when visualising the results of the analysis. Below is the code needed to access both data, variable names and object names.

# In[4]:


# Get the values from the data frame
data = data_df.values

# Get the variable or columns names
data_varNames = list(data_df.columns)

# Get the object or row names
data_objNames = list(data_df.index)


# Let's have a quick look at the column or variable names.

# In[5]:


data_varNames


# Now show the object or row names.

# In[6]:


data_objNames


# ---

# ### Apply PCA to our data

# Now, let's run PCA on the data using the ``nipalsPCA`` class. The documentation provides a [description of the input parameters](https://hoggorm.readthedocs.io/en/latest/pca.html). Using input paramter ``arrX`` we define which numpy array we would like to analyse. By setting input parameter ``Xstand=False`` we make sure that the variables are only mean centered, not scaled to unit variance. This is the default setting and actually doesn't need to expressed explicitly. Setting paramter ``cvType=["loo"]`` we make sure that we compute the PCA model using full cross validation. ``"loo"`` means "Leave One Out". By setting paramter ``numpComp=4`` we ask for four principal components (PC) to be computed.

# In[7]:


model = ho.nipalsPCA(arrX=data, Xstand=False, cvType=["loo"], numComp=4)


# That's it, the PCA model has been computed. Now we would like to inspect the results by visualising them. We can do this using the taylor-made plotting function for PCA from the separate [**hoggormPlot** package](https://hoggormplot.readthedocs.io/en/latest/). If we wish to plot the results for component 1 and component 2, we can do this by setting the input argument ``comp=[1, 2]``. The input argument ``plots=[1, 2, 3, 4, 6]`` lets the user define which plots are to be plotted. If this list for example contains value ``1``, the function will generate the scores plot for the model. If the list contains value ``2``, then the loadings plot will be plotted. Value ``3`` stands for correlation loadings plot and value ``4`` stands for bi-plot and ``6`` stands for explained variance plot. The hoggormPlot documentation provides a [description of input paramters](https://hoggormplot.readthedocs.io/en/latest/mainPlot.html).

# In[8]:


hop.plot(model, comp=[1, 2], 
         plots=[1, 2, 3, 4, 6], 
         objNames=data_objNames, 
         XvarNames=data_varNames)


# ---

# ### Accessing numerical results

# Now that we have visualised the PCA results, we may also want to access the numerical results. Below are some examples. For a complete list of accessible results, please see this part of the documentation.  

# In[9]:


# Get scores and store in numpy array
scores = model.X_scores()

# Get scores and store in pandas dataframe with row and column names
scores_df = pd.DataFrame(model.X_scores())
scores_df.index = data_objNames
scores_df.columns = ['PC{0}'.format(x+1) for x in range(model.X_scores().shape[1])]
scores_df


# In[10]:


help(ho.nipalsPCA.X_scores)


# In[11]:


# Dimension of the scores
np.shape(model.X_scores())


# We see that the numpy array holds the scores for all countries and OECD (35 in total) for four components as required when computing the PCA model.

# In[12]:


# Get loadings and store in numpy array
loadings = model.X_loadings()

# Get loadings and store in pandas dataframe with row and column names
loadings_df = pd.DataFrame(model.X_loadings())
loadings_df.index = data_varNames
loadings_df.columns = ['PC{0}'.format(x+1) for x in range(model.X_loadings().shape[1])]
loadings_df


# In[13]:


help(ho.nipalsPCA.X_loadings)


# In[14]:


np.shape(model.X_loadings())


# Here we see that the array holds the loadings for the 10 variables in the data across four components.

# In[15]:


# Get loadings and store in numpy array
loadings = model.X_corrLoadings()

# Get loadings and store in pandas dataframe with row and column names
loadings_df = pd.DataFrame(model.X_corrLoadings())
loadings_df.index = data_varNames
loadings_df.columns = ['PC{0}'.format(x+1) for x in range(model.X_corrLoadings().shape[1])]
loadings_df


# In[16]:


help(ho.nipalsPCA.X_corrLoadings)


# In[17]:


# Get calibrated explained variance of each component
calExplVar = model.X_calExplVar()

# Get calibrated explained variance and store in pandas dataframe with row and column names
calExplVar_df = pd.DataFrame(model.X_calExplVar())
calExplVar_df.columns = ['calibrated explained variance']
calExplVar_df.index = ['PC{0}'.format(x+1) for x in range(model.X_loadings().shape[1])]
calExplVar_df


# In[18]:


help(ho.nipalsPCA.X_calExplVar)


# In[19]:


# Get cumulative calibrated explained variance
cumCalExplVar = model.X_cumCalExplVar()

# Get cumulative calibrated explained variance and store in pandas dataframe with row and column names
cumCalExplVar_df = pd.DataFrame(model.X_cumCalExplVar())
cumCalExplVar_df.columns = ['cumulative calibrated explained variance']
cumCalExplVar_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
cumCalExplVar_df


# In[20]:


help(ho.nipalsPCA.X_cumCalExplVar)


# In[21]:


# Get cumulative calibrated explained variance for each variable
cumCalExplVar_ind = model.X_cumCalExplVar_indVar()

# Get cumulative calibrated explained variance for each variable and store in pandas dataframe with row and column names
cumCalExplVar_ind_df = pd.DataFrame(model.X_cumCalExplVar_indVar())
cumCalExplVar_ind_df.columns = data_varNames
cumCalExplVar_ind_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
cumCalExplVar_ind_df


# In[22]:


help(ho.nipalsPCA.X_cumCalExplVar_indVar)


# In[23]:


# Get calibrated predicted X for a given number of components

# Predicted X from calibration using 1 component
X_from_1_component = model.X_predCal()[1]

# Predicted X from calibration using 1 component stored in pandas data frame with row and columns names
X_from_1_component_df = pd.DataFrame(model.X_predCal()[1])
X_from_1_component_df.index = data_objNames
X_from_1_component_df.columns = data_varNames
X_from_1_component_df


# In[24]:


# Get predicted X for a given number of components

# Predicted X from calibration using 4 components
X_from_4_component = model.X_predCal()[4]

# Predicted X from calibration using 1 component stored in pandas data frame with row and columns names
X_from_4_component_df = pd.DataFrame(model.X_predCal()[4])
X_from_4_component_df.index = data_objNames
X_from_4_component_df.columns = data_varNames
X_from_4_component_df


# In[25]:


help(ho.nipalsPCA.X_predCal)


# In[26]:


# Get validated explained variance of each component
valExplVar = model.X_valExplVar()

# Get calibrated explained variance and store in pandas dataframe with row and column names
valExplVar_df = pd.DataFrame(model.X_valExplVar())
valExplVar_df.columns = ['validated explained variance']
valExplVar_df.index = ['PC{0}'.format(x+1) for x in range(model.X_loadings().shape[1])]
valExplVar_df


# In[27]:


help(ho.nipalsPCA.X_valExplVar)


# In[28]:


# Get cumulative validated explained variance
cumValExplVar = model.X_cumValExplVar()

# Get cumulative validated explained variance and store in pandas dataframe with row and column names
cumValExplVar_df = pd.DataFrame(model.X_cumValExplVar())
cumValExplVar_df.columns = ['cumulative validated explained variance']
cumValExplVar_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
cumValExplVar_df


# In[29]:


help(ho.nipalsPCA.X_cumValExplVar)


# In[30]:


# Get cumulative validated explained variance for each variable
cumCalExplVar_ind = model.X_cumCalExplVar_indVar()

# Get cumulative validated explained variance for each variable and store in pandas dataframe with row and column names
cumValExplVar_ind_df = pd.DataFrame(model.X_cumValExplVar_indVar())
cumValExplVar_ind_df.columns = data_varNames
cumValExplVar_ind_df.index = ['PC{0}'.format(x) for x in range(model.X_loadings().shape[1] + 1)]
cumValExplVar_ind_df


# In[31]:


help(ho.nipalsPCA.X_cumValExplVar_indVar)


# In[32]:


# Get validated predicted X for a given number of components

# Predicted X from validation using 1 component
X_from_1_component_val = model.X_predVal()[1]

# Predicted X from calibration using 1 component stored in pandas data frame with row and columns names
X_from_1_component_val_df = pd.DataFrame(model.X_predVal()[1])
X_from_1_component_val_df.index = data_objNames
X_from_1_component_val_df.columns = data_varNames
X_from_1_component_val_df


# In[33]:


# Get validated predicted X for a given number of components

# Predicted X from validation using 3 components
X_from_3_component_val = model.X_predVal()[3]

# Predicted X from calibration using 3 components stored in pandas data frame with row and columns names
X_from_3_component_val_df = pd.DataFrame(model.X_predVal()[3])
X_from_3_component_val_df.index = data_objNames
X_from_3_component_val_df.columns = data_varNames
X_from_3_component_val_df


# In[34]:


help(ho.nipalsPCA.X_predVal)


# In[35]:


# Get predicted scores for new measurements (objects) of X

# First pretend that we acquired new X data by using part of the existing data and overlaying some noise
import numpy.random as npr
new_data = data[0:4, :] + npr.rand(4, np.shape(data)[1])
np.shape(new_data)

# Now insert the new data into the existing model and compute scores for two components (numComp=2)
pred_scores = model.X_scores_predict(new_data, numComp=2)

# Same as above, but results stored in a pandas dataframe with row names and column names
pred_scores_df = pd.DataFrame(model.X_scores_predict(new_data, numComp=2))
pred_scores_df.columns = ['PC{0}'.format(x) for x in range(2)]
pred_scores_df.index = ['new object {0}'.format(x) for x in range(np.shape(new_data)[0])]
pred_scores_df


# In[36]:


help(ho.nipalsPCA.X_scores_predict)

