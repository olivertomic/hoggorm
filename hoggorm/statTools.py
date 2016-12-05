# -*- coding: utf-8 -*-
""" StatTools module

A collection of tools for data analysis
"""

import numpy
import numpy.linalg

def RVcoeff(dataList):
    """
    This function computes the Rv coefficients between two matrices at the
    time. The results are stored in a matrix described as 'between cosine
    matrix' and is labled C.

    REF: H. Abdi, D. Valentin; 'The STATIS method' (unofficial paper)

    <dataList>: type list holding rectangular matrices (no need for equal dim)
    """

    # First compute the scalar product matrices for each data set X
    scalArrList = []

    for arr in dataList:
        scalArr = numpy.dot(arr, numpy.transpose(arr))
        scalArrList.append(scalArr)


    # Now compute the 'between study cosine matrix' C
    C = numpy.zeros((len(dataList), len(dataList)), float)


    for index, element in numpy.ndenumerate(C):
        nom = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]), \
            scalArrList[index[1]]))
        denom1 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]), \
            scalArrList[index[0]]))
        denom2 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[1]]), \
            scalArrList[index[1]]))
        Rv = nom / numpy.sqrt(numpy.dot(denom1, denom2))
        C[index[0], index[1]] = Rv

    return C




def RV2coeff(dataList):
    """
    This function computes the RV2 coefficients between two matrices at the
    time. The RV2 coefficient is a modified version of the RV coefficient
    with values -1 <= RV2 <= 1. RV2 is independent of object and variable
    size.

    REF: A.K. Smilde, et al. Bioinformatics (2009) Vol 25, no 3, 401-405

    <dataList>: type list holding rectangular matrices (no need for equal dim)
    """

    # First compute the scalar product matrices for each data set X
    scalArrList = []

    for arr in dataList:
        scalArr = numpy.dot(arr, numpy.transpose(arr))
        diego = numpy.diag(numpy.diag(scalArr))
        scalArrMod = scalArr - diego
        scalArrList.append(scalArrMod)


    # Now compute the 'between study cosine matrix' C
    C = numpy.zeros((len(dataList), len(dataList)), float)


    for index, element in numpy.ndenumerate(C):
        nom = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]), \
            scalArrList[index[1]]))
        denom1 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]), \
            scalArrList[index[0]]))
        denom2 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[1]]), \
            scalArrList[index[1]]))
        Rv = nom / numpy.sqrt(denom1 * denom2)
        C[index[0], index[1]] = Rv

    return C



def RV2coeff_VEC(dataList):
    """
    This function computes the RV2 coefficients between two matrices at the
    time. The RV2 coefficient is a modified version of the RV coefficient
    with values -1 <= RV2 <= 1. RV2 is independent of object and variable
    size.

    REF: A.K. Smilde, et al. Bioinformatics (2009) Vol 25, no 3, 401-405

    <dataList>: type list holding rectangular matrices (no need for equal dim)
    """

    # Then compute the scalar product matrices for each data set X
    scalArrList = []

    for arr in dataList:
        scalArr = numpy.dot(arr, numpy.transpose(arr))
        diego = numpy.diag(numpy.diag(scalArr))
        scalArrMod = scalArr - diego
        scalArrList.append(scalArrMod)


    # Now compute the 'between study cosine matrix' C
    C = numpy.zeros((len(dataList), len(dataList)), float)


    for index, element in numpy.ndenumerate(C):
        nom1 = numpy.ndarray.flatten(scalArrList[index[0]], 'F')
        nom2 = numpy.ndarray.flatten(scalArrList[index[1]], 'F')        
        nom = numpy.dot(numpy.transpose(nom1), nom2)
        
        denom1 = numpy.dot(numpy.transpose(nom1), nom1)
        denom2 = numpy.dot(numpy.transpose(nom2), nom2)

        Rv = nom / numpy.sqrt(numpy.dot(denom1, denom2))
        C[index[0], index[1]] = Rv

    return C



def normProcrSim(dataList):
    """
    This function computes the normalised Procrustes similarity between two 
    matrices at the time. The results are stored in a matrix described as 
    'between cosine matrix' and is labled C.

    REF: E. Qannari, H. MacFie, P. Courcoux
         Food Quality and Preference 10 (1999) 17-21

    <dataList>: type list holding rectangular matrices (no need for equal dim)
    """

    # First centre matrices column-wise
    centArrList = []
    for arr in dataList:
        colMeans = numpy.mean(arr, axis=0)
        centArr = arr - colMeans
        centArrList.append(centArr)


    # Now compute the 'between study cosine matrix' C
    C = numpy.zeros((len(dataList), len(dataList)), float)


    for index, element in numpy.ndenumerate(C):
        nom = numpy.trace(numpy.dot(numpy.transpose(centArrList[index[0]]), \
            centArrList[index[1]]))
        denom1 = numpy.trace(numpy.dot(numpy.transpose(centArrList[index[0]]), \
            centArrList[index[0]]))
        denom2 = numpy.trace(numpy.dot(numpy.transpose(centArrList[index[1]]), \
            centArrList[index[1]]))
        NPS = nom / numpy.sqrt(denom1) * numpy.sqrt(denom2)
        C[index[0], index[1]] = NPS

    return C




def ortho(arr1, arr2):
    """
    This function orthogonalises arr1 with respect to arr2. The function then
    returns orthogonalised array arr1_orth.
    """

    # Find number of rows, such that identity matrix I can be created
    numberRows = numpy.shape(arr1)[0]
    I = numpy.identity(numberRows, float)

    # Compute transpose of arr1
    arr2_T = numpy.transpose(arr2)

    term1 = numpy.linalg.inv(numpy.dot(arr2_T, arr2))
    term2 = numpy.dot(arr2, term1)
    term3 = numpy.dot(term2, arr2_T)
    arr1_orth = numpy.dot((I - term3), arr1)

    return arr1_orth
    
    
    
def centre_old(Y):
    """
    This function centers an array column-wise.
    """

    # First make a copy of input matrix and make it a matrix with float
    # elements
    X = numpy.array(Y, float)
    numberOfObjects, numberOfVariables = numpy.shape(X)
    variableMean = numpy.average(X, 0)

    # Now center by subtracting column means.
    for row in range(0, numberOfObjects):
        X[row] = X[row] - variableMean

    return X



def centre(Y, axis=0):
    """
    This function centers an array column-wise or row-wise.
    
    Examples:
    --------
    
    centData = statTools.centre(data, axis=0) centres column-wise
    
    centData = statTools.centre(datak axis=1) centres row-wise
    """

    # First make a copy of input matrix and make it a matrix with float
    # elements
    X = numpy.array(Y, float)
    
    # Check whether column or row centring is required.
    # Centreing column-wise
    if axis == 0:
        variableMean = numpy.average(X, 0)
        centX = X - variableMean
    
    # Centreing row-wise. 
    if axis == 1:
        transX = numpy.transpose(X)
        objectMean = numpy.average(transX, 0)
        transCentX = transX - objectMean
        centX = numpy.transpose(transCentX)
    
    return centX



def STD(Y, selection):
    """
    This function standardises the input array either 
    column-wise (selection = 0) or row-wise (selection = 1).
    """
    # First make a copy of input array
    #X = array(Y, float)
    X = Y.copy()
    numberOfObjects, numberOfVariables = numpy.shape(X)
    colMeans = numpy.mean(X, axis=0)
    #rowMeans = numpy.mean(X, axis=1)
    
    colSTD = numpy.std(X, axis=0, ddof=1)
    #rowSTD = numpy.std(X, axis=1, ddof=1)


    # Standardisation column-wise
    if selection == 0:
        centX = X - colMeans
        stdX = centX / colSTD


    # Standardisation of row-wise
    # Transpose array first, such that broadcasting procedure works easier.
    # After standardisation transpose back to get final array.
    if selection == 1:
        transX = numpy.transpose(X)
        transColMeans = numpy.mean(transX, axis=0)
        transColSTD = numpy.mean(transX, axis=0)
        
        centTransX = transX - transColMeans
        stdTransX = centTransX / transColSTD
        stdX = numpy.transpose(stdTransX)
        

    return stdX



def matrixRank(arr, tol=1e-8):
    """
    Computes the rank of an array/matrix, i.e. number of linearly independent
    variables. This is not the same as numpy.rank() which only returns the
    number of ways (2-way, 3-way, etc) an array/matrix has. 
    """
    if len(arr.shape) != 2:
        raise ValueError('Input must be a 2-d array or Matrix object')
    
    s = numpy.linalg.svd(arr, compute_uv=0)
    return numpy.sum(numpy.where(s>tol, 1, 0))

    


class arrayIO:
    def __init__(self, fileName):
        """
        This class reads data from text files. First row are variable names
        and first row are object names. 
        
        INPUT:
        <fileName>: type string
        """
        
        # File is opened using name that is given by
        # the file-open dialog in the main file.
        dataFile = open(fileName, 'r')


        # All the data is read into a list.
        allText = dataFile.readlines()
        
        
        # Initiate lists that will hold variable names, object names and data. 
        varNames = []
        objNames = []
        data = []
        
        
        # Loop through allText and extract variable names, object names and
        # data. 
        for ind, row in enumerate(allText):
            
            # Get variable names from first row
            if ind == 0:
                firstRowList = row.split('\t')
                firstRowList[-1] = firstRowList[-1][:-1]
                varNames = firstRowList[1:]
            
            # Split remaining rows into object names and data
            else:
                rowObjectsList = row.split('\t')
                objNames.append(rowObjectsList[0])
                rowObjectsList.pop(0)
                
                # Convert strings into floats
                floatList = []
                for item in rowObjectsList:
                    floatList.append(float(item))
                    
                data.append(floatList)
            
        
        # Make variable names, object names and data available as 
        # class variables.
        self.varNames = varNames[:]
        self.objNames = objNames[:]
        self.data = numpy.array(data)
        
