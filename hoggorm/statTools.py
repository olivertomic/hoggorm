# -*- coding: utf-8 -*-

import numpy
import numpy.linalg



def ortho(arr1, arr2):
    """
    This function orthogonalises arr1 with respect to arr2. The function then
    returns orthogonalised array arr1_orth.

    PARAMETERS
    ----------
    arr1 : numpy array
        A numpy array containing some data

    arr2 : numpy array
        A numpy array containing some data

    RETURNS
    -------
    numpy array
        A numpy array holding orthogonalised numpy array ``arr1``.

    Examples
    --------
    some examples

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



def center(arr, axis=0):
    """
    This function centers an array column-wise or row-wise.

    PARAMETERS
    ----------
    arrX : numpy array
        A numpy array containing the data

    RETURNS
    -------
    numpy array
        Mean centered data.

    Examples
    --------
    >>> import hoggorm as ho
    >>> # Column centering of array
    >>> centData = ho.center(data, axis=0)

    >>> # Row centering of array
    >>> centData = ho.center(data, axis=1)
    """

    # First make a copy of input matrix and make it a matrix with float
    # elements
    X = numpy.array(arr, float)

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



def standardise(arr, mode=0):
    """
    This function standardises the input array either
    column-wise (mode = 0) or row-wise (mode = 1).

    PARAMETERS
    ----------
    arrX : numpy array
        A numpy array containing the data

    selection : int
        An integer indicating whether standardisation should happen column
        wise or row wise.

    RETURNS
    -------
    numpy array
        Standardised data.

    Examples
    --------
    >>> import hoggorm as ho
    >>> # Standardise array column-wise
    >>> standData = ho.standardise(data, mode=0)

    >>> # Standardise array row-wise
    >>> standData = ho.standarise(data, mode=1)
    """
    # First make a copy of input array
    X = arr.copy()

    # Standardisation column-wise
    if mode == 0:
        colMeans = numpy.mean(X, axis=0)
        colSTD = numpy.std(X, axis=0, ddof=1)
        centX = X - colMeans
        stdX = centX / colSTD


    # Standardisation of row-wise
    # Transpose array first, such that broadcasting procedure works easier.
    # After standardisation transpose back to get final array.
    if mode == 1:
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

    PARAMETERS
    ----------
    arrX : numpy array
        A numpy array containing the data

    RETURNS
    -------
    scalar
        Rank of matrix.

    Examples
    --------
    >>> import hoggorm as ho
    >>>
    >>> # Get the rank of the data
    >>> ho.matrixRank(myData)
    >>> 8

    """
    if len(arr.shape) != 2:
        raise ValueError('Input must be a 2-d array or Matrix object')

    s = numpy.linalg.svd(arr, compute_uv=0)
    return numpy.sum(numpy.where(s > tol, 1, 0))
