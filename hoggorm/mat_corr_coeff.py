# -*- coding: utf-8 -*-


import numpy


def RVcoeff(dataList):
    """
    This function computes the RV matrix correlation coefficients between pairs
    of arrays. The number and order of objects (rows) for the two arrays must
    match. The number of variables in each array may vary.

    REF: `H. Abdi, D. Valentin; 'The STATIS method'`_

    .. _H. Abdi, D. Valentin; 'The STATIS method': https://www.utdallas.edu/~herve/Abdi-Statis2007-pretty.pdf

    PARAMETERS
    ----------
    dataList : list
        A list holding numpy arrays for which the RV coefficient will be computed.

    RETURNS
    -------
    numpy array
        A numpy array holding RV coefficients for pairs of numpy arrays. The
        diagonal in the result array holds ones, since RV is computed on
        identical arrays, i.e. first array in ``dataList`` against frist array
        in

    Examples
    --------
    >>> import hoggorm as ho
    >>> import numpy as np
    >>>
    >>> # Generate some random data. Note that number of rows must match across arrays
    >>> arr1 = np.random.rand(50, 100)
    >>> arr2 = np.random.rand(50, 20)
    >>> arr3 = np.random.rand(50, 500)
    >>>
    >>> # Center the data before computation of RV coefficients
    >>> arr1_cent = arr1 - np.mean(arr1, axis=0)
    >>> arr2_cent = arr2 - np.mean(arr2, axis=0)
    >>> arr3_cent = arr3 - np.mean(arr3, axis=0)
    >>>
    >>> # Compute RV matrix correlation coefficients on mean centered data
    >>> rv_results = ho.RVcoeff([arr1_cent, arr2_cent, arr3_cent])
    >>> array([[ 1.        ,  0.41751839,  0.77769025],
               [ 0.41751839,  1.        ,  0.51194496],
               [ 0.77769025,  0.51194496,  1.        ]])
    >>>
    >>> # Get RV for arr1_cent and arr2_cent
    >>> rv_results[0, 1]
        0.41751838661314689
    >>>
    >>> # or
    >>> rv_results[1, 0]
        0.41751838661314689
    >>>
    >>> # Get RV for arr2_cent and arr3_cent
    >>> rv_results[1, 2]
        0.51194496245209853
    >>>
    >>> # or
    >>> rv_results[2, 1]
        0.51194496245209853

    """

    # First compute the scalar product matrices for each data set X
    scalArrList = []

    for arr in dataList:
        scalArr = numpy.dot(arr, numpy.transpose(arr))
        scalArrList.append(scalArr)


    # Now compute the 'between study cosine matrix' C
    C = numpy.zeros((len(dataList), len(dataList)), float)


    for index, element in numpy.ndenumerate(C):
        nom = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]),
                                    scalArrList[index[1]]))
        denom1 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]),
                                       scalArrList[index[0]]))
        denom2 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[1]]),
                                       scalArrList[index[1]]))
        Rv = nom / numpy.sqrt(numpy.dot(denom1, denom2))
        C[index[0], index[1]] = Rv

    return C



def RV2coeff(dataList):
    """
    This function computes the RV matrix correlation coefficients between pairs
    of arrays. The number and order of objects (rows) for the two arrays must
    match. The number of variables in each array may vary. The RV2 coefficient
    is a modified version of the RV coefficient with values -1 <= RV2 <= 1.
    RV2 is independent of object and variable size.

    REF: `A.K. Smilde, et al. Bioinformatics (2009) Vol 25, no 3, 401-405`_

    .. _A.K. Smilde, et al. Bioinformatics (2009) Vol 25, no 3, 401-405: https://academic.oup.com/bioinformatics/article/25/3/401/244239

    PARAMETERS
    ----------
    dataList : list
        A list holding an arbitrary number of numpy arrays for which the RV
        coefficient will be computed.

    RETURNS
    -------
    numpy array
        A list holding an arbitrary number of numpy arrays for which the RV
        coefficient will be computed.

    Examples
    --------
    >>> import hoggorm as ho
    >>> import numpy as np
    >>>
    >>> # Generate some random data. Note that number of rows must match across arrays
    >>> arr1 = np.random.rand(50, 100)
    >>> arr2 = np.random.rand(50, 20)
    >>> arr3 = np.random.rand(50, 500)
    >>>
    >>> # Center the data before computation of RV coefficients
    >>> arr1_cent = arr1 - np.mean(arr1, axis=0)
    >>> arr2_cent = arr2 - np.mean(arr2, axis=0)
    >>> arr3_cent = arr3 - np.mean(arr3, axis=0)
    >>>
    >>> # Compute RV matrix correlation coefficients on mean centered data
    >>> rv_results = ho.RVcoeff([arr1_cent, arr2_cent, arr3_cent])
    >>> array([[ 1.        , -0.00563174,  0.04028299],
               [-0.00563174,  1.        ,  0.08733739],
               [ 0.04028299,  0.08733739,  1.        ]])
    >>>
    >>> # Get RV for arr1_cent and arr2_cent
    >>> rv_results[0, 1]
        -0.00563174
    >>>
    >>> # or
    >>> rv_results[1, 0]
        -0.00563174
    >>>
    >>> # Get RV for arr2_cent and arr3_cent
    >>> rv_results[1, 2]
        0.08733739
    >>>
    >>> # or
    >>> rv_results[2, 1]
        0.08733739
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
        nom = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]),
                                    scalArrList[index[1]]))
        denom1 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[0]]),
                                       scalArrList[index[0]]))
        denom2 = numpy.trace(numpy.dot(numpy.transpose(scalArrList[index[1]]),
                                       scalArrList[index[1]]))
        Rv = nom / numpy.sqrt(denom1 * denom2)
        C[index[0], index[1]] = Rv

    return C
