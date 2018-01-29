# -*- coding: utf-8 -*-


import numpy
import hoggorm.statTools as st


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


class SMI:
    """
    Similarity of Matrices Index (SMI)
    
    A similarity index for comparing coupled data matrices. 
    A two-step process starts with extraction of stable subspaces using 
    Principal Component Analysis or some other method yielding two orthonormal bases. These bases
    are compared using Orthogonal Projection (OP / ordinary least squares) or Procrustes
    Rotation (PR). The result is a similarity measure that can be adjusted to various
    data sets and contexts and which includes explorative plotting and permutation based testing
    of matrix subspace equality.
    
    Reference: A similarity index for comparing coupled matrices - Ulf Geir Indahl, Tormod Næs, Kristian Hovde Liland

    PARAMETERS
    ----------
    X1 : numpy array
        first matrix to be compared.
    X2 : numpy array
        second matrix to be compared.
    ncomp1 : int, optional
        maximum number of subspace components from the first matrix.
    ncomp2 : int, optional
        maximum number of subspace components from the second matrix.
    projection : list, optional
        type of projection to apply, defaults to "Orthogonal", alternatively "Procrustes".
    Scores1 : numpy array, optional
        user supplied score-matrix to replace singular value decomposition of first matrix.
    Scores2 : numpy array, optional
        user supplied score-matrix to replace singular value decomposition of second matrix.
    
    RETURNS
    -------
    An SMI object containing all combinations of components.

    EXAMPLES
    --------
    >>> import numpy as np
    >>> import SMI as S
    >>> import statTools as st
    
    >>> X1 = st.centre(np.random.rand(100,300))
    >>> U, s, V = np.linalg.svd(X1, 0)
    >>> X2 = np.dot(np.dot(np.delete(U, 2,1), np.diag(np.delete(s,2))), np.delete(V,2,0))
    
    >>> smiOP = S.SMI(X1,X2, ncomp1 = 10, ncomp2 = 10)
    >>> smiPR = S.SMI(X1,X2, ncomp1 = 10, ncomp2 = 10, projection = "Procrustes")
    >>> smiCustom = S.SMI(X1,X2, ncomp1 = 10, ncomp2 = 10, Scores1 = U)
    
    >>> print(smiOP.smi)
    >>> print(smiOP.significance())
    >>> print(smiPR.significance(B = 100))
    """
    
    def __init__(self, X1, X2, **kargs):
        # Check dimensions
        assert numpy.shape(X1)[0] == numpy.shape(X2)[0], ValueError('Number of objects must be equal in X1 and X2')
        
        # Check number of components against rank
        rank1 = st.matrixRank(X1)
        rank2 = st.matrixRank(X2)
        if 'ncomp1' not in kargs.keys():
            self.ncomp1 = rank1
        else:
            self.ncomp1 = kargs['ncomp1']

        if 'ncomp2' not in kargs.keys():
            self.ncomp2 = rank2
        else:
            self.ncomp2 = kargs['ncomp2']
        assert self.ncomp1 <= rank1, ValueError('Number of components for X1 cannot be higher than the rank of X1')
        assert self.ncomp2 <= rank2, ValueError('Number of components for X2 cannot be higher than the rank of X2')

        # Handle projection types
        if 'projection' not in kargs.keys():
            self.projection = 'Orthogonal'
        else:
            self.projection = kargs['projection']
        
        assert self.projection in ['Orthogonal','Procrustes'], ValueError('Unknown projection, should be Orthogonal or Procrustes')
        
        # Calculate scores if needed
        if 'Scores1' not in kargs.keys():
            Scores1, s, V = numpy.linalg.svd(X1 - numpy.mean(X1, axis=0),0)
        else:
            Scores1 = kargs['Scores1']
        if 'Scores2' not in kargs.keys():
            Scores2, s, V = numpy.linalg.svd(X2 - numpy.mean(X2, axis=0),0)
        else:
            Scores2 = kargs['Scores2']
            
        # Compute SMI values
        if self.projection == 'Orthogonal':
            self.smi = numpy.cumsum(numpy.cumsum(numpy.square(numpy.dot(numpy.transpose(Scores1[:,:self.ncomp1]), Scores2[:,:self.ncomp2])),axis=1),axis=0) \
                                / (numpy.reshape(numpy.min(numpy.vstack([numpy.tile(range(self.ncomp1),self.ncomp1),numpy.repeat(range(self.ncomp2),self.ncomp2)]),0), [self.ncomp1,self.ncomp2]) + 1)
        else:
            # Procrustes
            self.smi = numpy.zeros([self.ncomp1, self.ncomp2])
            TU = numpy.dot(numpy.transpose(Scores1[:,0:self.ncomp1]),Scores2[:,0:self.ncomp2])
            for p in range(self.ncomp1):
                for q in range(self.ncomp2):
                    U, s, V = numpy.linalg.svd(TU[:p+1,:q+1])
                    self.smi[p,q] = numpy.square(numpy.mean(s))
        
        # Recover wrong calculations (due to numerics)
        self.smi[self.smi > 1] = 1; self.smi[self.smi < 0] = 0
           
        self.N = numpy.shape(Scores1)[0]
        self.Scores1 = Scores1
        self.Scores2 = Scores2


    def significance(self, **kargs):
        """
        Significance estimation for Similarity of Matrices Index (SMI)
        
        For each combination of components significance is estimated by sampling from a null distribution
        of no similarity, i.e. when the rows of one matrix is permuted B times and corresponding SMI values are
        computed. If the vector replicates is included, replicates will be kept together through
        permutations.
        
        PARAMETERS
        ----------
        B integer : int, optional
            number of permutations, default = 10000.
        replicates : numpy array
            integer vector of replicates (must be balanced).
    
        RETURNS
        -------
        An array containing P-values for all combinations of components.
        """
        if 'B' not in kargs.keys():
            B = 10000
        else:
            B = kargs['B']
        P = numpy.zeros([self.ncomp1, self.ncomp2])
        
        if self.projection == 'Orthogonal':
            m = (numpy.reshape(numpy.min(numpy.vstack([numpy.tile(range(self.ncomp1),self.ncomp1),numpy.repeat(range(self.ncomp2),self.ncomp2)]),0), [self.ncomp1,self.ncomp2]) + 1)
            if 'replicates' not in kargs.keys():
                BScores1 = self.Scores1.copy()
                i = 0
                while i < B:
                    numpy.random.shuffle(BScores1)
                    smiB = numpy.cumsum(numpy.cumsum(numpy.square(numpy.dot(numpy.transpose(BScores1[:,:self.ncomp1]), self.Scores2[:,:self.ncomp2])),axis=1),axis=0) / m
                    # Increase P-value if non-significant permutation
                    P[self.smi > numpy.maximum(smiB,1-smiB)] += 1
                    i += 1
            else:
                # With replicates
                AScores1 = self.Scores1.copy()
                BScores1 = self.Scores1.copy()
                i = 0
                replicates = kargs['replicates']
                uni = numpy.unique(replicates, return_inverse=True)
                vecOut = numpy.array(range(numpy.shape(uni[0])[0]))
                vecIn  = numpy.array(range(sum(uni[1]==0)))
                nOut = len(vecOut)
                nIn  = len(vecIn)
                while i < B:
                    numpy.random.shuffle(vecOut) # Permute across replicate sets
                    for j in range(nOut):
                        numpy.random.shuffle(vecIn) # Permute inside replicate sets
                        BScores1[uni[1]==j,:] = AScores1[vecOut[j]*(nIn)+vecIn,:]
                    smiB = numpy.cumsum(numpy.cumsum(numpy.square(numpy.dot(numpy.transpose(BScores1[:,:self.ncomp1]), self.Scores2[:,:self.ncomp2])),axis=1),axis=0) / m
                    # Increase P-value if non-significant permutation
                    P[self.smi > numpy.maximum(smiB,1-smiB)] += 1
                    i += 1
                
        else:
            if 'replicates' not in kargs.keys():
                BScores1 = self.Scores1.copy()
                i = 0
                smiB = numpy.zeros([self.ncomp1, self.ncomp2])
                while i < B:
                    numpy.random.shuffle(BScores1) # Permutation of rows
                    TU = numpy.dot(numpy.transpose(BScores1[:,0:self.ncomp1]),self.Scores2[:,0:self.ncomp2])
                    for p in range(self.ncomp1):
                        for q in range(self.ncomp2):
                            U, s, V = numpy.linalg.svd(TU[:p+1,:q+1])
                            smiB[p,q] = numpy.square(numpy.mean(s))
                    # Increase P-value if non-significant permutation
                    P[self.smi > numpy.maximum(smiB,1-smiB)] += 1
                    i += 1
            else:
                # With replicates
                AScores1 = self.Scores1.copy()
                BScores1 = self.Scores1.copy()
                i = 0
                smiB = numpy.zeros([self.ncomp1, self.ncomp2])
                replicates = kargs['replicates']
                uni = numpy.unique(replicates, return_inverse=True)
                vecOut = numpy.array(range(numpy.shape(uni[0])[0]))
                vecIn  = numpy.array(range(sum(uni[1]==0)))
                nOut = len(vecOut)
                nIn  = len(vecIn)
                while i < B:
                    numpy.random.shuffle(vecOut) # Permute across replicate sets
                    for j in range(nOut):
                        numpy.random.shuffle(vecIn) # Permute inside replicate sets
                        BScores1[uni[1]==j,:] = AScores1[vecOut[j]*(nIn)+vecIn,:]
                    TU = numpy.dot(numpy.transpose(BScores1[:,0:self.ncomp1]),self.Scores2[:,0:self.ncomp2])
                    for p in range(self.ncomp1):
                        for q in range(self.ncomp2):
                            U, s, V = numpy.linalg.svd(TU[:p+1,:q+1])
                            smiB[p,q] = numpy.square(numpy.mean(s))
                    # Increase P-value if non-significant permutation
                    P[self.smi > numpy.maximum(smiB,1-smiB)] += 1
                    i += 1                
            
        return P / B
