# -*- coding: utf-8 -*-

# Import necessary modules
import numpy as np
import numpy.linalg as npla
import hoggorm.statTools as st
import hoggorm.cross_val as cv




class nipalsPCA:
    """
    This class carries out Principal Component Analysis using the
    NIPALS algorithm.


    PARAMETERS
    ----------
    arrX : numpy array
        A numpy array containing the data

    numComp : int, optional
        An integer that defines how many components are to be computed

    Xstand : boolean, optional
        Defines whether variables in ``arrX`` are to be standardised/scaled or centered

        False : columns of ``arrX`` are mean centred (default)
            ``Xstand = False``

        True : columns of ``arrX`` are mean centred and devided by their own standard deviation
            ``Xstand = True``

    cvType : list, optional
        The list defines cross validation settings when computing the PCA model. Note if `cvType` is not provided, cross validation will not be performed and as such cross validation results will not be available. Choose cross validation type from the following:

        loo : leave one out / a.k.a. full cross validation (default)
            ``cvType = ["loo"]``

        KFold : leave out one fold or segment
            ``cvType = ["KFold", numFolds]``

            numFolds: int

            Number of folds or segments

        lolo : leave one label out
            ``cvType = ["lolo", lablesList]``

            lablesList: list

        Sequence of lables. Must be same lenght as number of rows in ``arrX``. Leaves out objects with same lable.


    RETURNS
    -------
    class
        A class that contains the PCA model and computational results


    EXAMPLES
    --------

    First import the hoggorm package.

    >>> import hoggorm as ho

    Import your data into a numpy array.

    >>> myData
    array([[ 5.7291665,  3.416667 ,  3.175    ,  2.6166668,  6.2208333],
           [ 6.0749993,  2.7416666,  3.6333339,  3.3833334,  6.1708336],
           [ 6.1166663,  3.4916666,  3.5208333,  2.7125003,  6.1625004],
           ...,
           [ 6.3333335,  2.3166668,  4.1249995,  4.3541665,  6.7500005],
           [ 5.8250003,  4.8291669,  1.4958333,  1.0958334,  6.0999999],
           [ 5.6499996,  4.6624999,  1.9291668,  1.0749999,  6.0249996]])
    >>> np.shape(myData)
    (14, 5)

    Examples of how to compute a PCA model using different settings for the input parameters.

    >>> model = ho.nipalsPCA(arrX=myData, numComp=5, Xstand=False)
    >>> model = ho.nipalsPCA(arrX=myData)
    >>> model = ho.nipalsPCA(arrX=myData, numComp=3)
    >>> model = ho.nipalsPCA(arrX=myData, Xstand=True)
    >>> model = ho.nipalsPCA(arrX=myData, cvType=["loo"])
    >>> model = ho.nipalsPCA(arrX=myData, cvType=["Kfold", 4])
    >>> model = ho.nipalsPCA(arrX=myData, cvType=["lolo", [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]])

    Examples of how to extract results from the PCA model.

    >>> scores = model.X_scores()
    >>> loadings = model.X_loadings()
    >>> cumulativeCalibratedExplainedVariance_allVariables = model.X_cumCalExplVar_indVar()

    """

    def __init__(self, arrX, numComp=None, Xstand=False, cvType=None):
        """
        On initialisation check how arrX and arrY are to be pre-processed
        (Xstand and Ystand are either True or False). Then check whether
        number of components chosen by user is OK.
        """

        # ===============================================================================
        #         Check what is provided by user
        # ===============================================================================

        # Check whether number of components that are to be computed is provided.
        # If NOT, then number of components is set to either number of objects or
        # variables of X whichever is smallest (numComp). If number of
        # components IS provided, then number is checked against maxPC and set to
        # numComp if provided number is larger.
        if numComp is None:
            self.numPC = min(np.shape(arrX))
        else:
            maxNumPC = min(np.shape(arrX))
            if numComp > maxNumPC:
                self.numPC = maxNumPC
            else:
                self.numPC = numComp


        # Define X and Y within class such that the data can be accessed from
        # all attributes in class.
        self.arrX_input = arrX


        # Pre-process data according to user request.
        # -------------------------------------------
        # Check whether standardisation of X and Y are requested by user. If
        # NOT, then X and y are centred by default.
        self.Xstand = Xstand


        # Standardise X if requested by user, otherwise center X.
        if self.Xstand:
            self.Xmeans = np.average(self.arrX_input, axis=0)
            self.Xstd = np.std(self.arrX_input, axis=0, ddof=1)
            self.arrX = (self.arrX_input - self.Xmeans) / self.Xstd
        else:
            self.Xmeans = np.average(self.arrX_input, axis=0)
            self.arrX = self.arrX_input - self.Xmeans


        # Check whether cvType is provided. If NOT, then no cross validation
        # is carried out.
        self.cvType = cvType


        # Before PLS2 NIPALS algorithm starts initiate and lists in which
        # results will be stored.
        self.X_scoresList = []
        self.X_loadingsList = []
        self.X_loadingsWeightsList = []
        self.coeffList = []
        self.X_residualsList = [self.arrX]


        # Collect residual matrices/arrays after each computed component
        self.resids = {}
        self.X_residualsDict = {}

        # Collect predicted matrices/array Xhat after each computed component
        self.calXhatDict_singPC = {}

        # Collect explained variance in each component
        self.calExplainedVariancesDict = {}
        self.X_calExplainedVariancesList = []


        # ===============================================================================
        #        Here the NIPALS PCA algorithm on X starts
        # ===============================================================================
        threshold = 1.0e-8
        X_new = self.arrX.copy()

        # Compute number of principal components as specified by user
        for j in range(self.numPC):

            # Check if first column contains only zeros. If yes, then
            # NIPALS will not converge and (npla.norm(num) will contain
            # nan's). Rather put in other starting values.
            if not np.any(X_new[:, 0]):
                X_repl_nonCent = np.arange(np.shape(X_new)[0])
                X_repl = X_repl_nonCent - np.mean(X_repl_nonCent)
                t = X_repl.reshape(-1,1)

            else:
                t = X_new[:,0].reshape(-1,1)

            # Iterate until score vector converges according to threshold
            while 1:
                num = np.dot(np.transpose(X_new), t)
                denom = npla.norm(num)

                p = num / denom
                t_new = np.dot(X_new, p)

                diff = t - t_new
                t = t_new.copy()
                SS = np.sum(np.square(diff))

                # Check whether sum of squares is smaller than threshold. Break
                # out of loop if true and start computation of next component.
                if SS < threshold:
                    self.X_scoresList.append(t)
                    self.X_loadingsList.append(p)
                    break

            # Peel off information explained by actual component and continue with
            # decomposition on the residuals (X_new = E).
            X_old = X_new.copy()
            Xhat_j = np.dot(t, np.transpose(p))
            X_new = X_old - Xhat_j

            # Store residuals E and Xhat in their dictionaries
            self.X_residualsDict[j+1] = X_new
            self.calXhatDict_singPC[j+1] = Xhat_j

            if self.Xstand:
                self.calXhatDict_singPC[j+1] = (Xhat_j * self.Xstd) + self.Xmeans
            else:
                self.calXhatDict_singPC[j+1] = Xhat_j + self.Xmeans


        # Collect scores and loadings for the actual component.
        self.arrT = np.hstack(self.X_scoresList)
        self.arrP = np.hstack(self.X_loadingsList)


        # ==============================================================================
        #         From here computation of CALIBRATED explained variance starts
        # ==============================================================================


        # ========== COMPUTATIONS FOR X ==========
        # ---------------------------------------------------------------------
        # Create a list holding arrays of Xhat predicted calibration after each
        # component. Xhat is computed with Xhat = T*P'
        self.calXpredList = []

        # Compute Xhat for 1 and more components (cumulatively).
        for ind in range(1,self.numPC+1):

            part_arrT = self.arrT[:,0:ind]
            part_arrP = self.arrP[:,0:ind]
            predXcal = np.dot(part_arrT, np.transpose(part_arrP))

            if self.Xstand:
                Xhat = (predXcal * self.Xstd) + self.Xmeans
            else:
                Xhat = predXcal + self.Xmeans
            self.calXpredList.append(Xhat)
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Collect all PRESSE for individual variables in a dictionary.
        # Keys represent number of component.
        self.PRESSEdict_indVar_X = {}

        # Compute PRESS for calibration / estimation
        PRESSE_0_indVar_X = np.sum(np.square(st.center(self.arrX_input)), axis=0)
        self.PRESSEdict_indVar_X[0] = PRESSE_0_indVar_X

        # Compute PRESS for each Xhat for 1, 2, 3, etc number of components
        # and compute explained variance
        for ind, Xhat in enumerate(self.calXpredList):
            diffX = self.arrX_input - Xhat
            PRESSE_indVar_X = np.sum(np.square(diffX), axis=0)
            self.PRESSEdict_indVar_X[ind+1] = PRESSE_indVar_X

        # Now store all PRESSE values into an array. Then compute MSEE and
        # RMSEE.
        self.PRESSEarr_indVar_X = np.array(list(self.PRESSEdict_indVar_X.values()))
        self.MSEEarr_indVar_X = self.PRESSEarr_indVar_X / np.shape(self.arrX_input)[0]
        self.RMSEEarr_indVar_X = np.sqrt(self.MSEEarr_indVar_X)
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Compute explained variance for each variable in X using the
        # MSEE for each variable. Also collect PRESSE, MSEE, RMSEE in
        # their respective dictionaries for each variable. Keys represent
        # now variables and NOT components as above with
        # self.PRESSEdict_indVar_X
        self.cumCalExplVarXarr_indVar = np.zeros(np.shape(self.MSEEarr_indVar_X))
        MSEE_0_indVar_X = self.MSEEarr_indVar_X[0,:]

        for ind, MSEE_indVar_X in enumerate(self.MSEEarr_indVar_X):
            explVar = (MSEE_0_indVar_X - MSEE_indVar_X) / MSEE_0_indVar_X * 100
            self.cumCalExplVarXarr_indVar[ind] = explVar

        self.PRESSE_indVar_X = {}
        self.MSEE_indVar_X = {}
        self.RMSEE_indVar_X = {}
        self.cumCalExplVarX_indVar = {}

        for ind in range(np.shape(self.PRESSEarr_indVar_X)[1]):
            self.PRESSE_indVar_X[ind] = self.PRESSEarr_indVar_X[:,ind]
            self.MSEE_indVar_X[ind] = self.MSEEarr_indVar_X[:,ind]
            self.RMSEE_indVar_X[ind] = self.RMSEEarr_indVar_X[:,ind]
            self.cumCalExplVarX_indVar[ind] = self.cumCalExplVarXarr_indVar[:,ind]
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Collect total PRESSE across all variables in a dictionary. Also,
        # compute total calibrated explained variance in X.
        self.PRESSE_total_dict_X = {}
        self.PRESSE_total_list_X = np.sum(self.PRESSEarr_indVar_X, axis=1)

        for ind, PRESSE_X in enumerate(self.PRESSE_total_list_X):
            self.PRESSE_total_dict_X[ind] = PRESSE_X
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Collect total MSEE across all variables in a dictionary. Also,
        # compute total validated explained variance in X.
        self.MSEE_total_dict_X = {}
        self.MSEE_total_list_X = np.sum(self.MSEEarr_indVar_X, axis=1) / np.shape(self.arrX_input)[1]
        MSEE_0_X = self.MSEE_total_list_X[0]

        # Compute total cumulated calibrated explained variance in X
        self.XcumCalExplVarList = []
        if not self.Xstand:
            for ind, MSEE_X in enumerate(self.MSEE_total_list_X):
                perc = (MSEE_0_X - MSEE_X) / MSEE_0_X * 100
                self.MSEE_total_dict_X[ind] = MSEE_X
                self.XcumCalExplVarList.append(perc)
        else:
            self.XcumCalExplVarArr = np.average(self.cumCalExplVarXarr_indVar, axis=1)
            self.XcumCalExplVarList = list(self.XcumCalExplVarArr)

        # Construct list with total explained variance in X for each component
        self.XcalExplVarList = []
        for ind, item in enumerate(self.XcumCalExplVarList):
            if ind == len(self.XcumCalExplVarList)-1:
                break
            explVarComp = self.XcumCalExplVarList[ind+1] - self.XcumCalExplVarList[ind]
            self.XcalExplVarList.append(explVarComp)

        # Construct a dictionary that holds predicted X (Xhat) from calibration
        # for each number of components.
        self.calXpredDict = {}
        for ind, item in enumerate(self.calXpredList):
            self.calXpredDict[ind+1] = item
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Compute total RMSEE and store values in a dictionary and list.
        self.RMSEE_total_dict_X = {}
        self.RMSEE_total_list_X = np.sqrt(self.MSEE_total_list_X)

        for ind, RMSEE_X in enumerate(self.RMSEE_total_list_X):
            self.RMSEE_total_dict_X[ind] = RMSEE_X
        # ---------------------------------------------------------------------


        # ==============================================================================
        #         From here cross validation procedure starts
        # ==============================================================================
        if self.cvType is not None:
            numObj = np.shape(self.arrX)[0]

            if self.cvType[0] == "loo":
                print("loo")
                cvComb = cv.LeaveOneOut(numObj)
            elif self.cvType[0] == "KFold":
                print("KFold")
                cvComb = cv.KFold(numObj, k=self.cvType[1])
            elif self.cvType[0] == "lolo":
                print("lolo")
                cvComb = cv.LeaveOneLabelOut(self.cvType[1])
            else:
                print('Requested form of cross validation is not available')


            # Collect predicted x (i.e. xhat) for each CV segment in a
            # dictionary according to number of component
            self.valXpredDict = {}
            for ind in range(1, self.numPC+1):
                self.valXpredDict[ind] = np.zeros(np.shape(self.arrX_input))


            # Collect: validation X scores T, validation X loadings P,
            # validation Y scores U, validation Y loadings Q,
            # validation X loading weights W and scores regression coefficients C
            # in lists for each component
            self.val_arrTlist = []
            self.val_arrPlist = []
            self.val_arrQlist = []

            # Collect train and test set in a dictionary for each component
            self.cvTrainAndTestDataList = []
            self.X_train_means_arr = np.zeros(np.shape(self.arrX_input))

            # First devide into combinations of training and test sets
            for train_index, test_index in cvComb:
                X_train, X_test = cv.split(train_index, test_index, self.arrX_input)

                subDict = {}
                subDict['x train'] = X_train
                subDict['x test'] = X_test
                self.cvTrainAndTestDataList.append(subDict)

                # -------------------------------------------------------------
                # Center or standardise X according to users choice
                if self.Xstand:
                    X_train_mean = np.average(X_train, axis=0).reshape(1,-1)
                    X_train_std = np.std(X_train, axis=0, ddof=1).reshape(1,-1)
                    X_train_proc = (X_train - X_train_mean) / X_train_std

                    # Standardise X test using mean and STD from training set
                    X_test_proc = (X_test - X_train_mean) / X_train_std

                else:
                    X_train_mean = np.average(X_train, axis=0).reshape(1,-1)
                    X_train_proc = X_train - X_train_mean

                    # Center X test using mean from training set
                    X_test_proc = X_test - X_train_mean
                # -------------------------------------------------------------
                self.X_train_means_arr[test_index,] = X_train_mean


                # Here the NIPALS PCA algorithm starts
                # ------------------------------------
                threshold = 1.0e-8
                X_new = X_train_proc.copy()

                # Collect scores and loadings in lists that will be later converted
                # to arrays.
                scoresList = []
                loadingsList = []

                # Compute number of principal components as specified by user
                for j in range(self.numPC):

                    # Check if first column contains only zeros. If yes, then
                    # NIPALS will not converge and (npla.norm(num) will contain
                    # nan's). Rather put in other starting values.
                    if not np.any(X_new[:, 0]):
                        X_repl_nonCent = np.arange(np.shape(X_new)[0])
                        X_repl = X_repl_nonCent - np.mean(X_repl_nonCent)
                        t = X_repl.reshape(-1,1)

                    else:
                        t = X_new[:,0].reshape(-1,1)

                    # Iterate until score vector converges according to threshold
                    while 1:
                        num = np.dot(np.transpose(X_new), t)
                        denom = npla.norm(num)

                        p = num / denom
                        t_new = np.dot(X_new, p)

                        diff = t - t_new
                        t = t_new.copy()
                        SS = np.sum(np.square(diff))

                        # Check whether sum of squares is smaller than threshold. Break
                        # out of loop if true and start computation of next component.
                        if SS < threshold:
                            scoresList.append(t)
                            loadingsList.append(p)
                            break

                    # Peel off information explained by actual component and continue with
                    # decomposition on the residuals (X_new = E).
                    X_old = X_new.copy()
                    Xhat_j = np.dot(t, np.transpose(p))
                    X_new = X_old - Xhat_j

                # Collect X scores and X loadings for the actual component.
                valT = np.hstack(scoresList)
                valP = np.hstack(loadingsList)

                self.val_arrTlist.append(valT)
                self.val_arrPlist.append(valP)


                # Compute the scores for the left out object
                projT = np.dot(X_test_proc, valP)
                dims = np.shape(projT)[1]

                # Construct validated predicted X first for one component,
                # then two, three, etc
                for ind in range(0, dims):

                    part_projT = projT[:, 0:ind+1]
                    part_valP = valP[:, 0:ind+1]
                    valPredX_proc = np.dot(part_projT, np.transpose(part_valP))


                    # Depending on preprocessing re-process in same manner
                    # in order to get values that compare to original values.
                    if self.Xstand:
                        valPredX = (valPredX_proc * X_train_std) + X_train_mean
                    else:
                        valPredX = valPredX_proc + X_train_mean

                    self.valXpredDict[ind+1][test_index, :] = valPredX


            # Put all predicitons into an array that corresponds to the
            # original array
            self.valXpredList = []
            valPreds = self.valXpredDict.values()
            for preds in valPreds:
                pc_arr = np.vstack(preds)
                self.valXpredList.append(pc_arr)


            # ==============================================================================
            # From here VALIDATED explained variance is computed
            # ==============================================================================

            # ========== Computations for X ==========
            # -----------------------------------------------------------------
            # Compute PRESSCV (PRediction Error Sum of Squares) for cross
            # validation
            self.valXpredList = self.valXpredDict.values()

            # Collect all PRESSCV in a dictionary. Keys represent number of
            # component.
            self.PRESSCVdict_indVar_X = {}

            # First compute PRESSCV for zero components
            self.PRESSCV_0_indVar_X = np.sum(np.square(self.arrX_input - self.X_train_means_arr), axis=0)
            self.PRESSCVdict_indVar_X[0] = self.PRESSCV_0_indVar_X

            # Compute PRESSCV for each Yhat for 1, 2, 3, etc number of
            # components and compute explained variance
            for ind, Xhat in enumerate(self.valXpredList):
                # diffX = self.arrX_input - Xhat
                diffX = self.arrX_input - Xhat
                PRESSCV_indVar_X = np.sum(np.square(diffX), axis=0)
                self.PRESSCVdict_indVar_X[ind+1] = PRESSCV_indVar_X

            # Now store all PRESSCV values into an array. Then compute MSECV
            # and RMSECV.
            self.PRESSCVarr_indVar_X = np.array(list(self.PRESSCVdict_indVar_X.values()))
            self.MSECVarr_indVar_X = self.PRESSCVarr_indVar_X / np.shape(self.arrX_input)[0]
            self.RMSECVarr_indVar_X = np.sqrt(self.MSECVarr_indVar_X)
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Compute explained variance for each variable in X using the
            # MSEP for each variable. Also collect PRESSCV, MSECV, RMSECV in
            # their respective dictionaries for each variable. Keys represent
            # now variables and NOT components as above with
            # self.PRESSCVdict_indVar
            self.cumValExplVarXarr_indVar = np.zeros(np.shape(self.MSECVarr_indVar_X))
            MSECV_0_indVar_X = self.MSECVarr_indVar_X[0,:]

            for ind, MSECV_indVar_X in enumerate(self.MSECVarr_indVar_X):
                explVar = (MSECV_0_indVar_X - MSECV_indVar_X) / MSECV_0_indVar_X * 100
                self.cumValExplVarXarr_indVar[ind] = explVar

            self.PRESSCV_indVar_X = {}
            self.MSECV_indVar_X = {}
            self.RMSECV_indVar_X = {}
            self.cumValExplVarX_indVar = {}

            for ind in range(np.shape(self.PRESSCVarr_indVar_X)[1]):
                self.PRESSCV_indVar_X[ind] = self.PRESSCVarr_indVar_X[:,ind]
                self.MSECV_indVar_X[ind] = self.MSECVarr_indVar_X[:,ind]
                self.RMSECV_indVar_X[ind] = self.RMSECVarr_indVar_X[:,ind]
                self.cumValExplVarX_indVar[ind] = self.cumValExplVarXarr_indVar[:,ind]
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Collect total PRESSCV across all variables in a dictionary.
            self.PRESSCV_total_dict_X = {}
            self.PRESSCV_total_list_X = np.sum(self.PRESSCVarr_indVar_X, axis=1)

            for ind, PRESSCV_X in enumerate(self.PRESSCV_total_list_X):
                self.PRESSCV_total_dict_X[ind] = PRESSCV_X
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Collect total MSECV across all variables in a dictionary. Also,
            # compute total validated explained variance in X.
            self.MSECV_total_dict_X = {}
            self.MSECV_total_list_X = np.sum(self.MSECVarr_indVar_X, axis=1) / np.shape(self.arrX_input)[1]
            MSECV_0_X = self.MSECV_total_list_X[0]

            # Compute total validated explained variance in X
            self.XcumValExplVarList = []
            if not self.Xstand:
                for ind, MSECV_X in enumerate(self.MSECV_total_list_X):
                    perc = (MSECV_0_X - MSECV_X) / MSECV_0_X * 100
                    self.MSECV_total_dict_X[ind] = MSECV_X
                    self.XcumValExplVarList.append(perc)
            else:
                self.XcumValExplVarArr = np.average(self.cumValExplVarXarr_indVar, axis=1)
                self.XcumValExplVarList = list(self.XcumValExplVarArr)

            # Construct list with total validated explained variance in X in
            # each component
            self.XvalExplVarList = []
            for ind, item in enumerate(self.XcumValExplVarList):
                if ind == len(self.XcumValExplVarList)-1:
                    break
                explVarComp = self.XcumValExplVarList[ind+1] - self.XcumValExplVarList[ind]
                self.XvalExplVarList.append(explVarComp)
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Compute total RMSECV and store values in a dictionary and list.
            self.RMSECV_total_dict_X = {}
            self.RMSECV_total_list_X = np.sqrt(self.MSECV_total_list_X)

            for ind, RMSECV_X in enumerate(self.RMSECV_total_list_X):
                self.RMSECV_total_dict_X[ind] = RMSECV_X
            # -----------------------------------------------------------------


    def modelSettings(self):
        """
        Returns a dictionary holding the settings under which NIPALS PCA was
        run.
        """
        # Collect settings under which PCA was run.
        self.settings = {}
        self.settings['numComp'] = self.numPC
        self.settings['Xstand'] = self.Xstand
        self.settings['arrX'] = self.arrX_input
        self.settings['analysed arrX'] = self.arrX

        return self.settings


    def X_means(self):
        """
        Returns array holding the column means of input array X.
        """
        return self.Xmeans.reshape(1,-1)


    def X_scores(self):
        """
        Returns array holding scores T. First column holds scores for
        component 1, second column holds scores for component 2, etc.
        """
        return self.arrT


    def X_loadings(self):
        """
        Returns array holding loadings P of array X. Rows represent variables
        and columns represent components. First column holds loadings for
        component 1, second column holds scores for component 2, etc.
        """
        return self.arrP


    def X_corrLoadings(self):
        """
        Returns array holding correlation loadings of array X. First column
        holds correlation loadings for component 1, second column holds
        correlation loadings for component 2, etc.
        """

        # Creates empty matrix for correlation loadings
        arr_corrLoadings = np.zeros((np.shape(self.arrT)[1],
                                     np.shape(self.arrP)[0]), float)

        # Compute correlation loadings:
        # For each component in score matrix
        for PC in range(np.shape(self.arrT)[1]):
            PCscores = self.arrT[:, PC]

            # For each variable/attribute in original matrix (not meancentered)
            for var in range(np.shape(self.arrX)[1]):
                origVar = self.arrX[:, var]
                corrs = np.corrcoef(PCscores, origVar)
                arr_corrLoadings[PC, var] = corrs[0, 1]

        self.arr_corrLoadings = np.transpose(arr_corrLoadings)

        return self.arr_corrLoadings


    def X_residuals(self):
        """
        Returns a dictionary holding arrays of residuals for array X after
        each computed component. Dictionary key represents order of component.
        """
        return self.X_residualsDict


    def X_calExplVar(self):
        """
        Returns a list holding the calibrated explained variance for
        each component. First number in list is for component 1, second number
        for component 2, etc.
        """
        return self.XcalExplVarList


    def X_cumCalExplVar_indVar(self):
        """
        Returns an array holding the cumulative calibrated explained variance
        for each variable in X after each component. First row represents zero
        components, second row represents one component, third row represents
        two components, etc. Columns represent variables.
        """
        return self.cumCalExplVarXarr_indVar


    def X_cumCalExplVar(self):
        """
        Returns a list holding the cumulative validated explained variance
        for array X after each component. First number represents zero
        components, second number represents component 1, etc.
        """
        return self.XcumCalExplVarList


    def X_predCal(self):
        """
        Returns a dictionary holding the predicted arrays Xhat from
        calibration after each computed component. Dictionary key represents
        order of component.
        """
        return self.calXpredDict


    def X_PRESSE_indVar(self):
        """
        Returns array holding PRESSE for each individual variable in X
        acquired through calibration after each computed component. First row
        is PRESSE for zero components, second row for component 1, third row
        for component 2, etc.
        """
        return self.PRESSEarr_indVar_X


    def X_PRESSE(self):
        """
        Returns array holding PRESSE across all variables in X acquired
        through calibration after each computed component. First row is PRESSE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.PRESSE_total_list_X


    def X_MSEE_indVar(self):
        """
        Returns an array holding MSEE for each variable in array X acquired
        through calibration after each computed component. First row holds MSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.MSEEarr_indVar_X


    def X_MSEE(self):
        """
        Returns an array holding MSEE across all variables in X acquired
        through calibration after each computed component. First row is MSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.MSEE_total_list_X


    def X_RMSEE_indVar(self):
        """
        Returns an array holding RMSEE for each variable in array X acquired
        through calibration after each components. First row holds RMSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.RMSEEarr_indVar_X


    def X_RMSEE(self):
        """
        Returns an array holding RMSEE across all variables in X acquired
        through calibration after each computed component. First row is RMSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.RMSEE_total_list_X


    def X_valExplVar(self):
        """
        Returns a list holding the validated explained variance for X after
        each component. First number in list is for component 1, second number
        for component 2, third number for component 3, etc.
        """
        return self.XvalExplVarList


    def X_cumValExplVar_indVar(self):
        """
        Returns an array holding the cumulative validated explained variance
        for each variable in X after each component. First row represents
        zero components, second row represents component 1, third row for
        compnent 2, etc. Columns represent variables.
        """
        return self.cumValExplVarXarr_indVar


    def X_cumValExplVar(self):
        """
        Returns a list holding the cumulative validated explained variance
        for array X after each component.
        """
        return self.XcumValExplVarList


    def X_predVal(self):
        """
        Returns a dictionary holding the predicted arrays Xhat from
        validation after each computed component. Dictionary key represents
        order of component.
        """
        return self.valXpredDict


    def X_PRESSCV_indVar(self):
        """
        Returns array holding PRESSEV for each individual variable in X
        acquired through cross validation after each computed component. First
        row is PRESSCV for zero components, second row for component 1, third
        row for component 2, etc.
        """
        return self.PRESSCVarr_indVar_X


    def X_PRESSCV(self):
        """
        Returns an array holding PRESSCV across all variables in X acquired
        through cross validation after each computed component. First row is
        PRESSEV for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.PRESSCV_total_list_X


    def X_MSECV_indVar(self):
        """
        Returns an arrary holding MSECV for each variable in X acquired through
        cross validation. First row is MSECV for zero components, second row
        for component 1, etc.
        """
        return self.MSECVarr_indVar_X


    def X_MSECV(self):
        """
        Returns an array holding MSECV across all variables in X acquired
        through cross validation after each computed component. First row is
        MSECV for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.MSECV_total_list_X


    def X_RMSECV_indVar(self):
        """
        Returns an arrary holding RMSECV for each variable in X acquired
        through cross validation after each computed component. First row is
        RMSECV for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.RMSECVarr_indVar_X


    def X_RMSECV(self):
        """
        Returns an array holding RMSECV across all variables in X acquired
        through cross validation after each computed component. First row is
        RMSECV for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.RMSECV_total_list_X


    def X_scores_predict(self, Xnew, numComp=[]):
        """
        Returns array of X scores from new X data using the exsisting model.
        Rows represent objects and columns represent components.
        """

        if len(numComp) == 0:
            numComp = self.numPC
        assert numComp <= self.numPC, ValueError('Maximum numComp = ' + str(self.numPC))
        assert numComp > -1, ValueError('numComp must be >= 0')

        # First pre-process new X data accordingly
        if self.Xstand:

            x_new = (Xnew - np.average(self.arrX_input, axis=0)) / np.std(self.arrX_input, ddof=1)

        else:

            x_new = (Xnew - np.average(self.arrX_input, axis=0))

        # Compute the scores for new object
        projT = np.dot(x_new, self.arrP[:, 0:numComp])

        return projT


    def cvTrainAndTestData(self):
        """
        Returns a list consisting of dictionaries holding training and test
        sets.
        """
        return self.cvTrainAndTestDataList


    def corrLoadingsEllipses(self):
        """
        Returns a dictionary hodling coordinates of ellipses that represent
        50% and 100% expl. variance in correlation loadings plot. The
        coordinates are stored in arrays.
        """
        # Create range for ellipses
        t = np.arange(0.0, 2*np.pi, 0.01)

        # Compuing the outer circle (100 % expl. variance)
        xcords100perc = np.cos(t)
        ycords100perc = np.sin(t)

        # Computing inner circle
        xcords50perc = 0.707 * np.cos(t)
        ycords50perc = 0.707 * np.sin(t)

        # Collect ellipse coordinates in dictionary
        ellipses = {}
        ellipses['x50perc'] = xcords50perc
        ellipses['y50perc'] = ycords50perc

        ellipses['x100perc'] = xcords100perc
        ellipses['y100perc'] = ycords100perc

        return ellipses
