# -*- coding: utf-8 -*-

# Import necessary modules
import numpy as np
import numpy.linalg as npla
import hoggorm.statTools as st
import hoggorm.cross_val as cv




class nipalsPCR:
    """
    This class carries out Principal Component Regression for two arrays using NIPALS algorithm.


    PARAMETERS
    ----------
    arrX : numpy array
        This is X in the PCR model. Number and order of objects (rows) must match those of ``arrY``.

    arrY : numpy array
        This is Y in the PCR model. Number and order of objects (rows) must match those of ``arrX``.

    numComp : int, optional
        An integer that defines how many components are to be computed. If not provided, the maximum possible number of components is used.

    Xstand : boolean, optional
        Defines whether variables in ``arrX`` are to be standardised/scaled or centered.

        False : columns of ``arrX`` are mean centred (default)
            ``Xstand = False``

        True : columns of ``arrX`` are mean centred and devided by their own standard deviation
            ``Xstand = True``

    Ystand : boolean, optional
        Defines whether variables in ``arrY`` are to be standardised/scaled or centered.

        False : columns of ``arrY`` are mean centred (default)
            ``Ystand = False``

        True : columns of ``arrY`` are mean centred and devided by their own standard deviation
            ``Ystand = True``

    cvType : list, optional
        The list defines cross validation settings when computing the PCA model. Note if `cvType` is not provided, cross validation will not be performed and as such cross validation results will not be available. Choose cross validation type from the following:

        loo : leave one out / a.k.a. full cross validation (default)
            ``cvType = ["loo"]``

        KFold : leave out one fold or segment
            ``cvType = ["KFold", numFolds]``

            numFolds: int

            Number of folds or segments

    lolo : leave one label out
            ``cvType = ["lolo", labelsList]``

            labelsList: list

            Sequence of lables. Must be same lenght as number of rows in ``arrX`` and ``arrY``. Leaves out objects with same lable.


    RETURNS
    -------
    class
        A class that contains the PCR model and computational results


    EXAMPLES
    --------

    First import the hoggormpackage

    >>> import hoggorm as ho

    Import your data into a numpy array.

    >>> np.shape(my_X_data)
    (14, 292)
    >>> np.shape(my_Y_data)
    (14, 5)

    Examples of how to compute a PCR model using different settings for the input parameters.

    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data, numComp=5)
    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data)
    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data, numComp=3, Ystand=True)
    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data, Xstand=False, Ystand=True)
    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data, cvType=["loo"])
    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data, cvType=["KFold", 7])
    >>> model = ho.nipalsPCR(arrX=my_X_data, arrY=my_Y_data, cvType=["lolo", [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]])

    Examples of how to extract results from the PCR model.

    >>> X_scores = model.X_scores()
    >>> X_loadings = model.X_loadings()
    >>> Y_loadings = model.Y_loadings()
    >>> X_cumulativeCalibratedExplainedVariance_allVariables = model.X_cumCalExplVar_indVar()
    >>> Y_cumulativeValidatedExplainedVariance_total = model.Y_cumCalExplVar()

    """

    def __init__(self, arrX, arrY, numComp=None, Xstand=False, Ystand=False, cvType=None):
        """
        On initialisation check how arrX and arrY are to be pre-processed
        (parameters Xstand and Ystand are either True or False). Then check
        whether number of components chosen by user is OK.
        """

        # ===============================================================================
        #         Check what is provided by user for PCA-part of PCR
        # ===============================================================================

        # Check whether number of PC's that are to be computed is provided.
        # If NOT, then number of PC's is set to either number of objects or
        # variables of X whichever is smallest (numPC). If number of
        # PC's IS provided, then number is checked against maxcomponent and set to
        # numcomponent if provided number is larger.
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
        self.arrY_input = arrY


        # Pre-process data according to user request.
        # -------------------------------------------
        # Check whether standardisation of X and Y are requested by user. If
        # NOT, then X and y are centred by default.
        self.Xstand = Xstand
        self.Ystand = Ystand


        # Standardise X if requested by user, otherwise center X.
        if self.Xstand:
            self.Xmeans = np.average(self.arrX_input, axis=0)
            self.Xstd = np.std(self.arrX_input, axis=0, ddof=1)
            self.arrX = (self.arrX_input - self.Xmeans) / self.Xstd
        else:
            self.Xmeans = np.average(self.arrX_input, axis=0)
            self.arrX = self.arrX_input - self.Xmeans


        # Standardise Y if requested by user, otherwise center Y.
        if self.Ystand:
            self.Ymeans = np.average(self.arrY_input, axis=0)
            self.Ystd = np.std(self.arrY_input, axis=0, ddof=1)
            self.arrY = (self.arrY_input - self.Ymeans) / self.Ystd
        else:
            self.Ymeans = np.average(self.arrY_input, axis=0)
            self.arrY = self.arrY_input - self.Ymeans


        # Check whether cvType is provided. If NOT, then no cross validation
        # is carried out.
        self.cvType = cvType


        # Before PLS2 NIPALS algorithm starts initiate and lists in which
        # results will be stored.
        self.X_scoresList = []
        self.Y_scoresList = []
        self.X_loadingsList = []
        self.Y_loadingsList = []
        self.X_loadingsWeightsList = []
        self.coeffList = []
        self.Y_residualsList = [self.arrY]
        self.X_residualsList = [self.arrX]


        # Collect residual matrices/arrays after each computed PC
        self.resids = {}
        self.X_residualsDict = {}

        # Collect predicted matrices/array Xhat after each computed PC
        self.calXhatDict_singPC = {}

        # Collect explained variance in each PC
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
                # out of loop if true and start computation of next PC.
                if SS < threshold:
                    self.X_scoresList.append(t)
                    self.X_loadingsList.append(p)
                    break

            # Peel off information explained by actual componentand continue with
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


        # Collect scores and loadings for the actual PC.
        self.arrT = np.hstack(self.X_scoresList)
        self.arrP = np.hstack(self.X_loadingsList)

        # Compute Y loadings by using MLR (see Module 6, Equ. 6.8 ++)
        term_1 = npla.inv(np.dot(np.transpose(self.arrT), self.arrT))
        term_2 = np.dot(np.transpose(self.arrT), self.arrY)
        self.arrQ = np.transpose(np.dot(term_1, term_2))


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
        # MSEP for each variable. Also collect PRESSE, MSEE, RMSEE in
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
        # compute total calibrated explained variance in Y.
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

        # Compute total calibrated explained variance in X
        self.XcumCalExplVarList = []
        if not self.Xstand:
            for ind, MSEE_X in enumerate(self.MSEE_total_list_X):
                perc = (MSEE_0_X - MSEE_X) / MSEE_0_X * 100
                self.MSEE_total_dict_X[ind] = MSEE_X
                self.XcumCalExplVarList.append(perc)
        else:
            self.XcumCalExplVarArr = np.average(self.cumCalExplVarXarr_indVar, axis=1)
            self.XcumCalExplVarList = list(self.XcumCalExplVarArr)

        # Construct list with total validated explained variance in X
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



        # ========== COMPUTATIONS FOR Y ============
        # ---------------------------------------------------------------------
        # Create a list holding arrays of Yhat predicted calibration after each
        # component. Yhat is computed with Yhat = T*Chat*Q'
        self.calYpredList = []

        for ind in range(1, self.numPC+1):

            x_scores = self.arrT[:,0:ind]
            y_loadings = self.arrQ[:,0:ind]
            # c_regrCoeff = self.arrC[0:ind,0:ind]

            # Depending on whether Y was standardised or not compute Yhat
            # accordingly.
            if self.Ystand:
                Yhat_stand = np.dot(x_scores, np.transpose(y_loadings))
                Yhat = (Yhat_stand * self.Ystd.reshape(1,-1)) + self.Ymeans.reshape(1,-1)
            else:
                Yhat = np.dot(x_scores, np.transpose(y_loadings)) + self.Ymeans.reshape(1,-1)
            self.calYpredList.append(Yhat)

            # Compute Y residuals and store in list
            self.Y_residualsList.append(self.arrY - Yhat)
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Collect all PRESSE for individual variables in a dictionary.
        # Keys represent number of component.
        self.PRESSEdict_indVar = {}

        # Compute PRESS for calibration / estimation
        PRESSE_0_indVar = np.sum(np.square(st.center(self.arrY_input)), axis=0)
        self.PRESSEdict_indVar[0] = PRESSE_0_indVar

        # Compute PRESS for each Yhat for 1, 2, 3, etc number of components
        # and compute explained variance
        for ind, Yhat in enumerate(self.calYpredList):
            diffY = self.arrY_input - Yhat
            PRESSE_indVar = np.sum(np.square(diffY), axis=0)
            self.PRESSEdict_indVar[ind+1] = PRESSE_indVar

        # Now store all PRESSE values into an array. Then compute MSEE and
        # RMSEE.
        self.PRESSEarr_indVar = np.array(list(self.PRESSEdict_indVar.values()))
        self.MSEEarr_indVar = self.PRESSEarr_indVar / np.shape(self.arrY_input)[0]
        self.RMSEEarr_indVar = np.sqrt(self.MSEEarr_indVar)
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Compute explained variance for each variable in Y using the
        # MSEP for each variable. Also collect PRESSE, MSEE, RMSEE in
        # their respective dictionaries for each variable. Keys represent
        # now variables and NOT components as above with
        # self.PRESSEdict_indVar
        self.cumCalExplVarYarr_indVar = np.zeros(np.shape(self.MSEEarr_indVar))
        MSEE_0_indVar = self.MSEEarr_indVar[0,:]

        for ind, MSEE_indVar in enumerate(self.MSEEarr_indVar):
            explVar = (MSEE_0_indVar - MSEE_indVar) / MSEE_0_indVar * 100
            self.cumCalExplVarYarr_indVar[ind] = explVar

        self.PRESSE_indVar = {}
        self.MSEE_indVar = {}
        self.RMSEE_indVar = {}
        self.cumCalExplVarY_indVar = {}

        for ind in range(np.shape(self.PRESSEarr_indVar)[1]):
            self.PRESSE_indVar[ind] = self.PRESSEarr_indVar[:,ind]
            self.MSEE_indVar[ind] = self.MSEEarr_indVar[:,ind]
            self.RMSEE_indVar[ind] = self.RMSEEarr_indVar[:,ind]
            self.cumCalExplVarY_indVar[ind] = self.cumCalExplVarYarr_indVar[:,ind]
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Collect total PRESSE across all variables in a dictionary. Also,
        # compute total calibrated explained variance in Y.
        self.PRESSE_total_dict = {}
        self.PRESSE_total_list = np.sum(self.PRESSEarr_indVar, axis=1)

        for ind, PRESSE in enumerate(self.PRESSE_total_list):
            self.PRESSE_total_dict[ind] = PRESSE
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Collect total MSEE across all variables in a dictionary. Also,
        # compute total calibrated explained variance in Y.
        self.MSEE_total_dict = {}
        self.MSEE_total_list = np.sum(self.MSEEarr_indVar, axis=1) / np.shape(self.arrY_input)[1]
        MSEE_0 = self.MSEE_total_list[0]

        # Compute total calibrated explained variance in Y
        self.YcumCalExplVarList = []
        if not self.Ystand:
            for ind, MSEE in enumerate(self.MSEE_total_list):
                perc = (MSEE_0 - MSEE) / MSEE_0 * 100
                self.MSEE_total_dict[ind] = MSEE
                self.YcumCalExplVarList.append(perc)
        else:
            self.YcumCalExplVarArr = np.average(self.cumCalExplVarYarr_indVar, axis=1)
            self.YcumCalExplVarList = list(self.YcumCalExplVarArr)

        # Construct list with total validated explained variance in Y
        self.YcalExplVarList = []
        for ind, item in enumerate(self.YcumCalExplVarList):
            if ind == len(self.YcumCalExplVarList)-1:
                break
            explVarComp = self.YcumCalExplVarList[ind+1] - self.YcumCalExplVarList[ind]
            self.YcalExplVarList.append(explVarComp)

        # Construct a dictionary that holds predicted Y (Yhat) from calibration
        # for each number of components.
        self.calYpredDict = {}
        for ind, item in enumerate(self.calYpredList):
            self.calYpredDict[ind+1] = item
        # ---------------------------------------------------------------------


        # ---------------------------------------------------------------------
        # Compute total RMSEP and store values in a dictionary and list.
        self.RMSEE_total_dict = {}
        self.RMSEE_total_list = np.sqrt(self.MSEE_total_list)

        for ind, RMSEE in enumerate(self.RMSEE_total_list):
            self.RMSEE_total_dict[ind] = RMSEE
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


            # Collect predicted y (i.e. yhat) for each CV segment in a
            # dictionary according to nubmer of PC
            self.valYpredDict = {}
            for ind in range(1, self.numPC+1):
                self.valYpredDict[ind] = np.zeros(np.shape(self.arrY_input))

            # Collect predicted x (i.e. xhat) for each CV segment in a
            # dictionary according to number of PC
            self.valXpredDict = {}
            for ind in range(1, self.numPC+1):
                self.valXpredDict[ind] = np.zeros(np.shape(self.arrX_input))

            # Collect train and test set in dictionaries for each componentand put
            # them in this list.
            self.cvTrainAndTestDataList = []


            # Collect: validation X scores T, validation X loadings P,
            # validation Y scores U, validation Y loadings Q,
            # validation X loading weights W and scores regression coefficients C
            # in lists for each PC
            self.val_arrTlist = []
            self.val_arrPlist = []
            self.val_arrQlist = []

            # Collect train and test set in a dictionary for each component
            self.cvTrainAndTestDataList = []
            self.X_train_means_list = np.zeros(np.shape(self.arrX_input))
            self.Y_train_means_list = np.zeros(np.shape(self.arrY_input))

            # First devide into combinations of training and test sets
            for train_index, test_index in cvComb:
                X_train, X_test = cv.split(train_index, test_index, self.arrX_input)
                Y_train, Y_test = cv.split(train_index, test_index, self.arrY_input)

                subDict = {}
                subDict['x train'] = X_train
                subDict['x test'] = X_test
                subDict['y train'] = Y_train
                subDict['y test'] = Y_test
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
                self.X_train_means_list[test_index,] = X_train_mean


                # -------------------------------------------------------------
                # Center or standardise Y according to users choice
                if self.Ystand:
                    Y_train_mean = np.average(Y_train, axis=0)
                    Y_train_std = np.std(Y_train, axis=0, ddof=1)
                    Y_train_proc = (Y_train - Y_train_mean) / Y_train_std
                else:
                    Y_train_mean = np.average(Y_train, axis=0)
                    Y_train_proc = Y_train - Y_train_mean
                # -------------------------------------------------------------
                self.Y_train_means_list[test_index,] = Y_train_mean


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
                        # out of loop if true and start computation of next PC.
                        if SS < threshold:
                            scoresList.append(t)
                            loadingsList.append(p)
                            break

                    # Peel off information explained by actual component and continue with
                    # decomposition on the residuals (X_new = E).
                    X_old = X_new.copy()
                    Xhat_j = np.dot(t, np.transpose(p))
                    X_new = X_old - Xhat_j

                # Collect X scores and X loadings for the actual PC.
                valT = np.hstack(scoresList)
                valP = np.hstack(loadingsList)

                self.val_arrTlist.append(valT)
                self.val_arrPlist.append(valP)

                # Compute Y loadings
                term_1 = npla.inv(np.dot(np.transpose(valT),valT))
                term_2 = np.dot(np.transpose(valT),Y_train_proc)
                valQ = np.transpose(np.dot(term_1,term_2))
                self.val_arrQlist.append(valQ)

                # Compute the scores for the left out object
                projT = np.dot(X_test_proc, valP)
                dims = np.shape(projT)[1]

                # Construct validated predicted X first for one component,
                # then two, three, etc
                for ind in range(0, dims):

                    # part_projT = projT[:,0:ind+1].reshape(1,-1)
                    part_projT = projT[:,0:ind+1]
                    part_valP = valP[:,0:ind+1]
                    valPredX_proc = np.dot(part_projT, np.transpose(part_valP))

                    part_valQ = valQ[:,0:ind+1]
                    valPredY_proc = np.dot(part_projT, np.transpose(part_valQ))

                    # Depending on preprocessing re-process in same manner
                    # in order to get values that compare to original values.
                    if self.Xstand:
                        valPredX = (valPredX_proc * X_train_std) + X_train_mean
                    else:
                        valPredX = valPredX_proc + X_train_mean

                    self.valXpredDict[ind+1][test_index,] = valPredX

                    if self.Ystand:
                        valPredY = (valPredY_proc * Y_train_std) + Y_train_mean
                    else:
                        valPredY = valPredY_proc + Y_train_mean

                    self.valYpredDict[ind+1][test_index,] = valPredY



            # Put all predicitons into an array that corresponds to the
            # original variable
            self.valXpredList = []
            valPreds = self.valXpredDict.values()
            for preds in valPreds:
                pc_arr = np.vstack(preds)
                self.valXpredList.append(pc_arr)


            # Put all predicitons into an array that corresponds to the
            # original variable
            self.valYpredList = []
            valPreds = self.valYpredDict.values()
            for preds in valPreds:
                pc_arr = np.vstack(preds)
                self.valYpredList.append(pc_arr)


            # ==============================================================================
            # From here VALIDATED explained variance is computed
            # ==============================================================================

            # ========== Computations for X ==========
            # -----------------------------------------------------------------
            # Compute PRESSCV (PRediction Error Sum of Squares) for cross
            # validation
            self.valXpredList = self.valXpredDict.values()

            # Collect all PRESS in a dictionary. Keys represent number of
            # component.
            self.PRESSCVdict_indVar_X = {}

            # First compute PRESSCV for zero components
            self.PRESSCV_0_indVar_X = np.sum(np.square(self.arrX_input-self.X_train_means_list), axis=0)
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
            # MSEP for each variable. Also collect PRESS, MSECV, RMSECV in
            # their respective dictionaries for each variable. Keys represent
            # now variables and NOT components as above with
            # self.PRESSdict_indVar
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



            # ========== Computations for Y ==========
            # -----------------------------------------------------------------
            # Compute PRESSCV (PRediction Error Sum of Squares) for cross
            # validation
            self.valYpredList = self.valYpredDict.values()

            # Collect all PRESS in a dictionary. Keys represent number of
            # component.
            self.PRESSdict_indVar = {}

            # First compute PRESSCV for zero components
            self.PRESSCV_0_indVar = np.sum(np.square(self.arrY_input-self.Y_train_means_list), axis=0)
            self.PRESSdict_indVar[0] = self.PRESSCV_0_indVar

            # Compute PRESSCV for each Yhat for 1, 2, 3, etc number of components
            # and compute explained variance
            for ind, Yhat in enumerate(self.valYpredList):
                diffY = self.arrY_input - Yhat
                PRESSCV_indVar = np.sum(np.square(diffY), axis=0)
                self.PRESSdict_indVar[ind+1] = PRESSCV_indVar

            # Now store all PRESSCV values into an array. Then compute MSECV and
            # RMSECV.
            self.PRESSCVarr_indVar = np.array(list(self.PRESSdict_indVar.values()))
            self.MSECVarr_indVar = self.PRESSCVarr_indVar / np.shape(self.arrY_input)[0]
            self.RMSECVarr_indVar = np.sqrt(self.MSECVarr_indVar)
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Compute explained variance for each variable in Y using the
            # MSECV for each variable. Also collect PRESS, MSECV, RMSECV in
            # their respective dictionaries for each variable.  Keys represent
            # now variables and NOT components as above with
            # self.PRESSdict_indVar
            self.cumValExplVarYarr_indVar = np.zeros(np.shape(self.MSECVarr_indVar))
            MSECV_0_indVar = self.MSECVarr_indVar[0,:]

            for ind, MSECV_indVar in enumerate(self.MSECVarr_indVar):
                explVar = (MSECV_0_indVar - MSECV_indVar) / MSECV_0_indVar * 100
                self.cumValExplVarYarr_indVar[ind] = explVar

            self.PRESSCV_indVar = {}
            self.MSECV_indVar = {}
            self.RMSECV_indVar = {}
            self.cumValExplVarY_indVar = {}

            for ind in range(np.shape(self.PRESSCVarr_indVar)[1]):
                self.PRESSCV_indVar[ind] = self.PRESSCVarr_indVar[:,ind]
                self.MSECV_indVar[ind] = self.MSECVarr_indVar[:,ind]
                self.RMSECV_indVar[ind] = self.RMSECVarr_indVar[:,ind]
                self.cumValExplVarY_indVar[ind] = self.cumValExplVarYarr_indVar[:,ind]
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Collect total PRESSCV across all variables in a dictionary.
            self.PRESSCV_total_dict = {}
            self.PRESSCV_total_list = np.sum(self.PRESSCVarr_indVar, axis=1)

            for ind, PRESSCV in enumerate(self.PRESSCV_total_list):
                self.PRESSCV_total_dict[ind] = PRESSCV
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Collect total MSECV across all variables in a dictionary. Also,
            # compute total validated explained variance in Y.
            self.MSECV_total_dict = {}
            self.MSECV_total_list = np.sum(self.MSECVarr_indVar, axis=1) / np.shape(self.arrY_input)[1]
            MSECV_0 = self.MSECV_total_list[0]

            # Compute total validated explained variance in Y
            self.YcumValExplVarList = []
            if not self.Ystand:
                for ind, MSECV in enumerate(self.MSECV_total_list):
                    perc = (MSECV_0 - MSECV) / MSECV_0 * 100
                    self.MSECV_total_dict[ind] = MSECV
                    self.YcumValExplVarList.append(perc)
            else:
                self.YcumValExplVarArr = np.average(self.cumValExplVarYarr_indVar, axis=1)
                self.YcumValExplVarList = list(self.YcumValExplVarArr)

            # Construct list with total validated explained variance in Y in
            # each component
            self.YvalExplVarList = []
            for ind, item in enumerate(self.YcumValExplVarList):
                if ind == len(self.YcumValExplVarList)-1:
                    break
                explVarComp = self.YcumValExplVarList[ind+1] - self.YcumValExplVarList[ind]
                self.YvalExplVarList.append(explVarComp)
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Compute total RMSECV and store values in a dictionary and list.
            self.RMSECV_total_dict = {}
            self.RMSECV_total_list = np.sqrt(self.MSECV_total_list)

            for ind, RMSECV in enumerate(self.RMSECV_total_list):
                self.RMSECV_total_dict[ind] = RMSECV
            # -----------------------------------------------------------------




    def modelSettings(self):
        """
        Returns a dictionary holding the settings under which NIPALS PCR was
        run.
        """
        # Collect settings under which PCA was run.
        self.settings = {}
        self.settings['numComp'] = self.numPC
        self.settings['Xstand'] = self.Xstand
        self.settings['arrX'] = self.arrX_input
        self.settings['analysed arrX'] = self.arrX
        self.settings['arrY'] = self.arrY_input
        self.settings['analysed arrY'] = self.arrX

        return self.settings


    def X_means(self):
        """
        Returns array holding column means of array X.
        """
        return self.Xmeans.reshape(1,-1)


    def X_scores(self):
        """
        Returns array holding scores of array X. First column holds scores
        for component 1, second column holds scores for component 2, etc.
        """
        return self.arrT


    def X_loadings(self):
        """
        Returns array holding loadings of array X. Rows represent variables
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
        arr_corrLoadings = np.zeros((np.shape(self.arrT)[1], np.shape(self.arrP)[0]), float)

        # Compute correlation loadings:
        # For each component in score matrix
        for PC in range(np.shape(self.arrT)[1]):
            PCscores = self.arrT[:, PC]

            # For each variable/attribute in original matrix (not meancentered)
            for var in range(np.shape(self.arrX)[1]):
                origVar = self.arrX[:, var]
                corrs = np.corrcoef(PCscores, origVar)
                arr_corrLoadings[PC, var] = corrs[0,1]

        self.arr_corrLoadings = np.transpose(arr_corrLoadings)

        return self.arr_corrLoadings


    def X_residuals(self):
        """
        Returns a dictionary holding the residual arrays for array X after
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
        Returns a list holding the cumulative calibrated explained variance
        for array X after each component.
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
        through calibration after each component. First row holds RMSEE
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
        for array X after each component. First number represents zero
        components, second number represents component 1, etc.
        """
        return self.XcumValExplVarList


    def X_predVal(self):
        """
        Returns dictionary holding arrays of predicted Xhat after each
        component from validation. Dictionary key represents order of
        component.
        """
        return self.valXpredDict


    def X_PRESSCV_indVar(self):
        """
        Returns array holding PRESSCV for each individual variable in X
        acquired through cross validation after each computed component. First
        row is PRESSCV for zero components, second row for component 1, third
        row for component 2, etc.
        """
        return self.PRESSCVarr_indVar_X


    def X_PRESSCV(self):
        """
        Returns an array holding PRESSCV across all variables in X acquired
        through cross validation after each computed component. First row is
        PRESSCV for zero components, second row for component 1, third row for
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


    def Y_means(self):
        """
        Returns array holding means of columns in array Y.
        """
        return self.Ymeans.reshape(1,-1)


    def Y_loadings(self):
        """
        Returns an array holding loadings C of array Y. Rows represent
        variables and columns represent components. First column for
        component 1, second columns for component 2, etc.
        """
        return self.arrQ


    def Y_corrLoadings(self):
        """
        Returns array holding correlation loadings of array X. First column
        holds correlation loadings for component 1, second column holds
        correlation loadings for component 2, etc.
        """

        # Creates empty matrix for correlation loadings
        arr_YcorrLoadings = np.zeros((np.shape(self.arrT)[1], np.shape(self.arrQ)[0]), float)

        # Compute correlation loadings:
        # For each component in score matrix
        for PC in range(np.shape(self.arrT)[1]):
            PCscores = self.arrT[:, PC]

            # For each variable/attribute in original matrix (not meancentered)
            for var in range(np.shape(self.arrY)[1]):
                origVar = self.arrY[:, var]
                corrs = np.corrcoef(PCscores, origVar)
                arr_YcorrLoadings[PC, var] = corrs[0,1]

        self.arr_YcorrLoadings = np.transpose(arr_YcorrLoadings)

        return self.arr_YcorrLoadings


    def Y_residuals(self):
        """
        Returns a dictionary holding residuals F of array Y after each
        component. Dictionary key represents order of component.
        """
        # Create empty dictionary that will hold residuals
        Y_residualsDict = {}

        # Fill dictionary with residuals arrays from residuals list
        for ind, item in enumerate(self.Y_residualsList):
            Y_residualsDict[ind] = item

        return Y_residualsDict


    def Y_calExplVar(self):
        """
        Returns a list holding the calibrated explained variance for each
        component. First number in list is for component 1, second number for
        component 2, etc.
        """
        return self.YcalExplVarList


    def Y_cumCalExplVar_indVar(self):
        """
        Returns an array holding the cumulative calibrated explained variance
        for each variable in Y after each component. First row represents zero
        components, second row represents one component, third row represents
        two components, etc. Columns represent variables.
        """
        return self.cumCalExplVarYarr_indVar


    def Y_cumCalExplVar(self):
        """
        Returns a list holding the cumulative calibrated explained variance
        for array X after each component. First number represents zero
        components, second number represents component 1, etc.
        """
        return self.YcumCalExplVarList


    def Y_predCal(self):
        """
        Returns dictionary holding arrays of predicted Yhat after each
        component from calibration. Dictionary key represents order of
        components.
        """
        return self.calYpredDict

    def Y_PRESSE_indVar(self):
        """
        Returns array holding PRESSE for each individual variable in Y
        acquired through calibration after each component. First row is
        PRESSE for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.PRESSEarr_indVar


    def Y_PRESSE(self):
        """
        Returns array holding PRESSE across all variables in Y acquired
        through calibration after each computed component. First row is PRESSE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.PRESSE_total_list


    def Y_MSEE_indVar(self):
        """
        Returns an array holding MSEE for each variable in array Y acquired
        through calibration after each computed component. First row holds MSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.MSEEarr_indVar


    def Y_MSEE(self):
        """
        Returns an array holding MSEE across all variables in Y acquired
        through calibration after each computed component. First row is MSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.MSEE_total_list


    def Y_RMSEE_indVar(self):
        """
        Returns an array holding RMSEE for each variable in array Y acquired
        through calibration after each component. First row holds RMSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.RMSEEarr_indVar


    def Y_RMSEE(self):
        """
        Returns an array holding RMSEE across all variables in Y acquired
        through calibration after each computed component. First row is RMSEE
        for zero components, second row for component 1, third row for
        component 2, etc.
        """
        return self.RMSEE_total_list


    def Y_valExplVar(self):
        """
        Returns a list holding the validated explained variance for Y after
        each component. First number in list is for component 1, second number
        for component 2, third number for component 3, etc.
        """
        return self.YvalExplVarList


    def Y_cumValExplVar_indVar(self):
        """
        Returns an array holding the cumulative validated explained variance
        for each variable in Y after each component. First row represents
        zero components, second row represents component 1, third row for
        compnent 2, etc. Columns represent variables.
        """
        return self.cumValExplVarYarr_indVar


    def Y_cumValExplVar(self):
        """
        Returns a list holding the cumulative validated explained variance
        for array X after each component. First number represents zero
        components, second number represents component 1, etc.
        """
        return self.YcumValExplVarList


    def Y_predVal(self):
        """
        Returns dictionary holding arrays of predicted Yhat after each
        component from validation. Dictionary key represents order of
        component.
        """
        return self.valYpredDict


    def Y_PRESSCV_indVar(self):
        """
        Returns an array holding PRESSCV of each variable in array Y acquired
        through cross validation after each computed component. First row is
        PRESSCV for zero components, second row component 1, third row for
        component 2, etc.
        """
        return self.PRESSCVarr_indVar


    def Y_PRESSCV(self):
        """
        Returns an array holding PRESSCV across all variables in Y acquired
        through cross validation after each computed component. First row is
        PRESSCV for zero components, second row component 1, third row for
        component 2, etc.
        """
        return self.PRESSCV_total_list


    def Y_MSECV_indVar(self):
        """
        Returns an array holding MSECV of each variable in array Y acquired
        through cross validation after each computed component. First row is
        MSECV for zero components, second row component 1, third row for
        component 2, etc.
        """
        return self.MSECVarr_indVar


    def Y_MSECV(self):
        """
        Returns an array holding MSECV across all variables in Y acquired
        through cross validation after each computed component. First row is
        MSECV for zero components, second row component 1, third row for
        component 2, etc.
        """
        return self.MSECV_total_list


    def Y_RMSECV_indVar(self):
        """
        Returns an array holding RMSECV for each variable in array Y acquired
        through cross validation after each computed component. First row is
        RMSECV for zero components, second row component 1, third row for
        component 2, etc.
        """
        return self.RMSECVarr_indVar


    def Y_RMSECV(self):
        """
        Returns an array holding RMSECV across all variables in Y acquired
        through cross validation after each computed component. First row is
        RMSECV for zero components, second row component 1, third row for
        component 2, etc.
        """
        return self.RMSECV_total_list


    def regressionCoefficients(self, numComp=1):
        """
        Returns regression coefficients from the fitted model using all
        available samples and a chosen number of components.
        """
        assert numComp <= self.numPC, ValueError('Maximum numComp = ' + str(self.numPC))
        assert numComp > -1, ValueError('numComp must be >= 0')

        # B = P*Q'
        if self.Ystand:
            return np.dot(self.arrP[:,0:numComp], np.transpose(self.arrQ[:,0:numComp])) \
                * np.std(self.arrY_input, ddof=1, axis=0).reshape(1,-1)
        else:
            return np.dot(self.arrP[:,0:numComp], np.transpose(self.arrQ[:,0:numComp]))


    def Y_predict(self, Xnew, numComp=1):
        """
        Return predicted Yhat from new measurements X.
        """

        assert numComp <= self.numPC, ValueError('Maximum numComp = ' + str(self.numPC))
        assert numComp > -1, ValueError('numComp must be >= 0')

        # Return average if numComp == 0
        if numComp == 0:
            Yhat = np.zeros(np.shape(self.arrY_input)) + np.average(self.arrY_input, axis=0)

        else:
            # First pre-process new X data accordingly
            if self.Xstand:
                x_new = (Xnew - np.average(self.arrX_input, axis=0)) / np.std(self.arrX_input, ddof=1, axis=0)
            else:
                x_new = (Xnew - np.average(self.arrX_input, axis=0))

            # Compute the scores for new object
            projT = np.dot(x_new, self.arrP[:, 0:numComp])

            # Compute processed responses
            y_pred_proc = np.dot(projT, np.transpose(self.arrQ[:, 0:numComp]))

            # Compute predicted values back to original scale
            if self.Ystand:
                Yhat = (y_pred_proc * np.std(self.arrY_input, ddof=1, axis=0).reshape(1,-1)) + np.average(self.arrY_input, axis=0)
            else:
                Yhat = y_pred_proc + np.average(self.arrY_input, axis=0)

        return Yhat


    def cvTrainAndTestData(self):
        """
        Returns a list consisting of dictionaries holding training and test
        sets.
        """
        return self.cvTrainAndTestDataList


    def corrLoadingsEllipses(self):
        """
        Returns coordinates for the ellipses that represent 50% and 100% expl.
        variance in correlation loadings plot.
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
