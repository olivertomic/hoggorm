# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:10:44 2011

@author: Oliver Tomic (OTO), <oliver.tomic@nofima.no>

"""

# Import necessary modules
import numpy as np
import numpy.linalg as npla
import statTools as st
import cross_val as cv
import matplotlib.pyplot as plt



class nipalsPCA:
    """
    GENERAL INFO:
    -------------
    This class carries out Principal Component Analysis on arrays using
    NIPALS algorithm.
    
    
    EXAMPLE USE:
    ----
    import pca    
    
    model = pca.nipalsPCA(array, numPC=5, Xstand=False)
    model = pca.nipalsPCA(array)
    model = pca.nipalsPCA(array, numPC=3)
    model = pca.nipalsPCA(array, Xstand=True)
    model = pca.nipalsPCA(array, cvType=["loo"])
    model = pca.nipalsPCA(array, cvType=["lpo", 4])
    model = pca.nipalsPCA(array, cvType=["lolo", [1,2,3,2,3,1]])
        
    
    TYPES:
    ------
    array: <array>
    numPC: <integer>
    mode: <boolean>  
                False: first column centre input data then run PCA 
                True: first scale columns of input data to equal variance
                         then run PCA
    """
    
    def __init__(self, arrX, **kargs):
        """
        On initialisation check how arrX and arrY are to be pre-processed 
        (Xstand and Ystand are either True or False). Then check whether 
        number of PC's chosen by user is OK.
        """
        
#===============================================================================
#         Check what is provided by user for PCA
#===============================================================================
        
        # Check whether number of PC's that are to be computed is provided.
        # If NOT, then number of PC's is set to either number of objects or
        # variables of X whichever is smallest (numPC). If number of  
        # PC's IS provided, then number is checked against maxPC and set to
        # numPC if provided number is larger.
        if 'numPC' not in kargs.keys(): 
            self.numPC = min(np.shape(arrX))
        else:
            maxNumPC = min(np.shape(arrX))           
            if kargs['numPC'] > maxNumPC:
                self.numPC = maxNumPC
            else:
                self.numPC = kargs['numPC']
        
        
        # Define X and Y within class such that the data can be accessed from
        # all attributes in class.
        self.arrX_input = arrX
        
                
        # Pre-process data according to user request.
        # -------------------------------------------
        # Check whether standardisation of X and Y are requested by user. If 
        # NOT, then X and y are centred by default. 
        if 'Xstand' not in kargs.keys():
            self.Xstand = False
        else:
            self.Xstand = kargs['Xstand']
        
                
        # Standardise X if requested by user, otherwise center X.
        if self.Xstand == True:
            self.Xmeans = np.average(self.arrX_input, axis=0)            
            self.Xstd = np.std(self.arrX_input, axis=0, ddof=1)
            self.arrX = (self.arrX_input - self.Xmeans) / self.Xstd
        else:
            self.Xmeans = np.average(self.arrX_input, axis=0)            
            self.arrX = self.arrX_input - self.Xmeans
            
                
        # Check whether cvType is provided. If NOT, then no cross validation
        # is carried out.
        if 'cvType' not in kargs.keys():
            self.cvType = None
        else:
            self.cvType = kargs['cvType']
        
        
        # Before PLS2 NIPALS algorithm starts initiate dictionaries and lists
        # in which results are stored.
        self.X_scoresList = []
        self.X_loadingsList = []
        self.X_loadingsWeightsList = []
        self.coeffList = []
        self.X_residualsList = [self.arrX]
        
        
        # Collect residual matrices/arrays after each computed PC
        self.resids = {}
        self.X_residualsDict = {}
        
        # Collect predicted matrices/array Xhat after each computed PC
        self.calXhatDict_singPC = {}
        
        # Collect explained variance in each PC
        self.calExplainedVariancesDict = {}
        self.X_calExplainedVariancesList = []

        
#===============================================================================
#        Here the NIPALS PCA algorithm on X starts 
#===============================================================================
        threshold = 1.0e-8
        #X_new = self.data.copy()
        X_new = self.arrX.copy()
        
        # Compute number of principal components as specified by user 
        for j in range(self.numPC): 
            
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
            
            # Peel off information explained by actual PC and continue with
            # decomposition on the residuals (X_new = E).
            X_old = X_new.copy()
            Xhat_j = np.dot(t, np.transpose(p))
            X_new = X_old - Xhat_j
            
            # Store residuals E and Xhat in their dictionaries
            self.X_residualsDict[j+1] = X_new
            self.calXhatDict_singPC[j+1] = Xhat_j
            
            if self.Xstand == True:
                self.calXhatDict_singPC[j+1] = (Xhat_j * self.Xstd) + \
                        self.Xmeans            
            else:
                self.calXhatDict_singPC[j+1] = Xhat_j + self.Xmeans
            
            
        # Collect scores and loadings for the actual PC.
        self.arrT = np.hstack(self.X_scoresList)
        self.arrP = np.hstack(self.X_loadingsList)
        
        
#==============================================================================
#         From here computation of CALIBRATED explained variance starts
#==============================================================================
        
                
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
            
            if self.Xstand == True:
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
        PRESSE_0_indVar_X = np.sum(np.square(st.centre(self.arrX_input)), axis=0)
        self.PRESSEdict_indVar_X[0] = PRESSE_0_indVar_X
        
        # Compute PRESS for each Xhat for 1, 2, 3, etc number of components
        # and compute explained variance
        for ind, Xhat in enumerate(self.calXpredList):
            diffX = st.centre(self.arrX_input) - st.centre(Xhat)
            PRESSE_indVar_X = np.sum(np.square(diffX), axis=0)
            self.PRESSEdict_indVar_X[ind+1] = PRESSE_indVar_X
                    
        # Now store all PRESSE values into an array. Then compute MSEE and
        # RMSEE.
        self.PRESSEarr_indVar_X = np.array(list(self.PRESSEdict_indVar_X.values()))
        self.MSEEarr_indVar_X = self.PRESSEarr_indVar_X / \
                np.shape(self.arrX_input)[0]
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
        self.MSEE_total_list_X = np.sum(self.MSEEarr_indVar_X, axis=1) / \
                np.shape(self.arrX_input)[1]
        MSEE_0_X = self.MSEE_total_list_X[0]

        # Compute total cumulated calibrated explained variance in X
        self.XcumCalExplVarList = []
        if self.Xstand == False:
            for ind, MSEE_X in enumerate(self.MSEE_total_list_X):
                perc = (MSEE_0_X - MSEE_X) / MSEE_0_X * 100
                self.MSEE_total_dict_X[ind] = MSEE_X
                self.XcumCalExplVarList.append(perc)
        else:
            self.XcumCalExplVarArr = np.average(self.cumCalExplVarXarr_indVar, axis=1)
            self.XcumCalExplVarList = list(self.XcumCalExplVarArr)
        
        # Construct list with total explained variance in X for each PC
        self.XcalExplVarList = []
        for ind, item in enumerate(self.XcumCalExplVarList):
            if ind == len(self.XcumCalExplVarList)-1: break
            explVarComp = self.XcumCalExplVarList[ind+1] - \
                    self.XcumCalExplVarList[ind]
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
                
        
#==============================================================================
#         From here cross validation procedure starts
#==============================================================================
        if self.cvType == None:
            pass
        else:
            numObj = np.shape(self.arrX)[0]
            
            if self.cvType[0] == "loo":
                print("loo")
                cvComb = cv.LeaveOneOut(numObj)
            elif self.cvType[0] == "lpo":
                print("lpo")
                cvComb = cv.LeavePOut(numObj, self.cvType[1])
            elif self.cvType[0] == "lolo":
                print("lolo")
                cvComb = cv.LeaveOneLabelOut(self.cvType[1])
            else:
                print('Requested form of cross validation is not available')
            
            
            # Collect predicted x (i.e. xhat) for each CV segment in a
            # dictionary according to number of PC
            self.valXpredDict = {}
            for ind in range(1, self.numPC+1):
                self.valXpredDict[ind] = []
            
            # Collect train and test set in dictionaries for each PC and put
            # them in this list.            
            self.cvTrainAndTestDataList = []            
            

            # Collect: validation X scores T, validation X loadings P,
            # validation Y scores U, validation Y loadings Q,
            # validation X loading weights W and scores regression coefficients C
            # in lists for each PC
            self.val_arrTlist = []
            self.val_arrPlist = []
            self.val_arrQlist = [] 

            # Collect train and test set in a dictionary for each PC            
            self.cvTrainAndTestDataList = []
            self.X_train_means_list = []          
            
            # First devide into combinations of training and test sets
            for train_index, test_index in cvComb:
                X_train, X_test = cv.split(train_index, test_index, self.arrX_input)
                
                subDict = {}
                subDict['x train'] = X_train
                subDict['x test'] = X_test  
                self.cvTrainAndTestDataList.append(subDict)
                
                # -------------------------------------------------------------                    
                # Center or standardise X according to users choice 
                if self.Xstand == True:
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
                self.X_train_means_list.append(X_train_mean)
                
        
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
                    
                    # Peel off information explained by actual PC and continue with
                    # decomposition on the residuals (X_new = E).
                    X_old = X_new.copy()
                    Xhat_j = np.dot(t, np.transpose(p))
                    X_new = X_old - Xhat_j
                
                # Collect X scores and X loadings for the actual PC.
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
                    
                    part_projT = projT[:,0:ind+1].reshape(1,-1)
                    part_valP = valP[:,0:ind+1]
                    valPredX_proc = np.dot(part_projT, np.transpose(part_valP))
                    
                    
                    # Depending on preprocessing re-process in same manner
                    # in order to get values that compare to original values.
                    if self.Xstand == True:
                        valPredX = (valPredX_proc * X_train_std) + \
                                X_train_mean
                    else:
                        valPredX = valPredX_proc + X_train_mean
                    
                    self.valXpredDict[ind+1].append(valPredX)
                
            
            # Convert list of one-row arrays into one array such that it 
            # corresponds to the orignial variable
            for ind in range(1, dims+1):
                self.valXpredDict[ind] = np.vstack(self.valXpredDict[ind])
                

            # Put all predicitons into an array that corresponds to the
            # original variable
            #self.valPredXarrList = []
            self.valXpredList = []
            valPreds = self.valXpredDict.values()
            for preds in valPreds:
                pc_arr = np.vstack(preds)
                self.valXpredList.append(pc_arr)
                        

#==============================================================================
# From here VALIDATED explained variance is computed
#==============================================================================
            
            # ========== Computations for X ==========
            # -----------------------------------------------------------------
            # Compute PRESSCV (PRediction Error Sum of Squares) for cross 
            # validation 
            self.valXpredList = self.valXpredDict.values()
            
            # Collect all PRESS in a dictionary. Keys represent number of 
            # component.            
            self.PRESSdict_indVar_X = {}
            
            # First compute PRESSCV for zero components            
            varX = np.var(self.arrX_input, axis=0, ddof=1)
            self.PRESSCV_0_indVar_X = (varX * np.square(np.shape(self.arrX_input)[0])) \
                    / (np.shape(X_train)[0])
            self.PRESSdict_indVar_X[0] = self.PRESSCV_0_indVar_X
            
            # Compute PRESSCV for each Yhat for 1, 2, 3, etc number of 
            # components and compute explained variance
            for ind, Xhat in enumerate(self.valXpredList):
                #diffX = self.arrX_input - Xhat
                diffX = st.centre(self.arrX_input) - st.centre(Xhat)
                PRESSCV_indVar_X = np.sum(np.square(diffX), axis=0)
                self.PRESSdict_indVar_X[ind+1] = PRESSCV_indVar_X
                        
            # Now store all PRESSCV values into an array. Then compute MSECV 
            # and RMSECV.
            self.PRESSCVarr_indVar_X = np.array(list(self.PRESSdict_indVar_X.values()))
            self.MSECVarr_indVar_X = self.PRESSCVarr_indVar_X / \
                    np.shape(self.arrX_input)[0]
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
            self.MSECV_total_list_X = np.sum(self.MSECVarr_indVar_X, axis=1) / \
                    np.shape(self.arrX_input)[1]
            MSECV_0_X = self.MSECV_total_list_X[0]

            # Compute total validated explained variance in X
            self.XcumValExplVarList = []
            if self.Xstand == False:
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
                if ind == len(self.XcumValExplVarList)-1: break
                explVarComp = self.XcumValExplVarList[ind+1] - \
                        self.XcumValExplVarList[ind]
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
        run. Dictionary key represents order of PC.
        """
        # Collect settings under which PCA was run.
        self.settings = {}
        self.settings['numPC'] = self.numPC
        self.settings['Xstand'] = self.Xstand
        self.settings['arrX'] = self.arrX_input
        self.settings['analysed arrX'] = self.arrX
        
        return self.settings    

    
    def X_means(self):
        """
        Returns the score matrix T. First column holds scores for PC1, 
        second column holds scores for PC2, etc.
        """
        return self.Xmeans.reshape(1,-1)    
        
    
    def X_scores(self):
        """
        Returns the score matrix T. First column holds scores for PC1, 
        second column holds scores for PC2, etc.
        """
        return self.arrT
        
    
    def X_loadings(self):
        """
        Returns the loading matrix P. First column holds loadings for PC1, 
        second column holds scores for PC2, etc.
        """
        return self.arrP
    
    
    def X_corrLoadings(self):
        """
        Returns correlation loadings. First column holds correlation loadings
        for PC1, second column holds scores for PC2, etc.
        """

        # Creates empty matrix for correlation loadings
        arr_corrLoadings = np.zeros((np.shape(self.arrT)[1], \
            np.shape(self.arrP)[0]), float)
        
        # Compute correlation loadings:
        # For each PC in score matrix
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
        Returns a dictionary holding the residual matrices E after each 
        computed PC. Dictionary key represents order of PC.
        """
        return self.X_residualsDict
    
    
    def X_calExplVar(self):
        """
        Returns a list holding the calibrated explained variance for 
        each PC. 
        """
        return self.XcalExplVarList
    
    
    def X_cumCalExplVar_indVar(self):
        """
        Returns an array holding the cumulative calibrated explained variance
        for each variable in X after each PC.
        """
        return self.cumCalExplVarXarr_indVar
    
    
    def X_cumCalExplVar(self):
        """
        Returns a list holding the cumulative calibrated explained variance for 
        each PC. Dictionary key represents order of PC. 
        """
        return self.XcumCalExplVarList
    
    
    def X_predCal(self):
        """
        Returns a dictionary holding the predicted matrices Xhat from 
        calibration after each computed PC. Dictionary key represents order 
        of PC.
        """
        return self.calXpredDict
    
    
    def X_PRESSE_indVar(self):
        """
        Returns array holding PRESSE for each individual variable acquired
        through calibration after each computed PC. First row is PRESS for zero
        components, second row component 1, third row for component 2, etc.
        """
        return self.PRESSEarr_indVar_X
    
    
    def X_PRESSE(self):
        """
        Returns an array holding PRESS across all variables in X acquired  
        through calibration after each computed PC. First row is PRESS for zero
        components, second row component 1, third row for component 2, etc.
        """   
        return self.PRESSE_total_list_X
    
    
    def X_MSEE_indVar(self):
        """
        Returns an arrary holding MSE from calibration for each variable in X. 
        First row is MSE for zero components, second row for component 1, etc.
        """
        return self.MSEEarr_indVar_X
    
    
    def X_MSEE(self):
        """
        Returns an array holding MSE across all variables in X acquired through 
        calibration after each computed PC. First row is PRESS for zero
        components, second row component 1, third row for component 2, etc.
        """   
        return self.MSEE_total_list_X
    
    
    def X_RMSEE_indVar(self):
        """
        Returns an arrary holding RMSE from calibration for each variable in X. 
        First row is MSE for zero components, second row for component 1, etc.
        """
        return self.RMSEEarr_indVar_X
    
    
    def X_RMSEE(self):
        """
        Returns an array holding RMSE across all variables in X acquired through 
        calibration after each computed PC. First row is PRESS for zero
        components, second row component 1, third row for component 2, etc.
        """   
        return self.RMSEE_total_list_X
    
    
    def X_valExplVar(self):
        """
        Returns list holding calibrated explained variance for each PC in X.
        """
        return  self.XvalExplVarList
    
    
    def X_cumValExplVar_indVar(self):
        """
        Returns array holding cumulative validated explained variance in X for
        each variable. Rows represent variables in X. Rows represent number of
        components.
        """
        return self.cumValExplVarXarr_indVar
    
    
    def X_cumValExplVar(self):
        """
        Returns list holding cumulative calibrated explained variance in X.
        """
        return self.XcumValExplVarList
    
    
    def X_predVal(self):
        """
        Returns dictionary holding arrays of predicted Xhat after each component 
        from validation. Dictionary key represents order of PC.
        """
        return self.valXpredDict
    
    
    def X_PRESSCV_indVar(self):
        """
        Returns array holding PRESS for each individual variable in X acquired
        through cross validation after each computed PC. First row is PRESS for
        zero components, second row component 1, third row for component 2, etc.
        """
        return self.PRESSCVarr_indVar_X
    
    
    def X_PRESSCV(self):
        """
        Returns an array holding PRESS across all variables in X acquired  
        through cross validation after each computed PC. First row is PRESS for 
        zero components, second row component 1, third row for component 2, etc.
        """   
        return self.PRESSCV_total_list_X
    
    
    def X_MSECV_indVar(self):
        """
        Returns an arrary holding MSE from cross validation for each variable  
        in X. First row is MSE for zero components, second row for component 1, 
        etc.
        """
        return self.MSECVarr_indVar_X
    
    
    def X_MSECV(self):
        """
        Returns an array holding MSE across all variables in X acquired through 
        cross validation after each computed PC. First row is PRESS for zero
        components, second row component 1, third row for component 2, etc.
        """   
        return self.MSECV_total_list_X
    
    
    def X_RMSECV_indVar(self):
        """
        Returns an arrary holding RMSE from cross validation for each variable
        in X. First row is MSE for zero components, second row for component 1, 
        etc.
        """
        return self.RMSECVarr_indVar_X
    
    
    def X_RMSECV(self):
        """
        Returns an array holding RMSE across all variables in X acquired through 
        cross validation after each computed PC. First row is PRESS for zero
        components, second row component 1, third row for component 2, etc.
        """   
        return self.RMSECV_total_list_X
    
    
    def cvTrainAndTestData(self):
        """
        Returns a list consisting of dictionaries holding training and test sets.
        """
        return self.cvTrainAndTestDataList
    
    
    def corrLoadingsEllipses(self):
        """
        Returns the ellipses that represent 50% and 100% expl. variance in
        correlation loadings plot.
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




def plots(model, pc=[1,2], plots=[1,2,3,4], objNames=[], varNames=[]):
    """
    This functions generates plots that visualise the most important results
    from PCA
    """
    
    # Generate names/numbers for objects if no objects are given
    if bool(objNames) == False:
        numObj, numVar = np.shape(model.modelSettings()['arrX'])
        
        for num in range(1, numObj+1):
            label = 'Obj {0}'.format(num)
            objNames.append(label)
    
    
    # Generate names/numbers for variables if no objects are given
    if bool(varNames) == False:
        numObj, numVar = np.shape(model.modelSettings()['arrX'])
        
        for num in range(1, numVar+1):
            label = 'Var {0}'.format(num)
            varNames.append(label)
    
    # Generate a list with names of PC's used for PCA
    obj, numPC = np.shape(model.X_scores())
    pcNames = []
    
    for num in range(numPC+1):
        label = 'PC{0}'.format(num)
        pcNames.append(label)
    
    # Generate plot as requested by user
    for item in plots:
        print(item)        
        
        # SCORES PLOT        
        if item == 1:
            
            # Access PCA scores and explained variances from model
            Xscores = model.X_scores()
            XexplVar = model.X_calExplVar()
            
            # Initiate plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
                        
            # Find maximum and minimum scores along along PC's selected
            # by the user
            xMax = max(Xscores[:,pc[0]-1])
            xMin = min(Xscores[:,pc[0]-1])
            
            yMax = max(Xscores[:,pc[1]-1])
            yMin = min(Xscores[:,pc[1]-1])
                        
            # Compute sufficient distance of label text from scatter point
            xSpace = (xMax / 100) * 5
            ySpace = (yMax / 100) * 4
                        
            # Set limits for dashed lines representing the axes.
            # x-axis
            if abs(xMax) >= abs(xMin):
                extraX = xMax * .4
                limX = xMax * .3
            
            elif abs(xMax) < abs(xMin):
                extraX = abs(xMin) * .4
                limX = abs(xMin) * .3
            
            # y-axis
            if abs(yMax) >= abs(yMin):
                extraY = yMax * .4
                limY = yMax * .3
            
            elif abs(yMax) < abs(yMin):
                extraY = abs(yMin) * .4
                limY = abs(yMin) * .3
                        
            # Loop through all coordinates (PC1,PC2) and names to plot scores.
            for ind, name in enumerate(objNames):
                
                ax.scatter(Xscores[ind,pc[0]-1], Xscores[ind,pc[1]-1], s=10, \
                        c='w', marker='o', edgecolor='grey')
                ax.text(Xscores[ind,pc[0]-1] + xSpace, \
                        Xscores[ind,pc[1]-1] + ySpace, name, fontsize=12)
            
            # Set limits for dashed lines representing axes
            xMaxLine = xMax + extraX
            xMinLine = xMin - extraX
            
            yMaxLine = yMax + extraY
            yMinLine = yMin - extraY
            
            # Plot dashes axes lines
            ax.plot([0,0], [yMaxLine,yMinLine], color='0.4', \
                    linestyle='dashed', linewidth=1)
            ax.plot([xMinLine,xMaxLine], [0,0], color='0.4', \
                    linestyle='dashed', linewidth=1)
                        
            
            # Set limits for plot regions.
            xMaxLim = xMax + limX
            xMinLim = xMin - limX
            
            yMaxLim = yMax + limY
            yMinLim = yMin - limY
            
            ax.set_xlim(xMinLim,xMaxLim)
            ax.set_ylim(yMinLim,yMaxLim)
            
            
            # Plot title, axis names.             
            ax.set_xlabel('{0} ({1}%)'.format(pcNames[pc[0]], \
                    str(round(XexplVar[pc[0]-1],1))))
            ax.set_ylabel('{0} ({1}%)'.format(pcNames[pc[1]], \
                    str(round(XexplVar[pc[1]-1],1))))
            
            ax.set_title('PCA scores plot')
            
            plt.show()
        
        
        # LOADINGS PLOT
        if item == 2:
            
            # Access PCA scores and explained variances from model
            Xloadings = model.X_loadings()
            XexplVar = model.X_calExplVar()
            
            # Initiate plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
                        
            # Find maximum and minimum scores along along PC's selected
            # by the user
            xMax = max(Xloadings[:,pc[0]-1])
            xMin = min(Xloadings[:,pc[0]-1])
            
            yMax = max(Xloadings[:,pc[1]-1])
            yMin = min(Xloadings[:,pc[1]-1])
                        
            # Compute sufficient distance of label text from scatter point
            xSpace = (xMax / 100) * 5
            ySpace = (yMax / 100) * 4
                        
            # Set limits for dashed lines representing the axes.
            # x-axis
            if abs(xMax) >= abs(xMin):
                extraX = xMax * .4
                limX = xMax * .3
            
            elif abs(xMax) < abs(xMin):
                extraX = abs(xMin) * .4
                limX = abs(xMin) * .3
            
            # y-axis
            if abs(yMax) >= abs(yMin):
                extraY = yMax * .4
                limY = yMax * .3
            
            elif abs(yMax) < abs(yMin):
                extraY = abs(yMin) * .4
                limY = abs(yMin) * .3
                        
            # Loop through all coordinates (PC1,PC2) and names to plot scores.
            for ind, name in enumerate(varNames):
                
                ax.scatter(Xloadings[ind,pc[0]-1], Xloadings[ind,pc[1]-1], \
                        s=10, c='w', marker='o', edgecolor='grey')
                ax.text(Xloadings[ind,pc[0]-1] + xSpace, \
                        Xloadings[ind,pc[1]-1] + ySpace, name, fontsize=12)
            
            # Set limits for dashed lines representing axes
            xMaxLine = xMax + extraX
            xMinLine = xMin - extraX
            
            yMaxLine = yMax + extraY
            yMinLine = yMin - extraY
            
            # Plot dashes axes lines
            ax.plot([0,0], [yMaxLine,yMinLine], color='0.4', \
                    linestyle='dashed', linewidth=1)
            ax.plot([xMinLine,xMaxLine], [0,0], color='0.4', \
                    linestyle='dashed', linewidth=1)
                        
            
            # Set limits for plot regions.
            xMaxLim = xMax + limX
            xMinLim = xMin - limX
            
            yMaxLim = yMax + limY
            yMinLim = yMin - limY
            
            ax.set_xlim(xMinLim,xMaxLim)
            ax.set_ylim(yMinLim,yMaxLim)
            
            
            # Plot title, axis names. 
            ax.set_xlabel('{0} ({1}%)'.format(pcNames[pc[0]], \
                    str(round(XexplVar[pc[0]-1],1))))
            ax.set_ylabel('{0} ({1}%)'.format(pcNames[pc[1]], \
                    str(round(XexplVar[pc[1]-1],1))))
            
            ax.set_title('PCA loadings plot')
            
            plt.show()
        
        
        # CORRELATION LOADINGS PLOT
        if item == 3:
            
            # Access PCA scores and explained variances from model
            XcorrLoadings = model.X_corrLoadings()
            XexplVar = model.X_calExplVar()
            
            # Compute coordinates for  circles in correlation loadings plot
            t = np.arange(0.0, 2*np.pi, 0.01)
            
            # Coordinates for outer circle
            xcords = np.cos(t)
            ycords = np.sin(t)
            
            # Coordinates for inner circle
            xcords50percent = 0.707 * np.cos(t)
            ycords50percent = 0.707 * np.sin(t)
            
            # Initiate plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.plot(xcords, ycords, 'b-')
            ax.plot(xcords50percent, ycords50percent, 'b-')
            
            #ax.scatter(pc1_CL, pc2_CL, s=10, c='r', marker='o', edgecolor='grey')
            # Loop through all coordinates (PC1,PC2) and names to plot scores.
            for ind, name in enumerate(varNames):
                
                ax.scatter(XcorrLoadings[ind,pc[0]-1], \
                        XcorrLoadings[ind,pc[1]-1], \
                        s=10, c='w', marker='o', edgecolor='grey')
                ax.text(XcorrLoadings[ind,pc[0]-1] + xSpace, \
                        XcorrLoadings[ind,pc[1]-1] + ySpace, name, fontsize=12)
            
            # Plot lines through origo.
            left = -1.3; right = 1.3; top = 1.3; bottom = -1.3
            ax.plot([0,0], [top,bottom], color='0.4', linestyle='dashed', \
                    linewidth=1)
            ax.plot([left,right], [0,0], color='0.4', linestyle='dashed', \
                    linewidth=1)
            
            # Plot title, axis names. 
            ax.set_xlabel('{0} ({1}%)'.format(pcNames[pc[0]], \
                    str(round(XexplVar[pc[0]-1],1))))
            ax.set_ylabel('{0} ({1}%)'.format(pcNames[pc[1]], \
                    str(round(XexplVar[pc[1]-1],1))))
            
            ax.set_title('PCA correlation loadings plot')
            
            ax.set_xlim(-1.4,1.4)
            ax.set_ylim(-1.1,1.1)
            
            plt.show()
            
        
        # Explained variances plot        
        if item == 4:
            
            # Access PCA scores and explained variances from model
            cal = model.X_cumCalExplVar()
            val = model.X_cumValExplVar()
            
            # Plot explained variances
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            left = -0.2; right = len(pcNames) - 0.5; top = 105; bottom = -5
            xPos = range(len(pcNames))
            ax.plot(xPos, cal, color='0.4', linestyle='solid', linewidth=1, \
                label='calibrated explained variance')
            ax.plot(xPos, val, color='0.4', linestyle='dashed', linewidth=1, \
                label='validated explained variance')
            
            ax.set_xticks(xPos)
            
            ax.set_xticklabels((pcNames), rotation=0, ha='center')
            ax.set_ylabel('Explained variance')
            
            plt.legend(loc='lower right', shadow=True, labelspacing=.1)
            ltext = plt.gca().get_legend().get_texts()
            plt.setp(ltext[0], fontsize = 10, color = 'k')
            
            ax.set_xlim(left,right)
            ax.set_ylim(bottom,top)
            
            plt.show()
    
