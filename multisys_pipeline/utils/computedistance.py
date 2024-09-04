import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
#from computeCCA import computeCCA, computeCCA_crossconditionvalidation
#from computeCKA import computeCKA, computeCKA_crossconditionvalidation
#from computemetricCKA import computemetricCKA, computemetricCKA_crossconditionvalidation
#from computeWilliams2021 import computeWilliams2021, computeWilliams2021_crossconditionvalidation
from multisys_pipeline.utils.computeCCA import computeCCA, computeCCA_crossconditionvalidation
from multisys_pipeline.utils.computeCKA import computeCKA, computeCKA_crossconditionvalidation
from multisys_pipeline.utils.computemetricCKA import computemetricCKA, computemetricCKA_crossconditionvalidation
from multisys_pipeline.utils.computeWilliams2021 import computeWilliams2021, computeWilliams2021_crossconditionvalidation

def computedistance(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, similarityname='Williams2021', NORMALIZE_BETWEEN0AND1=1):
    ##########################################################################
    # INPUTS
    # Xtrain: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain.
    # Ytrain: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain. 
    # Xtest:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before computing the distance measure, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before computing the distance measure, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that the distance measure does not rely on low variance dimensions.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing the distance
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing the distance
    # If both options are None then don't perform PCA
    
    # OUTPUTS
    # distance_train: a number between 0 and 1(most similar) quantifying the similarity between Xtrain and Ytrain, potentially after projecting Xtrain and Ytrain onto some lower dimensional space with PCA
    # distance_test: a number between 0 and 1(most similar) quantifying the similarity between Xtest and Ytest, potentially after projecting Xtest and Ytest onto some lower dimensional space with PCA
    ##########################################################################
    assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None)) > 0, "Error: One option can be None or both options can be None. If both options are None then PCA is not performed."
    
    #---------------
    if similarityname == 'CCA': 
        CCA_train, CCA_test = computeCCA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, n_CCs_KEEP=None)
        distance_train = 1 - np.mean(CCA_train)# convert all numbers to distances so CCA distance = 1 - CCA similarity, varies between 0(most similar) and 1
        distance_test = 1 - np.mean(CCA_test)# convert all numbers to distances so CCA distance = 1 - CCA similarity, varies between 0(most similar) and 1
    #---------------
    if similarityname == 'CKA': 
        CKA_train, CKA_test = computeCKA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP)
        distance_train = 1 - CKA_train# convert all numbers to distances so CKA distance = 1 - CKA similarity, varies between 0(most similar) and 1
        distance_test = 1 - CKA_test# convert all numbers to distances so CKA distance = 1 - CKA similarity, varies between 0(most similar) and 1
    #---------------
    if similarityname == 'metricCKA': 
        distance_train, distance_test = computemetricCKA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP)
        # convert all numbers to distances, metricCKA distance varies between 0(most similar) and pi/2
        if NORMALIZE_BETWEEN0AND1:
            distance_train = distance_train / (np.pi/2)# scale distance to be between 0(most similar) and 1
            distance_test = distance_test / (np.pi/2)# scale distance to be between 0(most similar) and 1
    #---------------
    if similarityname == 'Williams2021': 
        distance_train, distance_test = computeWilliams2021(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, alpha=1)
        # convert all numbers to distances, Williams2021 distance varies between 0(most similar) and something(can be greater than 1)
        # The output of metric.score from Williams2021 is np.arccos(C) where C is some number that technically is bounded between -1 and 1, so Williams2021 distance is bounded between 0 and pi
        # However, C seems to always fall between 0 and 1 so then Williams2021 distance is bounded between 0 and pi/2
        if NORMALIZE_BETWEEN0AND1:
            distance_train = distance_train / (np.pi/2)# scale distance to be between 0(most similar) and 1
            distance_test = distance_test / (np.pi/2)# scale distance to be between 0(most similar) and 1
    #---------------   
    return distance_train, distance_test
   
    
   
    
    
def computedistance_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, similarityname='Williams2021', NORMALIZE_BETWEEN0AND1=1):
     ##########################################################################
     # INPUTS
     # X: (p, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
     # Y: (q, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
     # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
     # similarityname: string 'CCA' 'CKA' 'metricCKA' 'Williams2021'
     
     # PCA eigenvectors are determined with data from X and Y across all conditions except one. The one condition serves as the test data for cross-validation. This process is repeated for all conditions.
     # For each condition create the following arrays and call the function computeWilliams2021
     # Xtrain = X[:,:,all-conditions-except-condition-icondition]: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain.
     # Ytrain = Y[:,:,all-conditions-except-condition-icondition]: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain.
     # Xtest = X[:,:,icondition]:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
     # Ytest = Y[:,:,icondition]:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
     
     # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before computing the distance measure, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
     # n_PCs_KEEP: Before computing the distance measure, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
     # Xtrain and Ytrain are first reduced using PCA. This ensures that the distance measure does not rely on low variance dimensions.
     # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing the distance
     # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing the distance
     # If both options are None then don't perform PCA
     
     # OUTPUTS
     # distance_train (n_conditions,) array contains a number between 0 and something (can be greater than 1) quantifying the similarity between all data in X and Y that is not from condition icondition
     # distance_test (n_conditions,) array contains a number between 0 and something (can be greater than 1) quantifying the similarity between data from X[:,:,icondition] and Y[:,:,icondition]
     ##########################################################################
     assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None)) > 0, "Error: One option can be None or both options can be None. If both options are None then PCA is not performed."
    
     
     #---------------
     if similarityname == 'CCA': 
         CCA_train, CCA_test = computeCCA_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP)
         distance_train = 1 - CCA_train# convert all numbers to distances so CCA distance = 1 - CCA similarity, varies between 0(most similar) and 1
         distance_test = 1 - CCA_test# convert all numbers to distances so CCA distance = 1 - CCA similarity, varies between 0(most similar) and 1
     #---------------
     if similarityname == 'CKA': 
         CKA_train, CKA_test = computeCKA_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP)
         distance_train = 1 - CKA_train# convert all numbers to distances so CKA distance = 1 - CKA similarity, varies between 0(most similar) and 1
         distance_test = 1 - CKA_test# convert all numbers to distances so CKA distance = 1 - CKA similarity, varies between 0(most similar) and 1
     #---------------
     if similarityname == 'metricCKA': 
         distance_train, distance_test = computemetricCKA_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP)
         # convert all numbers to distances, metricCKA distance varies between 0(most similar) and pi/2
         if NORMALIZE_BETWEEN0AND1:
             distance_train = distance_train / (np.pi/2)# scale distance to be between 0(most similar) and 1
             distance_test = distance_test / (np.pi/2)# scale distance to be between 0(most similar) and 1
     #---------------
     if similarityname == 'Williams2021': 
         distance_train, distance_test = computeWilliams2021_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP)# (n_conditions,)
         # convert all numbers to distances, Williams2021 distance varies between 0(most similar) and something(can be greater than 1)
         # The output of metric.score from Williams2021 is np.arccos(C) where C is some number that technically is bounded between -1 and 1, so Williams2021 distance is bounded between 0 and pi
         # However, C seems to always fall between 0 and 1 so then Williams2021 distance is bounded between 0 and pi/2
         if NORMALIZE_BETWEEN0AND1:
             distance_train = distance_train / (np.pi/2)# scale distance to be between 0(most similar) and 1
             distance_test = distance_test / (np.pi/2)# scale distance to be between 0(most similar) and 1
     #---------------   
     return distance_train, distance_test# (n_conditions,) array
    
    
    
