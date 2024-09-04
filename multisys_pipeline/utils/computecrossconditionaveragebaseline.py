import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions

#import sys
from multisys_pipeline.utils.computedistance import computedistance
from multisys_pipeline.utils.computedistance import computedistance_crossconditionvalidation
#from test_pipeline.utils.computedistance import computedistance


# cross-condition average baseline for model/data similarity is computed using only neural data
# compare original neural dataset to one in which each neuron's firing rate across conditions is replaced by a single time varying firing rate that is obtained after averaging across conditions (each neuron can have a different time varying firing rate in this baseline dataset)
# Train baseline: compute the distance between a concatenation of the original data over all N conditions, and the new average-over-N-conditions dataset concatenated over all N conditions (in the new dataset each condition is the same and is just repeated N times). 
def computecrossconditionaveragebaseline(neuraldata, similarityname, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, NORMALIZE_BETWEEN0AND1=1):# neuraldata is a n_neurons x n_timesteps x n_conditions array
    Y = neuraldata.copy()
    [n_neurons, n_T, n_conditions] = Y.shape
    X = -700*np.ones((n_neurons, n_T, n_conditions))
    Yaverage = np.mean(neuraldata.copy(),2)# n_neurons x n_T array, average across conditions
    for icondition in range(n_conditions):
        X[:,:,icondition] =  Yaverage 
    #---------------
    p, n_datapoints, n_conditions = X.shape
    q, n_datapoints, n_conditions = Y.shape
    if similarityname != 'DSA': 
        X = X.reshape(p,n_datapoints*n_conditions, order='F')# (p, n_datapoints*n_conditions) array
        Y = Y.reshape(q,n_datapoints*n_conditions, order='F')# (q, n_datapoints*n_conditions) array
    else:
        X = np.transpose(X.copy(), axes=[2, 1, 0]) # b x T x N
        Y = np.transpose(Y.copy(), axes=[2, 1, 0])
    #---------------
    distance_train_cross_condition_average_baseline, distance_test_cross_condition_average_baseline = computedistance(X, Y, X, Y, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)# (n_conditions,) array
    #---------------
    return distance_train_cross_condition_average_baseline# a single number


# cross-condition average baseline for model/data similarity is computed using only neural data
# compare original neural dataset to one in which each neuron's firing rate across conditions is replaced by a single time varying firing rate that is obtained after averaging across conditions (each neuron can have a different time varying firing rate in this baseline dataset)
# Return N=n_conditions numbers. For each heldout condition compute the following:
# Train baseline: compute the distance between a concatenation of the original data over N-1 conditions, and the new average-over-(N-1)-conditions dataset concatenated over N-1 conditions (in the new dataset each condition is the same and is just repeated N-1 times). 
# Don't include the test condition in this average. The train distance does not have access to any of the neural data in the test=Nth condition
# Test baseline: apply parameters learned from train baseline to compute distance on the Nth condition of the original dataset and the new average-over-all-N-conditions dataset. Now include the test condition in this average.
  
# Original
# Train baseline: compute the similarity between a concatenation of the original data over N-1 conditions, and the new average-over-all-N-conditions dataset concatenated over N-1 conditions (in the new dataset each condition is the same and is just repeated N-1 times)
# Test baseline: apply parameters learned from train baseline to compute similarity on the Nth condition of the original dataset and the new average-over-all-N-conditions dataset
def computecrossconditionaveragebaseline_crossconditionvalidation(neuraldata, similarityname, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, NORMALIZE_BETWEEN0AND1=1):# neuraldata is a n_neurons x n_timesteps x n_conditions array
    '''
    Y = neuraldata.copy()
    Yaverage = np.mean(neuraldata.copy(),2)# n_neurons x n_T array, average across all conditions
    n_neurons, n_T, n_conditions = Y.shape
    distance_train_cross_condition_average_baseline = -700*np.ones(n_conditions)
    distance_test_cross_condition_average_baseline = -700*np.ones(n_conditions)
    for iconditiontest in range(n_conditions):# test on data from condition icondition
        iconditionstrain = np.arange(0,n_conditions)# indices of all conditions 
        iconditionstrain = np.delete(iconditionstrain, iconditiontest)# indices of all conditions except icondition
        
        Ytrain = Y[:,:,iconditionstrain].copy()# (n_neurons, n_T, n_conditions-1) array
        Ytrain_average = np.mean(Ytrain.copy(),2)# n_neurons x n_T array, average across N-1 training conditions
        Ytrain = Ytrain.reshape(n_neurons,n_T*(n_conditions-1), order='F')# (n_neurons, n_datapoints_train) array
        Xtrain = -700*np.ones((n_neurons, n_T, n_conditions-1))# duplicate Ytrain_average n_conditions-1 times
        for icondition in range(n_conditions-1):
            Xtrain[:,:,icondition] =  Ytrain_average
            #Xtrain[:,:,icondition] =  Yaverage# old 
        Xtrain = Xtrain.reshape(n_neurons,n_T*(n_conditions-1), order='F')# (n_neurons, n_datapoints_train) array    
        
        #Xtrain_check = -700*np.ones((n_neurons,n_T*(n_conditions-1)))# (n_neurons, n_datapoints_train) array
        #Ytrain_check = -700*np.ones((n_neurons,n_T*(n_conditions-1)))# (n_neurons, n_datapoints_train) array
        #ifill = 0
        #for icondition in iconditionstrain:
        #    Xtrain_check[:,ifill*n_T:(ifill+1)*n_T] = Ytrain_average# (n_neurons, n_T) array
        #    Ytrain_check[:,ifill*n_T:(ifill+1)*n_T] = Y[:,:,icondition]# (n_neurons, n_T) array
        #    ifill = ifill + 1
        #print(f"Do Xtrain and Xtrain_check have the same shape and are element-wise equal within a tolerance? {Xtrain.shape == Xtrain_check.shape and np.allclose(Xtrain, Xtrain_check)}")
        #print(f"Do Ytrain and Ytrain_check have the same shape and are element-wise equal within a tolerance? {Ytrain.shape == Ytrain_check.shape and np.allclose(Ytrain, Ytrain_check)}")
        

        Xtest = Yaverage.copy()# (n_neurons, n_datapoints_test/n_T) array, average neural data across all conditions
        Ytest = Y[:,:,iconditiontest]# (n_neurons, n_datapoints_test/n_T) array
        
        distance_train, distance_test = computedistance(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)
        distance_train_cross_condition_average_baseline[iconditiontest] = distance_train
        distance_test_cross_condition_average_baseline[iconditiontest] = distance_test
    '''
         
    # Original
    Y = neuraldata.copy()
    [n_neurons, n_T, n_conditions] = Y.shape
    X = -700*np.ones((n_neurons, n_T, n_conditions))
    Yaverage = np.mean(neuraldata.copy(),2)# n_neurons x n_T array, average across conditions
    for icondition in range(n_conditions):
        X[:,:,icondition] =  Yaverage 
    #---------------
    distance_train_cross_condition_average_baseline, distance_test_cross_condition_average_baseline = computedistance_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)# (n_conditions,) array
    #---------------
    return distance_train_cross_condition_average_baseline, distance_test_cross_condition_average_baseline# (n_conditions,) array