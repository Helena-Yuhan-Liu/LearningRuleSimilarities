import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions

#import sys
from multisys_pipeline.utils.computedistance import computedistance

# Return a distance that is normalized between ~0(best) and ~1(worst)
# 
# Split neural data into nonoverlapping groups each containing n_sample neurons (ineurons1, ineurons2)
# Sample n_sample units from the RNN model (iunits)
# Compute distance between two samples of neural data d1 = D(ineurons2, ineurons1). d1 is the lowest we can hope to get given the variability in the neurons that were recorded.
# Compute distance between samples of the model and neural data d2 = D(iunits, ineurons1)
# Map distances from old interval [d1,1] to new interval [0,1]. I'm assuming d1 and d2 are normalized distances between [0,1].
# distance_normalized = (d2 - d1) / (1 - d1)

# Do the same thing for the cross-condition average dataset. Sample ineurons2 from cross-condition average dataset and compute d3 = D(ineurons2, ineurons1)
# Note that ineurons2 is a different set of neurons than ineurons1, as this more closely replicates the comparison to a model that has different neurons
# distance_normalized_baseline = (d3 - d1) / (1 - d1)

# For each iteration of this procedure we get a new estimate for the normalized distance.
# The final normalized distance is the mean of all of these estimates.


def computedistance_normalizedwithdata(X_neuraldata, Y, similarityname, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, n_sample=10, n_iterations=100, random_seed=1):
     # INPUTS
     # X_neuraldata: (p, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
     # Y:            (q, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
     # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
     # similarityname: string 'CCA' 'CKA' 'metricCKA' 'Williams2021'
     
     # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before computing the distance measure, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
     # n_PCs_KEEP: Before computing the distance measure, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
     # Xtrain and Ytrain are first reduced using PCA. This ensures that the distance measure does not rely on low variance dimensions.
     # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing the distance
     # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing the distance
     # If both options are None then don't perform PCA
    
     # n_sample: all distances are computed between between n_sample neurons/units
     # n_iterations: compute normalized distance n_iterations times with different samples from X and Y on each iteration
     # random_seed: random seed that determines the random samples
     
     # OUTPUTS
     # distance_normalized (n_iterations,) array contains a number between 0(best) and 1(worst) quantifying the similarity between all data in X_neuraldata and Y 
     # distance_normalized_baseline (n_iterations,) array contains a number between 0(best) and 1(worst) quantifying the similarity between all data in X_neuraldata and X_crossconditionaverage
     ##########################################################################
     assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None)) > 0, "Error: One option can be None or both options can be None. If both options are None then PCA is not performed."
     n_sample_max = np.floor(X_neuraldata.shape[0]/2)
     assert n_sample <= n_sample_max, "Error: to compute the neural/neural distance the neural data is split into two nonoverlapping groups so n_sample must not be greater than n_neurons/2"
     
     # cross-condition average baseline for model/data similarity is computed using only neural data
     # compare original neural dataset to one in which each neuron's firing rate across conditions is replaced by a single time varying firing rate that is obtained after averaging across conditions (each neuron can have a different time varying firing rate in this baseline dataset)
     [n_neurons, n_T, n_conditions] = X_neuraldata.shape
     X_crossconditionaverage = -700*np.ones((n_neurons, n_T, n_conditions))
     Xaverage = np.mean(X_neuraldata.copy(),2)# n_neurons x n_T array, average across conditions
     for icondition in range(n_conditions):
         X_crossconditionaverage[:,:,icondition] =  Xaverage.copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)  
     #import sys; sys.exit()# stop script at current line
     
     rng = np.random.default_rng(random_seed)
     distance_normalized = -700*np.ones(n_iterations)# contains a number between 0(best) and 1(worst) quantifying the similarity between all data in X_neuraldata and Y 
     distance_normalized_baseline = -700*np.ones(n_iterations)# contains a number between 0(best) and 1(worst) quantifying the similarity between all data in X_neuraldata and X_crossconditionaverage
     distance_neuraltoneuralsubsampled_between0and1 = -700*np.ones(n_iterations)
     distance_modeltoneuralsubsampled_between0and1 = -700*np.ones(n_iterations)
     distance_crossconditionaverageneuraltoneuralsubsampled_between0and1 = -700*np.ones(n_iterations)
     distance_normalizedwithdata_output = {}# empty dictionary
     
     if similarityname == 'DSA':
         X_neuraldata = np.transpose(X_neuraldata.copy(), axes=[2, 1, 0]) # b x T x N
         Y = np.transpose(Y.copy(), axes=[2, 1, 0])
         
     for iteration in range(n_iterations):
         ineurons1 = rng.permutation(n_neurons)[0:n_sample]# (n_sample,) array 
         ineurons2 = np.arange(0,n_neurons)# indices of all neurons
         ineurons2 = np.delete(ineurons2, ineurons1)# indices of all neurons except ineurons1
         ineurons2 = rng.permutation(ineurons2)# permute ineurons2
         assert len(np.intersect1d(ineurons1, ineurons2)) == 0, "Error: ineurons1 and ineurons2 should not contain any overlap"
         ineurons2 = ineurons2[0:n_sample]# (n_sample,) array 
         iunits = rng.permutation(Y.shape[0])[0:n_sample]# (n_sample,) array 
         #---------------
         # Xtrain: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain. The CCA weights for Xtrain and Xtest are determined with training data from Xtrain (after performing PCA) 
         # Ytrain: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain. The CCA weights for Ytrain and Ytest are determined with training data from Ytrain (after performing PCA) 
         # Xtest:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
         # Ytest:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
         # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
         #---------------
         # Compute distance between two samples of neural data d1 = D(ineurons2, ineurons1). d1 is the lowest we can hope to get given the variability in the neurons that were recorded.
         if similarityname == 'DSA':
             Xtrain = X_neuraldata[:,:,ineurons1]
             Ytrain = X_neuraldata[:,:,ineurons2]
         else:
             Xtrain = X_neuraldata[ineurons1,:,:].reshape(ineurons1.size,n_T*n_conditions, order='F')
             Ytrain = X_neuraldata[ineurons2,:,:].reshape(ineurons2.size,n_T*n_conditions, order='F')
         Xtest = Xtrain.copy(); Ytest = Ytrain.copy()
         '''
         Xtrain_check = -700*np.ones((ineurons1.size,n_T*n_conditions))
         Ytrain_check = -700*np.ones((ineurons2.size,n_T*n_conditions))
         ifill = 0
         for icondition in range(n_conditions):
             Xtrain_check[:,ifill*n_T:(ifill+1)*n_T] = X_neuraldata[ineurons1,:,icondition]
             Ytrain_check[:,ifill*n_T:(ifill+1)*n_T] = X_neuraldata[ineurons2,:,icondition]
             ifill = ifill + 1
         print(f"Do Xtrain and Xtrain_check have the same shape and are element-wise equal within a tolerance? {Xtrain.shape == Xtrain_check.shape and np.allclose(Xtrain, Xtrain_check)}")
         print(f"Do Ytrain and Ytrain_check have the same shape and are element-wise equal within a tolerance? {Ytrain.shape == Ytrain_check.shape and np.allclose(Ytrain, Ytrain_check)}")
         '''
         d1, d1_test = computedistance(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1=1)# a number between 0(most similar) and 1
         #---------------
         # Compute distance between samples of the model and neural data d2 = D(iunits, ineurons1)
         if similarityname == 'DSA':
             train = X_neuraldata[:,:,ineurons1]
             Ytrain = Y[:,:,iunits]
         else:
             Xtrain = X_neuraldata[ineurons1,:,:].reshape(ineurons1.size,n_T*n_conditions, order='F')
             Ytrain = Y[iunits,:,:].reshape(iunits.size,n_T*n_conditions, order='F')
         Xtest = Xtrain.copy(); Ytest = Ytrain.copy()
         d2, d2_test = computedistance(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1=1)# a number between 0(most similar) and 1
         #---------------
         # Do the same thing for the cross-condition average dataset. Sample ineurons2 from cross-condition average dataset and compute d3 = D(ineurons2, ineurons1)
         # Note that ineurons2 is a different set of neurons than ineurons1, as this more closely replicates the comparison to a model that has different neurons
         if similarityname == 'DSA':
             Xtrain = X_neuraldata[:,:,ineurons1] 
             Ytrain = X_crossconditionaverage[ineurons2,:,:]
             Ytrain = np.transpose(Ytrain.copy(), axes=[2, 1, 0])
         else:
             Xtrain = X_neuraldata[ineurons1,:,:].reshape(ineurons1.size,n_T*n_conditions, order='F')
             Ytrain = X_crossconditionaverage[ineurons2,:,:].reshape(ineurons2.size,n_T*n_conditions, order='F')
         Xtest = Xtrain.copy(); Ytest = Ytrain.copy()
         d3, d3_test = computedistance(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1=1)# a number between 0(most similar) and 1
         #---------------
         # Map distances from old interval [d1,1] to new interval [0,1]
         d2_normalized = (d2 - d1) / (1 - d1)
         d3_normalized = (d3 - d1) / (1 - d1)
         distance_normalized[iteration] = d2_normalized# contains a number between 0(best) and 1(worst) quantifying the similarity between all data in X_neuraldata and Y 
         distance_normalized_baseline[iteration] = d3_normalized# contains a number between 0(best) and 1(worst) quantifying the similarity between all data in X_neuraldata and X_crossconditionaverage
         #---------------
         distance_neuraltoneuralsubsampled_between0and1[iteration] = d1
         distance_modeltoneuralsubsampled_between0and1[iteration] = d2
         distance_crossconditionaverageneuraltoneuralsubsampled_between0and1[iteration] = d3
     distance_normalizedwithdata_output['distance_normalizedwithdata'] = distance_normalized# (n_iterations,) array
     distance_normalizedwithdata_output['distance_normalizedwithdata_baseline'] = distance_normalized_baseline# (n_iterations,) array
     distance_normalizedwithdata_output['distance_neuraltoneuralsubsampled_between0and1'] = distance_neuraltoneuralsubsampled_between0and1# (n_iterations,) array
     distance_normalizedwithdata_output['distance_modeltoneuralsubsampled_between0and1'] = distance_modeltoneuralsubsampled_between0and1# (n_iterations,) array
     distance_normalizedwithdata_output['distance_crossconditionaverageneuraltoneuralsubsampled_between0and1'] = distance_crossconditionaverageneuraltoneuralsubsampled_between0and1# (n_iterations,) array
     #return distance_normalized, distance_normalized_baseline, distance_neuraltoneuralsubsampled_between0and1, distance_modeltoneuralsubsampled_between0and1, distance_crossconditionaverageneuraltoneuralsubsampled_between0and1# (n_iterations,) arrays
     return distance_normalizedwithdata_output  



