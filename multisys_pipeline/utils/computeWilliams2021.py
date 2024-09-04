# Williams et al. 2021 "Generalized Shape Metrics on Neural Representations"
# https://github.com/ahwillia/netrep
# Symmetry: distance(X,Y) = distance(Y,X)
# Triangle Inequality: distance(X,Y) <= distance(X,Z) + distance(Z,Y)

import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
from netrep.metrics import LinearMetric


def computeWilliams2021(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, alpha=1, PLOTFIGURES=0):
    ##########################################################################
    # INPUTS
    # Xtrain: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain.
    # Ytrain: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain. 
    # Xtest:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before Williams2021, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before Williams2021, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that Williams2021 does not rely on low variance dimensions.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing Williams2021 distance
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing Williams2021 distance
    # If both options are None then don't perform PCA
    
    # Valid values for the regularization term are 0 <= alpha <= 1. When alpha == 0, the resulting metric is similar to CCA and allows for an invertible linear transformation to align the activations. 
    # When alpha == 1, the model is fully regularized and only allows for rotational alignments.
    # We recommend starting with the fully regularized model where alpha == 1.

    # OUTPUTS
    # Williams2021_train_distance: a number between 0(most similar) and something (can be greater than 1) quantifying the similarity between Xtrain and Ytrain, potentially after projecting Xtrain and Ytrain onto some lower dimensional space with PCA
    # Williams2021_test_distance: a number between 0(most similar) and something (can be greater than 1) quantifying the similarity between Xtest and Ytest, potentially after projecting Xtest and Ytest onto some lower dimensional space with PCA
    # The output of metric.score from Williams2021 is np.arccos(C) where C is some number that technically is bounded between -1 and 1, so Williams2021 distance is bounded between 0 and pi
    # However, C seems to always fall between 0 and 1 so then Williams2021 distance is bounded between 0 and pi/2
    ##########################################################################
    assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None)) > 0, "Error: One option can be None or both options can be None. If both options are None then PCA is not performed."
    
    if (VARIANCEEXPLAINED_KEEP is not None) or (n_PCs_KEEP is not None):# if both options are None then don't perform PCA 
        from sklearn.decomposition import PCA 
        for iteration in range(2):# 0) perform PCA on X, 1) perform PCA on Y
            if iteration==0: data = Xtrain; data_test = Xtest; namePCA = 'X'; figuresuffix = ''
            if iteration==1: data = Ytrain; data_test = Ytest; namePCA = 'Y'; figuresuffix = ''
            
            datadimensionality, n_datapoints = np.shape(data)# datadimensionality x n_datapoints array
            meandata = 1/n_datapoints * np.sum(data,1)# (datadimensionality,) array
            #dataminusmean_check = data - np.outer(meandata,np.ones(n_datapoints))# datadimensionality x n_datapoints array
            dataminusmean = data - meandata[:,None]# datadimensionality x n_datapoints array
            #print(f"Do dataminusmean and dataminusmean_check have the same shape and are element-wise equal within a tolerance? {dataminusmean.shape == dataminusmean_check.shape and np.allclose(dataminusmean, dataminusmean_check)}")
            
            # [u,s,v] = svd(A); A = u*s*v’; columns of u are eigenvectors of covariance matrix A*A’; rows of v’ are eigenvectors of covariance matrix A’*A; s is a diagonal matrix that has elements = sqrt(eigenvalues of A’*A and A*A’)
            #eigVec, eigVal, vT = np.linalg.svd(dataminusmean/np.sqrt(n_datapoints-1))# np.linalg.svd returns v transpose!
            #eigVal = eigVal**2# (datadimensionality,) array, largest first, np.sum(np.var(data, axis=1, ddof=1)) is the same as np.sum(eigVal)
            
            modelPCA = PCA(n_components = min(datadimensionality,n_datapoints)).fit(data.T)
            eigVal = modelPCA.explained_variance_# (datadimensionality,) array, largest is first 
            eigVec = modelPCA.components_.T# eigVec[:,i] is the ith eigenvector/principal component

            
            fraction = eigVal/np.sum(eigVal)# fraction of variance explained by each eigenvector
            if VARIANCEEXPLAINED_KEEP is not None:
                if VARIANCEEXPLAINED_KEEP==1:
                    n_PCs_KEEP = eigVec.shape[1]# keep all PCs
                else:
                    VARIANCEEXPLAINED = VARIANCEEXPLAINED_KEEP# a number from 0 to 1
                    n_PCs_KEEP = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
                    n_PCs_KEEP = n_PCs_KEEP[0]
                
            # project the data onto the first k eigenvectors
            k = datadimensionality
            k = n_PCs_KEEP
            datanew = np.matmul(eigVec[:,0:k].T, dataminusmean)# k(dimension) x n_datapoints array, np.var(datanew, axis=1, ddof=1) is the same as eigVal
            datanew_test = np.matmul(eigVec[:,0:k].T, (data_test - meandata[:,None]))# k(dimension) x n_datapoints array, np.var(datanew, axis=1, ddof=1) is the same as eigVal
            
            if PLOTFIGURES:
                VARIANCEEXPLAINED = 0.5# a number from 0 to 1
                n_PCs50 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
                n_PCs50 = n_PCs50[0]
                VARIANCEEXPLAINED = 0.9# a number from 0 to 1
                n_PCs90 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
                n_PCs90 = n_PCs90[0]
                VARIANCEEXPLAINED = 0.95# a number from 0 to 1
                n_PCs95 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
                n_PCs95 = n_PCs95[0]
                VARIANCEEXPLAINED = 0.99# a number from 0 to 1
                n_PCs99 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
                n_PCs99 = n_PCs99[0]
                
                import matplotlib.pyplot as plt
                # The sum of the variance-along-each-axis is preserved under a rotation of the axes, e.g. np.sum(eigVal) = np.sum(np.var(data, axis=1, ddof=1))
                # In other words, the variance of neuron 1 over time + variance of neuron 2 over time + ... + variance of last neuron over time = sum of variances after projecting data onto each principal component
                var_data = np.var(data.copy(), axis=1, ddof=1)# not sorted
                indices = np.argsort(-var_data)# sort the variances in decreasing order, so largest is first
                var_data = var_data[indices]# largest variance is first                  
                cumulative_fraction_var_data = np.cumsum(var_data) / np.sum(var_data)
                
                var_datanew = np.var(datanew.copy(), axis=1, ddof=1)# not sorted
                #indices = np.argsort(-var_datanew)# sort the variances in decreasing order, so largest is first
                #var_datanew = var_datanew[indices]# largest variance is first                  
                #cumulative_fraction_var_datanew = np.cumsum(var_datanew) / np.sum(var_datanew)# np.sum(var_datanew) may not be the same as np.sum(var_data)=np.sum(eigVal) if data is projected onto fewer dimensions
                cumulative_fraction_var_datanew = np.cumsum(var_datanew) / np.sum(eigVal)
                
                fig, ax = plt.subplots()# cumulative fraction of total variance-along-each-axis
                fontsize = 14
                handle1 = ax.plot(np.arange(1,datadimensionality+1), cumulative_fraction_var_data, 'r-', linewidth=3)
                handle2 = ax.plot(np.arange(1,datadimensionality+1), np.cumsum(eigVal)/np.sum(eigVal), 'k-', linewidth=3)
                handle3 = ax.plot(np.arange(1,k+1), cumulative_fraction_var_datanew, 'k.')# fraction of variance kept in k-dimensional projection
                ax.legend(handles=[handle1[0],handle2[0]], labels=['Original data','Data projected onto PCA axes'], loc='best', frameon=True)
                ax.set_xlabel('Number of axes', fontsize=fontsize)
                ax.set_ylabel('Cumulative fraction of total variance-along-each-axis', fontsize=fontsize)
                ax.set_title(f'{n_PCs50} principal components explain {100*np.sum(eigVal[0:n_PCs50])/np.sum(eigVal):.0f}% of the variance\n\
                             {n_PCs90} principal components explain {100*np.sum(eigVal[0:n_PCs90])/np.sum(eigVal):.0f}% of the variance\
                             \n{n_PCs95} principal components explain {100*np.sum(eigVal[0:n_PCs95])/np.sum(eigVal):.0f}% of the variance\n\
                {n_PCs99} principal components explain {100*np.sum(eigVal[0:n_PCs99])/np.sum(eigVal):.0f}% of the variance\
                \n{n_PCs_KEEP} principal components explain {100*np.sum(eigVal[0:n_PCs_KEEP])/np.sum(eigVal):.0f}% of the variance', fontsize=fontsize)
                ax.set_xlim(xmin=None, xmax=None); ax.set_ylim(ymin=0, ymax=None)
                ax.tick_params(axis='both', labelsize=fontsize)
                #fig.savefig('%s/PCA_variance_%s%s.pdf'%(dir,namePCA.replace(" ", ""),figuresuffix), bbox_inches='tight');# add bbox_inches='tight' to keep title from being cutoff
                fig.savefig('PCA_variance_%s%s.pdf'%(namePCA.replace(" ", ""),figuresuffix), bbox_inches='tight');# add bbox_inches='tight' to keep title from being cutoff
                
            if iteration==0: Xtrain = datanew.copy(); Xtest = datanew_test.copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)
            if iteration==1: Ytrain = datanew.copy(); Ytest = datanew_test.copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)  
        
    
    metric = LinearMetric(alpha=alpha, center_columns=True)
    metric.fit(Xtrain.T, Ytrain.T)# fit alignment transformations
    Williams2021_train_distance = metric.score(Xtrain.T, Ytrain.T)# evaluate distance between X and Y, using alignments fit above
    Williams2021_test_distance = metric.score(Xtest.T, Ytest.T)# evaluate distance between X and Y, using alignments fit above
    
    return Williams2021_train_distance, Williams2021_test_distance




def computeWilliams2021_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, alpha=1):
    ##########################################################################
    # INPUTS
    # X: (p, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
    # Y: (q, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # PCA eigenvectors are determined with data from X and Y across all conditions except one. The one condition serves as the test data for cross-validation. This process is repeated for all conditions.
    # For each condition create the following arrays and call the function computeWilliams2021
    # Xtrain = X[:,:,all-conditions-except-condition-icondition]: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain.
    # Ytrain = Y[:,:,all-conditions-except-condition-icondition]: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain.
    # Xtest = X[:,:,icondition]:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest = Y[:,:,icondition]:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before Williams2021, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before Williams2021, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that Williams2021 does not rely on low variance dimensions.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing Williams2021 distance
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing Williams2021 distance
    # If both options are None then don't perform PCA
    
    # Valid values for the regularization term are 0 <= alpha <= 1. When alpha == 0, the resulting metric is similar to CCA and allows for an invertible linear transformation to align the activations. 
    # When alpha == 1, the model is fully regularized and only allows for rotational alignments.
    # We reccomend starting with the fully regularized model where alpha == 1.

    # OUTPUTS
    # Williams2021_train: (n_conditions,) array, Williams2021_train[j] contains a number between 0 and something (can be greater than 1) quantifying the similarity between all data in X and Y that is not from condition icondition
    # Williams2021_test:  (n_conditions,) array, Williams2021_test[j] contains a number between 0 and something (can be greater than 1) quantifying the similarity between data from X[:,:,icondition] and Y[:,:,icondition]
    ##########################################################################
    assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None)) > 0, "Error: One option can be None or both options can be None. If both options are None then PCA is not performed."
    
    
    p, n_datapoints, n_conditions = X.shape
    q, n_datapoints, n_conditions = Y.shape
    Williams2021_train = -700*np.ones(n_conditions)
    Williams2021_test = -700*np.ones(n_conditions)
    for iconditiontest in range(n_conditions):# test on data from condition icondition
        
        iconditionstrain = np.arange(0,n_conditions)# indices of all conditions 
        iconditionstrain = np.delete(iconditionstrain, iconditiontest)# indices of all conditions except icondition
        
        Xtrain = X[:,:,iconditionstrain].reshape(p,n_datapoints*(n_conditions-1), order='F')# (p, n_datapoints_train) array
        Ytrain = Y[:,:,iconditionstrain].reshape(q,n_datapoints*(n_conditions-1), order='F')# (q, n_datapoints_train) array
        '''
        Xtrain_check = -700*np.ones((p,n_datapoints*(n_conditions-1)))# (p, n_datapoints_train) array
        Ytrain_check = -700*np.ones((q,n_datapoints*(n_conditions-1)))# (q, n_datapoints_train) array
        ifill = 0
        for icondition in iconditionstrain:
            Xtrain_check[:,ifill*n_datapoints:(ifill+1)*n_datapoints] = X[:,:,icondition]# (p, n_datapoints) array
            Ytrain_check[:,ifill*n_datapoints:(ifill+1)*n_datapoints] = Y[:,:,icondition]# (q, n_datapoints) array
            ifill = ifill + 1
        print(f"Do Xtrain and Xtrain_check have the same shape and are element-wise equal within a tolerance? {Xtrain.shape == Xtrain_check.shape and np.allclose(Xtrain, Xtrain_check)}")
        print(f"Do Ytrain and Ytrain_check have the same shape and are element-wise equal within a tolerance? {Ytrain.shape == Ytrain_check.shape and np.allclose(Ytrain, Ytrain_check)}")
        '''
        Xtest = X[:,:,iconditiontest]# (p, n_datapoints_test) array
        Ytest = Y[:,:,iconditiontest]# (q, n_datapoints_test) array
        
        Williams2021_train_, Williams2021_test_ = computeWilliams2021(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP, alpha=alpha, PLOTFIGURES=0)    
        Williams2021_train[iconditiontest] = Williams2021_train_
        Williams2021_test[iconditiontest] = Williams2021_test_
 
    return Williams2021_train, Williams2021_test


'''
# a note on stacking data
data = np.arange(0,24).reshape(3,4,2, order='F')
# data[:,:,0] = 
# array([[0,  3,  6,  9],
#        [1,  4,  7, 10],
#        [2,  5,  8, 11]])
# data[:,:,1] = 
# array([[12, 15, 18, 21],
#        [13, 16, 19, 22],
#        [14, 17, 20, 23]])

A = data.reshape(3,8,order='F')# if the second dimension of data represents time, e.g. data[neuron,time,condition], then this reshaping preserves the ordering of timepoints
# array([[0,  3,  6,  9, 12, 15, 18, 21],
#        [1,  4,  7, 10, 13, 16, 19, 22],
#        [2,  5,  8, 11, 14, 17, 20, 23]])

B = data.reshape(3,8)
# array([[0, 12,  3, 15,  6, 18,  9, 21],
#        [1, 13,  4, 16,  7, 19, 10, 22],
#        [2, 14,  5, 17,  8, 20, 11, 23]])
'''


#%%############################################################################
if __name__ == "__main__":# execute example code below if running .py file as the main program, but don't execute code below if importing function
    # Example
    # If two timeseries are perfectly anticorrelated then the correlation coefficient is -1 but the Williams2021 distance is 0
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(start=0, stop=2*np.pi, num=100, endpoint=True)# (100,) array
    y1 = np.sin(x)
    y2 = y1 + 1.5
    y3 = -y1
    y4 = np.cos(x)
    
    #y1 = np.array([1,0])# np.cos(0), np.sin(0)
    #y2 = np.array([np.cos(np.pi), np.sin(np.pi)])
    #normalizer = np.linalg.norm(y1.ravel()) * np.linalg.norm(y2.ravel())
    #np.dot( y1.flatten(), y2.flatten())
    #np.dot( y1.ravel(), y2.ravel())
     
    corr12 = np.corrcoef(y1, y2)[0,1]# 1
    corr13 = np.corrcoef(y1, y3)[0,1]# -1
    corr14 = np.corrcoef(y1, y4)[0,1]# 0
    distance12, distance12 = computeWilliams2021(y1[np.newaxis,:],y2[np.newaxis,:], y1[np.newaxis,:],y2[np.newaxis,:])# 0
    distance13, distance13 = computeWilliams2021(y1[np.newaxis,:],y3[np.newaxis,:], y1[np.newaxis,:],y3[np.newaxis,:])# 0
    distance14, distance14 = computeWilliams2021(y1[np.newaxis,:],y4[np.newaxis,:], y1[np.newaxis,:],y4[np.newaxis,:])# pi/2 (maximum)
    
    fig, ax = plt.subplots()
    fontsize = 11
    ax.plot(x, y1, 'k-', linewidth=3, label='y1')
    ax.plot(x, y2, 'r-', linewidth=3, label='y2')
    ax.plot(x, y3, 'g-', linewidth=3, label='y3')
    ax.plot(x, y4, 'b-', linewidth=3, label='y4')
    ax.legend(frameon=True)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_ylabel('Signal value', fontsize=fontsize)
    ax.set_title(f'If two timeseries are perfectly anticorrelated then\nthe correlation coefficient is -1 but the Williams2021 distance is 0\nPearson correlation coefficient y1/y2 = {corr12:.3g}, y1/y3 = {corr13:.3g}, y1/y4 = {corr14:.3g}\nWilliams2021 distance y1/y2 = {distance12:.3g}, y1/y3 = {distance13:.3g}, y1/y4 = {distance14:.6g} ', loc='right', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('computeWilliams2021_example1.pdf', bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


    #--------------------------------------------------------------------------
    # plot Williams2021/(np.pi/2) distance versus 1 - abs(Pearson correlation coefficient)
    n_iterations = 100
    distance_store = -700*np.ones(n_iterations)
    correlation_store = -700*np.ones(n_iterations)
    np.random.seed(1)# set random seed for reproducible results
    for i in range(n_iterations):
        n_T = np.random.randint(2,51)# number of timesteps
        if i < n_iterations/2:
            x = np.random.randn(n_T)
            y = np.random.randn(n_T)
        else:
            x = np.linspace(start=0, stop=2*np.pi, num=n_T, endpoint=True)# (n_T,) array
            y = x + np.random.randn(n_T)
        
        d, d = computeWilliams2021(x[np.newaxis,:],y[np.newaxis,:],x[np.newaxis,:],y[np.newaxis,:])
        distance_store[i] = d / (np.pi/2)
        correlation_store[i] = 1 - np.abs(np.corrcoef(x, y)[0,1])
        
        
    fig, ax = plt.subplots()
    fontsize = 9
    ax.plot(correlation_store, distance_store, 'k.', markersize=10)
    ymin, ymax = ax.get_ylim()
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k--', linewidth=0.5)
    ax.set_xlabel('1 - abs(correlation)', fontsize=fontsize)
    ax.set_ylabel('Procrustes distance / (pi/2)', fontsize=fontsize)
    ax.set_title('Procrustes distance / (pi/2) is approximately equal to 1 - abs(correlation)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    #ax.axis('equal')
    ax.set_xlim(xmin=-0.1, xmax=1.1); ax.set_ylim(ymin=-0.1, ymax=1.1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    fig.savefig('computeWilliams2021_example2.pdf', bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


