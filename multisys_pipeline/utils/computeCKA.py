# centered kernel alignment (CKA)
# 1) Compute n_datapoints x n_datapoints similarity matrix for data in X and Y separately
# 2) Zero the row and column means of these square similarity matrices
# 3) Compute the similarity, via dot product, between the similarity matrices for X and Y
# 4) Normalize this so CKA is a number between 0 and 1
# Kornblith et al. 2019 "Similarity of Neural Network Representations Revisited"
# Nguyen et al. 2021 "Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth"
# https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ


import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions


def computeCKA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, PLOTFIGURES=0):
    ##########################################################################
    # INPUTS
    # Xtrain: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain.
    # Ytrain: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain. 
    # Xtest:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before CKA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before CKA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that CKA does not find dimensions of high correlation but low data variance.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing CKA correlations
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing CKA correlations
    # If both options are None then don't perform PCA
    
    # OUTPUTS
    # CKA_train: a number between 0 and 1(most similar) quantifying the similarity between Xtrain and Ytrain, potentially after projecting Xtrain and Ytrain onto some lower dimensional space with PCA
    # CKA_test: a number between 0 and 1(most similar) quantifying the similarity between Xtest and Ytest, potentially after projecting Xtest and Ytest onto some lower dimensional space with PCA
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
        
    
    # There is an equivalence between centered kernel alignment (CKA) and a sum involving principal components/eigenvectors and eigenvalues of the two datasets to be compared. 
    # equation 4 of Nguyen et al. 2021 "Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth"
    # equation 14 of Kornblith et al. 2019 "Similarity of Neural Network Representations Revisited"
    # In order for this equation to hold it seems two things must happen:
    # 1) The mean for each neuron has to be subtracted from the data before applying PCA. PCA is not done on the neural trajectory over time where there are number-of-timesteps datapoints but is done in a space with number-of-neurons (technically p or q) datapoints. We're finding PCA axes to discover structure across neurons.    
    # 2) The mean for each datapoint/timestep has to be subtracted from the data before applying CKA. This makes sense if we think of building up a number-of-datapoints x number-of-datapoints similarity matrix (Gram matrix) and want the entries to be dot products between zero-centered vectors, i.e. entries in the Gram matrix are like unnormalized correlation coefficients.   
    '''
    # For each datapoint/timestep subtract the mean across p/q responses. For each datapoint/timestep the mean across all p/q numbers is 0. This is not what Kornblith et al. do but their equation 14 doesn't seem to make sense unless this is assumed!
    # Do this so that the matrix comparison is similar to correlation, i.e. entries in the Gram matrix are like unnormalized correlation coefficients
    Xmean = np.mean(Xtrain,0,keepdims=True)# (1, n_datapoints_train)
    Ymean = np.mean(Ytrain,0,keepdims=True)# (1, n_datapoints_train)
    Xminusmean = Xtrain - Xmean# (p, n_datapoints_train) array, subtract the mean across rows, so the mean of each column is 0, np.mean(Xminusmean[:,j]) = 0
    Yminusmean = Ytrain - Ymean# (q, n_datapoints_train) array, subtract the mean across rows, so the mean of each column is 0, np.mean(Yminusmean[:,j]) = 0
    Xtrain = Xminusmean; Ytrain = Yminusmean
    Xtest = Xtest - np.mean(Xtest,0,keepdims=True)
    Ytest = Ytest - np.mean(Ytest,0,keepdims=True)
    '''
    
    '''
    # Linear CKA can be computed either based on dot products between examples or dot products between features:
    # dot-product( vec(X^T X), vec(Y^T Y) ) = Frobenius-norm(YX^T)^2
    # X and Y above are assumed to have shape (something, n_datapoints)
    # The formulation based on similarities between features (right-hand side) is faster than the formulation based on similarities between examples (left-hand side) when the number of examples exceeds the number of features. We provide both formulations here and demonstrate that they are equvialent.
    cka_from_examples = cka(gram_linear(Xtrain.transpose()), gram_linear(Ytrain.transpose()))# inputs to function are (n_datapoints, something) arrays
    cka_from_features = feature_space_linear_cka(Xtrain.transpose(), Ytrain.transpose())# inputs to function are (n_datapoints, something) arrays 
    print('Method 1: Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    print('Method 2: Linear CKA from Features: {:.5f}'.format(cka_from_features))
    np.testing.assert_almost_equal(cka_from_examples, cka_from_features)
    # METHOD 3
    # Nguyen et al. 2021 "Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth"
    p, n_datapoints = Xtrain.shape
    K = Xtrain.transpose() @ Xtrain# (n_datapoints, n_datapoints) Gram matrix
    L = Ytrain.transpose() @ Ytrain# (n_datapoints, n_datapoints) Gram matrix
    H = np.identity(n_datapoints) - np.ones((n_datapoints,n_datapoints)) / n_datapoints# (n_datapoints, n_datapoints) centering matrix
    Kprime = H @ K @ H# the mean across rows is 0, i.e. for each j np.sum(Kprime[:,j]) = 0, the mean across columns is zero, i.e. for each j np.sum(Kprime[j,:]) = 0
    Lprime = H @ L @ H# the mean across rows is 0, i.e. for each j np.sum(Kprime[:,j]) = 0, the mean across columns is zero, i.e. for each j np.sum(Kprime[j,:]) = 0
    HSIC_KL = np.dot( Kprime.flatten(), Lprime.flatten() ) / ((n_datapoints-1)**2)
    HSIC_KK = np.dot( Kprime.flatten(), Kprime.flatten() ) / ((n_datapoints-1)**2)
    HSIC_LL = np.dot( Lprime.flatten(), Lprime.flatten() ) / ((n_datapoints-1)**2)
    CKA_KL = HSIC_KL / np.sqrt( HSIC_KK * HSIC_LL)
    print('Method 3: Linear CKA from Features: {:.5f}'.format(CKA_KL))
    np.testing.assert_almost_equal(CKA_KL, cka_from_features)
    '''
    cka_from_features_train = feature_space_linear_cka(Xtrain.transpose(), Ytrain.transpose())# inputs are (n_datapoints, something) arrays 
    cka_from_features_test = feature_space_linear_cka(Xtest.transpose(), Ytest.transpose())# inputs are (n_datapoints, something) arrays 
    return cka_from_features_train, cka_from_features_test




##############################################################################
# https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)
##############################################################################




def computeCKA_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None):
    ##########################################################################
    # INPUTS
    # X: (p, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
    # Y: (q, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # PCA eigenvectors are determined with data from X and Y across all conditions except one. The one condition serves as the test data for cross-validation. This process is repeated for all conditions.
    # For each condition create the following arrays and call the function computeCKA
    # Xtrain = X[:,:,all-conditions-except-condition-icondition]: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain.
    # Ytrain = Y[:,:,all-conditions-except-condition-icondition]: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain.
    # Xtest = X[:,:,icondition]:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest = Y[:,:,icondition]:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before CKA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before CKA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that CKA does not find dimensions of high correlation but low data variance.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing CKA correlations
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing CKA correlations
    # If both options are None then don't perform PCA
    
    # OUTPUTS
    # CKA_train: (n_conditions,) array, CKA_train[j] contains a number between 0 and 1 quantifying the similarity between all data in X and Y that is not from condition icondition
    # CKA_test:  (n_conditions,) array, CKA_test[j] contains a number between 0 and 1 quantifying the similarity between data from X[:,:,icondition] and Y[:,:,icondition]
    ##########################################################################
    assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None)) > 0, "Error: One option can be None or both options can be None. If both options are None then PCA is not performed."
    
    
    p, n_datapoints, n_conditions = X.shape
    q, n_datapoints, n_conditions = Y.shape
    CKA_train = -700*np.ones(n_conditions)
    CKA_test = -700*np.ones(n_conditions)
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
        
        CKA_train_, CKA_test_ = computeCKA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP, PLOTFIGURES=0)    
        CKA_train[iconditiontest] = CKA_train_
        CKA_test[iconditiontest] = CKA_test_
 
    return CKA_train, CKA_test


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

if __name__ == "__main__":# execute example code below if running .py file as the main program, but don't execute code below if importing function
    # Example
    # If two timeseries are perfectly anticorrelated then the correlation coefficient is -1 but CKA is +1
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(start=0, stop=2*np.pi, num=100, endpoint=True)# (100,) array
    y1 = np.sin(x)
    y2 = y1 + 1.5
    y3 = -y1
    y4 = np.cos(x)
    corr12 = np.corrcoef(y1, y2)[0,1]# 1
    corr13 = np.corrcoef(y1, y3)[0,1]# -1
    corr14 = np.corrcoef(y1, y4)[0,1]# 0
    cka12, cka12 = computeCKA(y1[np.newaxis,:],y2[np.newaxis,:], y1[np.newaxis,:],y2[np.newaxis,:])# 1
    cka13, cka13 = computeCKA(y1[np.newaxis,:],y3[np.newaxis,:], y1[np.newaxis,:],y3[np.newaxis,:])# 1
    cka14, cka14 = computeCKA(y1[np.newaxis,:],y4[np.newaxis,:], y1[np.newaxis,:],y4[np.newaxis,:])# 0
    fig, ax = plt.subplots()
    fontsize = 11
    ax.plot(x, y1, 'k-', linewidth=3, label='y1')
    ax.plot(x, y2, 'r-', linewidth=3, label='y2')
    ax.plot(x, y3, 'g-', linewidth=3, label='y3')
    ax.plot(x, y4, 'b-', linewidth=3, label='y4')
    ax.legend(frameon=True)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_ylabel('Signal value', fontsize=fontsize)
    ax.set_title(f'If two timeseries are perfectly anticorrelated then\nthe correlation coefficient is -1 but CKA is +1\nPearson correlation coefficient y1/y2 = {corr12:.3g}, y1/y3 = {corr13:.3g}, y1/y4 = {corr14:.3g}\nCKA similarity y1/y2 = {cka12:.3g}, y1/y3 = {cka13:.3g}, y1/y4 = {cka14:.2g}', loc='right', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('computeCKA_example.pdf', bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


