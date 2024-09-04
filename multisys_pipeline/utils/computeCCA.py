import numpy as np
#import sys
import rcca


def computeCCA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, n_CCs_KEEP=None, PLOTFIGURES=0):
    ##########################################################################
    # INPUTS
    # Xtrain: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Xtrain and Xtest are determined with training data from Xtrain. The CCA weights for Xtrain and Xtest are determined with training data from Xtrain (after performing PCA) 
    # Ytrain: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The PCA eigenvectors for Ytrain and Ytest are determined with training data from Ytrain. The CCA weights for Ytrain and Ytest are determined with training data from Ytrain (after performing PCA) 
    # Xtest:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before CCA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before CCA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that CCA does not find dimensions of high correlation but low data variance.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing CCA correlations
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing CCA correlations
    # If n_CCs_KEEP is specified then don't perform PCA and keep n_CCs_KEEP canonical components
    
    # OUTPUTS
    # returns numbers between 0 and 1(most similar)
    # r_pyrcca_train: (something,) array of correlation coefficients between the canonical component pairs of Xtrain and Ytrain, potentially after projecting Xtrain and Ytrain onto some lower dimensional space with PCA
    # r_pyrcca_test:  (something,) array of correlation coefficients between the canonical component pairs of Xtest and Ytest, potentially after projecting Xtest and Ytest onto some lower dimensional space with PCA
    ##########################################################################
    assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None) + (n_CCs_KEEP is None)) == 2, "All options should be None except for one"
    
    if n_CCs_KEEP is None:# if we explicitly specify the number of canonical components to keep then don't perform PCA 
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
           
    p, n_datapoints = Xtrain.shape
    q, n_datapoints = Ytrain.shape
    n_CC = np.minimum(p,q)
    if n_CCs_KEEP is not None: n_CC = n_CCs_KEEP
    
    # subtract the mean
    Xmean = np.mean(Xtrain,1)[:,None]
    Ymean = np.mean(Ytrain,1)[:,None]
    Xminusmean = Xtrain - Xmean# subtract the mean across rows, so the mean of each row is 0
    Yminusmean = Ytrain - Ymean# subtract the mean across rows, so the mean of each row is 0
   
      
    ##############################################################################
    #%% 
    # Bilenko et al. 2016 “Pyrcca: regularized kernel canonical correlation analysis in Python and its applications to neuroimaging”
    # https://github.com/google/svcca/
    # INPUT TO CCA.TRAIN MUST BE MEAN CENTERED DATA!!
    
    # Only one of the following two options should be selected. These options change r_pyrcca_test but not r_pyrcca_train
    correlation_between_canonical_component_pairs = 1# Returns n_CC correlations between Xtest and Ytest after projecting onto canonical components from Xtrain and Ytrain
    correlation_between_data_and_reconstruction = 0# For each dimension in the test data, correlations between predicted and actual data are computed. cca.validate returns p+q correlations (between each dimension of predicted and actual data) and is described by equation (13) from Bilenko et al. 2016 except the test data (both Xtest and Ytest) are zscored before the predictions are made. zscore means each of the p/q test dimensions has mean 0 and std 1. When comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps. So after zscoring, each neuron's activity over time has mean=0 and std=1.

    
    #cca = rcca.CCA(kernelcca = False, reg = 0., numCC = n_CC)
    cca = rcca.CCA(kernelcca = False, reg = 1e-3, numCC = n_CC, verbose=False)
    #cca = rcca.CCA(kernelcca = False, reg = 0)
    #cca.train([X.transpose(), Y.transpose()])# inputs are n_datapoints x p, and n_datapoints x q arrays 
    cca.train([Xminusmean.transpose(), Yminusmean.transpose()])# inputs are n_datapoints x p, and n_datapoints x q arrays 
    r_pyrcca = cca.cancorrs# correlations between canonical variate pairs, correlations between the canonical components for X and Y
    r_pyrcca_train = r_pyrcca
    
    if correlation_between_canonical_component_pairs:
        A_pyrcca, B_pyrcca = cca.ws
        # A: p x n_CC matrix, The jth column of A contains the linear weights that multiply mean centered X to yield the jth canonical component of X
        # B: q x n_CC matrix, The jth column of B contains the linear weights that multiply mean centered Y to yield the jth canonical component of Y
        U_pyrcca, V_pyrcca = cca.comps# canonical variate pairs: U contains the canonical components for X, V contains the canonical components for Y
        U_pyrcca = U_pyrcca.transpose()# n_CC x n_datapoints matrix, U contains the canonical components for X, i.e. weights (in columns of A) multiplied by mean centered X
        V_pyrcca = V_pyrcca.transpose()# n_CC x n_datapoints matrix, V contains the canonical components for Y, i.e. weights (in columns of B) multiplied by mean centered Y
        
        '''
        # check self-consistency between correlations of the canonical components
        r1_check = np.corrcoef(U_pyrcca[0,:], V_pyrcca[0,:])[0,1]
        r2_check = np.corrcoef(U_pyrcca[1,:], V_pyrcca[1,:])[0,1]
        print(f"Self-consistency: Do correlations between first canonical components agree? {np.allclose(r_pyrcca[0],r1_check)}")
        print(f"Self-consistency: Do correlations between second canonical components agree? {np.allclose(r_pyrcca[1],r2_check)}")
        print(f"Self-consistency: Are U1 and V2 uncorrelated? corrcoef(U1,V2) = {np.corrcoef(U_pyrcca[0,:], V_pyrcca[1,:])[0,1]:.6g}")
        print(f"Self-consistency: Are U2 and V1 uncorrelated? corrcoef(U2,V1) = {np.corrcoef(U_pyrcca[1,:], V_pyrcca[0,:])[0,1]:.6g}")
        
        # check self-consistency of weights
        U_check = A_pyrcca.T @ (X - np.mean(X,1)[:,None])# n_CC x n_datapoints matrix, equal to U
        V_check = B_pyrcca.T @ (Y - np.mean(Y,1)[:,None])# n_CC x n_datapoints matrix, equal to V
        #U_check = A_pyrcca.T @ X# n_CC x n_datapoints matrix, not equal to U
        #V_check = B_pyrcca.T @ Y# n_CC x n_datapoints matrix, not equal to V
        print(f"Do U_pyrcca and U_check have the same shape and are element-wise equal within a tolerance? {U_pyrcca.shape == U_check.shape and np.allclose(U_pyrcca, U_check)}")
        print(f"Do V_pyrcca and V_check have the same shape and are element-wise equal within a tolerance? {V_pyrcca.shape == V_check.shape and np.allclose(V_pyrcca, V_check)}")
        print("")# add a line break
        '''
        
        # corr(X,Y) is the same as corr(a*X + b, m*Y + n) up to a possible sign change determined by sign(a)*sign(m)
        # so to find the correlation coefficient on the test data it doesn't matter if we subtract off some mean value or not
        U_test = A_pyrcca.T @ (Xtest - Xmean)# n_CC x n_datapoints matrix, equal to U
        V_test = B_pyrcca.T @ (Ytest - Ymean)# n_CC x n_datapoints matrix, equal to V
        #U_test = A_pyrcca.T @ Xtest# n_CC x n_datapoints matrix, equal to U
        #V_test = B_pyrcca.T @ Ytest# n_CC x n_datapoints matrix, equal to V
        
        #print(f'U_test.shape = {U_test.shape}')
        #import sys; sys.exit()# stop script at current line
        
        r_pyrcca_test = -700*np.ones(n_CC)
        for i in range(n_CC):
            r_pyrcca_test[i] = np.corrcoef(U_test[i,:], V_test[i,:])[0,1]
        
        
    if correlation_between_data_and_reconstruction:
        corrs = cca.validate([Xtest.transpose(), Ytest.transpose()])# same result with or without mean subtraction
        corrs_predictXfromY, corrs_predictYfromX = corrs# correlation between prediction-for-each-dimension-of-X and that dimension of X, correlation between prediction-for-each-dimension-of-Y and that dimension of Y
        # corrs_predictXfromY is a (p,) dimensional array
        # corrs_predictYfromX is a (q,) dimensional array
        r_pyrcca_test = (np.mean(corrs_predictXfromY) + np.mean(corrs_predictYfromX)) / 2
        
    return r_pyrcca_train, r_pyrcca_test



def computeCCA_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP=None, n_PCs_KEEP=None, n_CCs_KEEP=None):
    ##########################################################################
    # INPUTS
    # X: (p, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
    # Y: (q, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
    # For example, when comparing neural recordings taken over time, p and q = number of neurons and n_datapoints = number of timesteps
    
    # CCA weights are determined with data from X and Y across all conditions except one. The one condition serves as the test data for cross-validation. This process is repeated for all conditions.
    # For each condition create the following arrays and call the function computeCCA
    # Xtrain = X[:,:,all-conditions-except-condition-icondition]: (p, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The CCA weights are determined with training data from Xtrain and Ytrain (after performing PCA) 
    # Ytrain = Y[:,:,all-conditions-except-condition-icondition]: (q, n_datapoints_train) array, p and q can be different but n_datapoints_train must be the same for both datasets. The CCA weights are determined with training data from Xtrain and Ytrain (after performing PCA)
    # Xtest = X[:,:,icondition]:  (p, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    # Ytest = Y[:,:,icondition]:  (q, n_datapoints_test) array, p and q can be different but n_datapoints_test must be the same for both datasets
    
    # VARIANCEEXPLAINED_KEEP is a number from 0 to 1. Before CCA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto k eigenvectors such that the fraction of variance explained by k is at least VARIANCEEXPLAINED. The value of k may be different for Xtrain and Ytrain.
    # n_PCs_KEEP: Before CCA, perform PCA on Xtrain and Ytrain separately. Project Xtrain and Ytrain onto n_PCs_KEEP eigenvectors. 
    # Xtrain and Ytrain are first reduced using PCA. This ensures that CCA does not find dimensions of high correlation but low data variance.
    # Use the PCs/eigenvectors found on Xtrain to reduce the dimensionality of Xtest before computing CCA correlations
    # Use the PCs/eigenvectors found on Ytrain to reduce the dimensionality of Ytest before computing CCA correlations
    # If n_CCs_KEEP is specified then don't perform PCA and keep n_CCs_KEEP canonical components
    
    # OUTPUTS
    # meanr_pyrcca_train: (n_conditions,) array, meanr_pyrcca_train[j] contains the mean correlation coefficients between the canonical component pairs of all data in X and Y that is not from condition icondition
    # meanr_pyrcca_test:  (n_conditions,) array, meanr_pyrcca_test[j] contains the mean correlation coefficients between the canonical component pairs of data from X[:,:,icondition] and Y[:,:,icondition]
    ##########################################################################
    assert np.sum((VARIANCEEXPLAINED_KEEP is None) + (n_PCs_KEEP is None) + (n_CCs_KEEP is None)) == 2, "All options should be None except for one"
    p, n_datapoints, n_conditions = X.shape
    q, n_datapoints, n_conditions = Y.shape
    mean_r_pyrcca_train = -700*np.ones(n_conditions)
    mean_r_pyrcca_test = -700*np.ones(n_conditions)
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
        #if VARIANCEEXPLAINED_KEEP is not None:
        #    r_pyrcca_train, r_pyrcca_test = computeCCA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, PLOTFIGURES=0)
        #if n_PCs_KEEP is not None:
        #    r_pyrcca_train, r_pyrcca_test = computeCCA(Xtrain, Ytrain, Xtest, Ytest, n_PCs_KEEP=n_PCs_KEEP, PLOTFIGURES=0)
        r_pyrcca_train, r_pyrcca_test = computeCCA(Xtrain, Ytrain, Xtest, Ytest, VARIANCEEXPLAINED_KEEP=VARIANCEEXPLAINED_KEEP, n_PCs_KEEP=n_PCs_KEEP, n_CCs_KEEP=n_CCs_KEEP, PLOTFIGURES=0)    
        mean_r_pyrcca_train[iconditiontest] = np.mean(r_pyrcca_train)
        mean_r_pyrcca_test[iconditiontest] = np.mean(r_pyrcca_test)
 
    return mean_r_pyrcca_train, mean_r_pyrcca_test


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
    # If two timeseries are perfectly anticorrelated then the correlation coefficient is -1 but CCA is +1
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
    cca12_train, cca12 = computeCCA(y1[np.newaxis,:],y2[np.newaxis,:], y1[np.newaxis,:],y2[np.newaxis,:], n_CCs_KEEP=1)# 1
    cca13_train, cca13 = computeCCA(y1[np.newaxis,:],y3[np.newaxis,:], y1[np.newaxis,:],y3[np.newaxis,:], n_CCs_KEEP=1)# 1
    cca14_train, cca14 = computeCCA(y1[np.newaxis,:],y4[np.newaxis,:], y1[np.newaxis,:],y4[np.newaxis,:], n_CCs_KEEP=1)# 0
    fig, ax = plt.subplots()
    fontsize = 11
    ax.plot(x, y1, 'k-', linewidth=3, label='y1')
    ax.plot(x, y2, 'r-', linewidth=3, label='y2')
    ax.plot(x, y3, 'g-', linewidth=3, label='y3')
    ax.plot(x, y4, 'b-', linewidth=3, label='y4')
    ax.legend(frameon=True)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_ylabel('Signal value', fontsize=fontsize)
    ax.set_title(f'If two timeseries are perfectly anticorrelated then\nthe correlation coefficient is -1 but CCA is +1\nPearson correlation coefficient y1/y2 = {corr12:.3g}, y1/y3 = {corr13:.3g}, y1/y4 = {corr14:.3g}\nCCA similarity y1/y2 = {cca12[0]:.3g}, y1/y3 = {cca13[0]:.3g}, y1/y4 = {cca14[0]:.2g}', loc='right', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('computeCCA_example.pdf', bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


