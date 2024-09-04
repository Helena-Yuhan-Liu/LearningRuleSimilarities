
import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



#%%############################################################################
#                   principal component analysis (PCA)
###############################################################################
# In Python, a variable declared outside of the function or in global scope is known as a global variable. This means that a global variable can be accessed inside or outside of the function.
#def computePCA(data, Tdata, figure_xlabel ,figure_dir, figure_suffix, figure_title, colormapforconditions):# data has shape n_neurons x n_T x n_conditions
def computePCA(data, PLOTFIGURES=0, PCAinput_dictionary={}):# data has shape n_neurons x n_T x n_conditions
    # data: (n_neurons, n_T, n_conditions) array, this is the only information that enters into the PCA computation, the rest of the inputs are for the figures
    # Tdata: (n_T,) array of times used in figure
    # save figures to figure_dir
    # example: figure_xlabel = f'Time relative to {Elabelneural} (ms)'
    # example: figure_suffix = f'_{n_parameter_updates_model}parameterupdates' 
    # PCAinput_dictionary = {'Tdata':Tdata, 'figure_xlabel':figure_xlabel, 'figure_dir':figure_dir, 'figure_suffix':figure_suffix, 'figure_title':figure_title, 'colormapforconditions':colormapforconditions}
    if PLOTFIGURES:# currently PCAinput_dictionary is only storing parameters for figures
        Tdata = PCAinput_dictionary['Tdata']
        figure_xlabel = PCAinput_dictionary['figure_xlabel']
        figure_dir = PCAinput_dictionary['figure_dir']
        figure_suffix = PCAinput_dictionary['figure_suffix']
        figure_title = PCAinput_dictionary['figure_title']
        colormapforconditions = PCAinput_dictionary['colormapforconditions']
        n_figuresplot = 1 # 5# default value
        # if 'n_figuresplot' in PCAinput_dictionary:# check if key exists in dictionary
        #     n_figuresplot = PCAinput_dictionary['n_figuresplot']# plot data over time projected onto PC1, PC2, PC3,...,PCn_figuresplot
        
    [n_neurons, n_T, n_conditions] = data.shape
    data_check = -700*np.ones((n_neurons, n_T*n_conditions))
    for i in range(n_conditions):
        data_check[:,i*n_T:(i+1)*n_T] = data[:,:,i].copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)  
    data = data.reshape(n_neurons, n_T*n_conditions, order='F')# the same as matlab command: data2 = data(:,:)
    assert data.shape == data_check.shape and np.allclose(data, data_check), "Error: data and data_check do not have the same shape and are not element-wise equal within a tolerance."
    
    datadimensionality, n_datapoints = np.shape(data)# datadimensionality x n_datapoints array
    meandata = 1/n_datapoints * np.sum(data,1)# (datadimensionality,) array
    dataminusmean = data - meandata[:,np.newaxis]# datadimensionality x n_datapoints array
    
    n_components = np.minimum(datadimensionality, n_datapoints)# this can be less than datadimensionality if n_datapoints < datadimensionality 
    #modelPCA = PCA(n_components = datadimensionality).fit(data.T)
    modelPCA = PCA(n_components = n_components).fit(data.T)
    eigVal = modelPCA.explained_variance_# (n_components,) array, largest is first, eigVal[j] is the amount of variance explained by the jth principal component/eigenvector, i.e. the variance of the data after projecting onto the jth principal component/eigenvector  
    eigVec = modelPCA.components_.T# (datadimensionality, n_components) array, eigVec[:,i] is the ith eigenvector/principal component
    
    #eigVal = np.zeros(datadimensionality); eigVal[np.random.randint(0,datadimensionality)] = np.abs(np.random.randn())# check dimensionality 1: if all the variance is concentrated in a single dimension then only one eigenvalue is nonzero and dimensionality = 1
    #eigVal = np.abs(np.random.randn()) * np.ones(datadimensionality)# check dimensionality 2: if variance is evenly spread across all datadimensionality dimensions then all eigenvalues are the same and dimensionality = datadimensionality
    dimensionality_participation_ratio = np.sum(eigVal)**2 / np.sum(eigVal**2)
    #print(f'dimensionality = {dimensionality_participation_ratio}')
    
    # project the data onto the first k eigenvectors
    k = n_components
    datanew = eigVec[:,0:k].T @ dataminusmean# k(dimension) x n_datapoints array, np.var(datanew, axis=1, ddof=1) is the same as eigVal
    
    # project the (possibly reduced dimensionality) data back onto the original basis
    # databacktooriginalbasis = eigVec[:,0:k] @ datanew + np.outer(meandata,np.ones(n_datapoints))# datadimensionality x n_datapoints array
    
    fraction = eigVal/np.sum(eigVal)# fraction of variance explained by each eigenvector/principal component
    VARIANCEEXPLAINED = 0.99# a number from 0 to 1
    n_PCs99 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
    n_PCs99 = n_PCs99[0]
    VARIANCEEXPLAINED = 0.95# a number from 0 to 1
    n_PCs95 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
    n_PCs95 = n_PCs95[0]
    VARIANCEEXPLAINED = 0.9# a number from 0 to 1
    n_PCs90 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
    n_PCs90 = n_PCs90[0]
    VARIANCEEXPLAINED = 0.5# a number from 0 to 1
    n_PCs50 = np.where(np.cumsum(fraction) >= VARIANCEEXPLAINED)[0] + 1# minimum number of principal components required to explain at least VARIANCEEXPLAINED% of the variance
    n_PCs50 = n_PCs50[0]
    
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
    
    if PLOTFIGURES:
        fig, ax = plt.subplots()# cumulative fraction of total variance-along-each-axis
        fontsize = 13
        handle1 = ax.plot(np.arange(1,datadimensionality+1), cumulative_fraction_var_data, 'r-', linewidth=3)
        handle2 = ax.plot(np.arange(1,n_components+1), np.cumsum(eigVal)/np.sum(eigVal), 'k-', linewidth=3)
        handle3 = ax.plot(np.arange(1,k+1), cumulative_fraction_var_datanew, 'k.')# fraction of variance kept in k-dimensional projection
        ax.legend(handles=[handle1[0],handle2[0]], labels=['Original data','Data projected onto PCA axes'], loc='best', frameon=False, fontsize=fontsize-1)
        ax.set_xlabel('Number of axes', fontsize=fontsize)
        ax.set_ylabel('Cumulative fraction of total variance-along-each-axis', fontsize=fontsize)
        ax.set_title(figure_title + f'PCA on {n_neurons} neurons, {n_T} timesteps, {n_conditions} conditions\n'
                 f'{n_PCs99} principal components explain {100*np.sum(eigVal[0:n_PCs99])/np.sum(eigVal):.0f}% of the variance\n'
                 f'{n_PCs95} principal components explain {100*np.sum(eigVal[0:n_PCs95])/np.sum(eigVal):.0f}% of the variance\n'
                 f'{n_PCs90} principal components explain {100*np.sum(eigVal[0:n_PCs90])/np.sum(eigVal):.0f}% of the variance\n'
                 f'{n_PCs50} principal components explain {100*np.sum(eigVal[0:n_PCs50])/np.sum(eigVal):.0f}% of the variance', fontsize=fontsize)   
        ax.set_xlim(xmin=None, xmax=None); ax.set_ylim(ymin=0, ymax=None)
        #ax.set_xticks(np.arange(1,datadimensionality+1)); ax.set_xticklabels(np.arange(1,datadimensionality+1))
        #ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels([0, 0.5, 1])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', labelsize=fontsize)
        fig.savefig('%s/PCA_variance_nTPCA%g%s.pdf'%(figure_dir,n_T,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        
        datanewreshape = -700*np.ones((k,n_T,n_conditions))
        for i in range(n_conditions):
            datanewreshape[:,:,i] = datanew[:, i*n_T:(i+1)*n_T]
        
        for iPC in range(n_figuresplot):# plot data over time projected onto PC1, PC2, PC3,...,PCn_figuresplot
            fig, ax = plt.subplots()
            for icondition in range(n_conditions):
                ax.plot(Tdata, datanewreshape[iPC,:,icondition], '-', linewidth=3, c = colormapforconditions[icondition,:])
            #minylim, maxylim = plt.ylim(); ax.plot(np.zeros(100),np.linspace(minylim,maxylim,100), 'k--', linewidth=1)# vertical line at x=0
            #ax.set_xlabel(f'Time relative to {Elabel} (ms)', fontsize=fontsize)
            ax.set_xlabel(figure_xlabel, fontsize=fontsize)
            ax.set_ylabel(f'Data projected onto PC {iPC+1}', fontsize=fontsize)
            ax.set_title(figure_title + f'principal component {iPC+1} explains {100*fraction[iPC]:.5g}% of the variance', fontsize=fontsize)
            frame1 = plt.gca(); 
            frame1.axes.get_yaxis().set_ticks([])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)# ax.spines['bottom'].set_visible(False); 
            ax.tick_params(axis='both', labelsize=fontsize)
            fig.savefig('%s/PCA_PC%govertime_nTPCA%g%s.pdf'%(figure_dir,iPC+1,n_T,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

    
    PCAoutput_dictionary = {}# empty dictionary
    PCAoutput_dictionary['dimensionality'] = dimensionality_participation_ratio
    return PCAoutput_dictionary





#%%############################################################################
#%                          test computePCA
###############################################################################
if __name__ == "__main__":
    import numpy as np
    from matplotlib import cm
    import os
    figure_dir = os.path.dirname(__file__) + '/'# return the folder path for the file we're currently executing
    #root_dir = '/om/user/cjcueva/test_rnn'
    os.chdir(figure_dir)# print(f'current working direction is {os.getcwd()}')
    
    ###########################################################################
    #                        generate synthetic data
    n_T = 300# number of timesteps 
    T = np.arange(n_T)# 0,1,...,n_T-1
    n_conditions = 5# Number of conditions. For example, each condition in a motor experiment could be a different reach direction. Or each condition could be a different visual stimuli on a computer screen.
    n_neurons = 50# number of neurons
    datatype1 = 'random walk'; datatype2 = 'Gaussian' 
    datatype1 = 'random walk'; datatype2 = 'Binary +1 and -1'
    datatype1 = 'Ornstein-Uhlenbeck'; datatype2 = ''# datatype2 = f'randseed{randseed}'
    
    dataALL = -700*np.ones((n_neurons, n_T, n_conditions))
    for icondition in range(n_conditions):
        randseed = icondition# set random seed for reproducible results
        
        # Gaussian random walk data
        if datatype1=='random walk':
            np.random.seed(randseed)# set random seed for reproducible results
            data= np.zeros((n_neurons,n_T))
            for itrial in range(n_neurons):
                for t in range(n_T-1):
                    if datatype2=='Gaussian': data[itrial,t+1] = data[itrial,t] + np.random.randn()
                    if datatype2 == 'Binary +1 and -1': data[itrial,t+1] = data[itrial,t] + 2*np.random.randint(0,2)-1# convert 0,1 to -1,1
        
        # Ornstein-Uhlenbeck process: dx(t) = theta * (mu - x(t)) * dt + sigma * dW
        # The covariance is cov[OU(t),OU(t+T)] = E[(OU(t) - E[OU(t)]) * (OU(t+T) - E[OU(t+T)])] = sigma^2/(2*theta) * exp(-theta*T)
        # The equilibrium distribution of OU is a Gaussian with variance sigma^2/(2*theta)
        if datatype1=='Ornstein-Uhlenbeck':
            np.random.seed(randseed)# set random seed for reproducible results
            theta = 1000# correlation time = 1/theta, larger theta = faster decay
            sigma = np.sqrt(2*theta*3.5)# set variance = sigma^2/(2*theta) = C, sigma = sqrt(2*theta*C)
            dt = 1e-5# 1e-5, set the discrete time stepsize
            n_sim = n_neurons# number of simulations
            mu = 0# In general, the solution starts at OU(0) and over time moves towards the value mu, but experiences random "wobbles" whose size is determined by sigma. Increasing theta makes the solution move towards the mean faster.
            dw = np.sqrt(dt) * np.random.randn(n_sim,n_T)# compute the Brownian increments
            OU = -700*np.ones((n_sim,n_T)) 
            OU[:,0] = mu + np.sqrt(sigma**2/(2*theta)) * np.random.randn(n_sim)# set initial variance = final variance = sigma^2/(2*theta), standard deviation = sqrt(sigma^2/(2*theta))
            #OU[:,0] = 20;
            for isim in range(n_sim):
                for j in range(n_T-1):
                    OU[isim,j+1] = OU[isim,j] + theta*(mu - OU[isim,j])*dt + sigma*dw[isim,j]# Euler approximate integration process
            data = OU  
        dataALL[:,:,icondition] = data
    
    ###########################################################################
    #                principal component analysis (PCA)
    figure_xlabel = 'Time'
    figure_suffix = ''
    figure_title = f'{n_neurons} neurons, {n_T} timesteps, {n_conditions} conditions\n'# examples: figure_title = '' or figure_title = f'first line\nsecond line'
    n_figuresplot = 2# plot data over time projected onto PC1, PC2, PC3,...,PCn_figuresplot
    #------
    n_curves = n_conditions
    turbo = cm.get_cmap('turbo', n_curves)
    colormap = turbo(range(n_curves))# (n_curves, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
    colormapforconditions = colormap
    #------
    PCAinput_dictionary = {'Tdata':T, 'figure_xlabel':figure_xlabel, 'figure_dir':figure_dir, 'figure_suffix':figure_suffix, 'figure_title':figure_title, 'colormapforconditions':colormapforconditions, 'n_figuresplot':n_figuresplot}
    PCAoutput_dictionary = computePCA(dataALL, PLOTFIGURES=1, PCAinput_dictionary=PCAinput_dictionary)

