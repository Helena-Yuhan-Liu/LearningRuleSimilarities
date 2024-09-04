from dPCA import dPCA
import matplotlib.pyplot as plt




#%%############################################################################
#                 demixed principal component analysis (dPCA)
#   Kobak, Brendel, et al. 2016 "Demixed principal component analysis of neural population data"
###############################################################################
# In Python, a variable declared outside of the function or in global scope is known as a global variable. This means that a global variable can be accessed inside or outside of the function.
def computeDPCA(data, Tdata, figure_xlabel, figure_dir, figure_suffix, figure_title, colormapforconditions, n_figuresplotforeachcomponent=1, FLIPAXES=0):
    # data: (n_neurons, n_T, n_conditions) array, this is the only information that enters into the dPCA computation, the rest of the inputs are for the figures
    # Tdata: (n_T,) array of times used in figure
    # save figures to figure_dir
    # example: figure_xlabel = f'Time relative to {Elabelneural} (ms)'
    # example: figure_suffix = f'_{n_parameter_updates_model}parameterupdates' 
    
    #dpca = dPCA.dPCA(labels='ts',regularizer='auto')# we set regularizer to 'auto' to optimize the regularization parameter when we fit the data.
    [n_neurons, n_T, n_conditions] = data.shape
    dpca = dPCA.dPCA(labels='tc', n_components=n_neurons)
    dpca.protect = ['t']
    # Now fit the data using the model we just instatiated. Note that we only need trial-to-trial data when we want to optimize over the regularization parameter.
    #Z = dpca.fit_transform(data,trialdata)# Z.keys() gives the dictionary keys
    Z = dpca.fit_transform(data)# Z.keys() gives the dictionary keys
    
    A = dpca.explained_variance_ratio_# A.keys() gives the dictionary keys
    # np.sum(A['tc'])+np.sum(A['t'])+np.sum(A['c'])# shouldn't this be equal to 1??
    # I think the explained variance ratio python code is wrong.
    # Looking at the python code, it only uses the variance of the projection of the original data onto each decoder dimension to calculate the explained variance ratio. While this would technically work for PCA since the encoder and decoder matrices are the same and each vector has 2-norm of 1, here that's not true, so it doesn't work.
    # I think the original matlab code had it right, where you have to reproject using the encoding matrix, then calculate the variance explained.
    # https://github.com/machenslab/dPCA/issues/32
    
    #figure_suffix = figure_suffix + '_flip'
    for icomponent in range(3):# 0,1,2
        if icomponent==0: componentlabel = 'time component'; key = 't'
        if icomponent==1: componentlabel = 'condition component'; key = 'c'
        if icomponent==2: componentlabel = 'mixing component'; key = 'tc'

        for iplot in range(n_figuresplotforeachcomponent):# for each of the three components (time, condition, time x condition) plot n_figuresplotforeachcomponent figures    
            fig, ax = plt.subplots()
            fontsize = 13
            for icondition in range(n_conditions):
                 if FLIPAXES==0: ax.plot(Tdata, Z[key][iplot,:,icondition], '-', linewidth=2, c = colormapforconditions[icondition,:])
                 if FLIPAXES==1: ax.plot(Tdata, -Z[key][iplot,:,icondition], '-', linewidth=2, c = colormapforconditions[icondition,:])# flip
            #ax.set_xlabel(f'Time relative to {Elabel} (ms)', fontsize=fontsize)
            ax.set_xlabel(figure_xlabel, fontsize=fontsize)
            ax.set_ylabel(f'Data projected onto\n{componentlabel} {iplot+1}', fontsize=fontsize)
            ax.set_title(figure_title + f"{componentlabel} {iplot+1}, variance {100*A[key][iplot]:.6g}%", fontsize=fontsize)
            frame1 = plt.gca(); 
            frame1.axes.get_yaxis().set_ticks([])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)# ax.spines['bottom'].set_visible(False); 
            ax.tick_params(axis='both', labelsize=fontsize)
            fig.savefig('%s/dPCA_%s%g_nTDPCA%g%s.pdf'%(figure_dir,componentlabel.replace(" ", ""),iplot+1,n_T,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff



#%%############################################################################
#%                          test computeDPCA
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
    n_neurons = 100# number of neurons
    datatype1 = 'random walk'; datatype2 = 'Gaussian' 
    #datatype1 = 'random walk'; datatype2 = 'Binary +1 and -1'
    #datatype1 = 'Ornstein-Uhlenbeck'; datatype2 = ''# datatype2 = f'randseed{randseed}'
    
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
    #                demixed principal component analysis (dPCA)
    figure_xlabel = 'Time'
    figure_suffix = ''
    figure_title = f'{n_neurons} neurons, {n_T} timesteps, {n_conditions} conditions\n'# examples: figure_title = '' or figure_title = f'first line\nsecond line'
    n_figuresplotforeachcomponent = 1# for each of the three components (time, condition, time x condition) plot n_figuresplotforeachcomponent figures
    #------
    n_curves = n_conditions
    turbo = cm.get_cmap('turbo', n_curves)
    colormap = turbo(range(n_curves))# (n_curves, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
    colormapforconditions = colormap
    #------
    computeDPCA(dataALL, T, figure_xlabel, figure_dir, figure_suffix, figure_title, colormapforconditions, n_figuresplotforeachcomponent)# data has shape n_neurons x n_T x n_conditions
    
    
    
    
    
    
    
    
    
    