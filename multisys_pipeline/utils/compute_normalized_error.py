
import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions


def compute_normalized_error(TARGETOUT, output, output_mask):# all inputs are arrays with shape (n_trials, n_T, n_output)
    ##########################################################################
    # INPUTS
    # TARGETOUT:   (n_trials, n_T, n_output) array. Desired target output for the RNN
    # output:      (n_trials, n_T, n_output) array. Output from the RNN 
    # output_mask: (n_trials, n_T, n_output) array. The elements of output_mask are 0 or 1. Only timepoints and trials where output_mask is 1 are included in the error
    
    # normalized error, if the RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
    # outputforerror = output(output_mask==1)
    # TARGETOUTforerror = TARGETOUT(output_mask==1)
    # errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
    
    # OUTPUTS
    # errornormalized: a number between 0 and something (can be greater than 1)
    ##########################################################################
    n_trials, n_T, n_output = TARGETOUT.shape
    errornormalized = 0
    for i in range(n_output):
        outputforerror = output[:,:,i]# (n_trials, n_T)
        outputforerror = outputforerror[output_mask[:,:,i]==1][:,None]# (something,1)
        TARGETOUTforerror = TARGETOUT[:,:,i]# (n_trials, n_T)
        TARGETOUTforerror = TARGETOUTforerror[output_mask[:,:,i]==1][:,None]# (something,1)
        A = outputforerror - TARGETOUTforerror
        B = np.mean(TARGETOUTforerror) - TARGETOUTforerror
        if np.all(B==0):# TARGETOUTforerror is a constant
            errornormalized = errornormalized + (A.T @ A) 
        else:
            errornormalized = errornormalized + (A.T @ A) / (B.T @ B)# normalized error when using outputs for which output_mask = 1
        #B = torch.mean(TARGETOUTforerror) - TARGETOUTforerror
        #errornormalized = errornormalized + (A.transpose(0,1) @ A) / (B.transpose(0,1) @ B)# normalized error when using outputs for which output_mask = 1
    #errornormalized = torch.squeeze(errornormalized) / n_output# use torch.squeeze so normalized error goes from having shape (1,1) to being a number, this makes it easier to plot in figure titles, etc.
    errornormalized = np.squeeze(errornormalized) / n_output# use np.squeeze so normalized error goes from having shape (1,1) to being a number, this makes it easier to plot in figure titles, etc.
    return errornormalized



#%%############################################################################
#                       test generateINandTARGETOUT
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    TEST_OPTION = 1
    # if TEST_OPTION=1 test that if the RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
    # if TEST_OPTION=0 test that errornormalized varies continuously as the standard deviation of TARGETOUT varies
    
    np.random.seed(1)# set random seed for reproducible results 
    std_set = np.linspace(0,1,num=10,endpoint=True)
    n_T = 14# number of timesteps in each trial
    n_trials = 4# number of trials
    n_output = 3# number of outputs from the RNN
    output_mask = np.ones((n_trials, n_T, n_output))
    
    errornormalized_store = -700*np.ones(std_set.size)
    std_TARGETOUT_store = -700*np.ones(std_set.size)
    for i in range(std_set.size):
        output = np.random.randn(n_trials, n_T, n_output)
        TARGETOUT = std_set[i]*np.random.randn(n_trials, n_T, n_output)
        
        if TEST_OPTION:
            for j in range(n_output):
                output[:,:,j] = np.mean(TARGETOUT[:,:,j])
                
        std_TARGETOUT_store[i] = np.std(TARGETOUT)
        errornormalized_store[i] = compute_normalized_error(TARGETOUT, output, output_mask)
        
        
    
    fig, ax = plt.subplots()
    fontsize = 12
    ax.plot(std_TARGETOUT_store, errornormalized_store, '.-', markersize=20, linewidth=4, color='k')
    ax.set_xlabel('Standard deviation of TARGETOUT', fontsize=fontsize)
    ax.set_ylabel('Normalized error\nbetween output and TARGETOUT', fontsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    #fig.savefig('%s/main_scatterplot%snormalizedwithdata_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similarityname,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    
        
