
# from multisys_pipeline.models.generateINandTARGETOUT_Hatsopoulos2007_onehotbumpinput_holdcue import generateINandTARGETOUT_Hatsopoulos2007_onehotbumpinput_holdcue# from file import function
# from multisys_pipeline.models.generateINandTARGETOUT_Hatsopoulos2007_onehotextendedinput_holdcue import generateINandTARGETOUT_Hatsopoulos2007_onehotextendedinput_holdcue# from file import function
# from multisys_pipeline.models.generateINandTARGETOUT_Hatsopoulos2007_sincosextendedinput_holdcue import generateINandTARGETOUT_Hatsopoulos2007_sincosextendedinput_holdcue# from file import function
from multisys_pipeline.models.generateINandTARGETOUT_Hatsopoulos2007_sincosbumpinput_holdcue import generateINandTARGETOUT_Hatsopoulos2007_sincosbumpinput_holdcue# from file import function
# from multisys_pipeline.models.generateINandTARGETOUT_Hatsopoulos2007_sincosextendedinput_gocueoffset import generateINandTARGETOUT_Hatsopoulos2007_sincosextendedinput_gocueoffset# from file import function

def generateINandTARGETOUT_Hatsopoulos2007(task_input_dict):
    task_name = task_input_dict['task_name']# switch between different functions depending on the string task_name
    
    
    if task_name == 'Hatsopoulos2007_sincosbumpinput_holdcue':
        return generateINandTARGETOUT_Hatsopoulos2007_sincosbumpinput_holdcue(task_input_dict)
    
   


#%%############################################################################
#                       test generateINandTARGETOUT
###############################################################################
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    figure_dir = os.path.dirname(__file__)# return the folder path for the file we're currently executing

    
    task_name = 'Hatsopoulos2007_onehotbumpinput_holdcue'
    task_name = 'Hatsopoulos2007_onehotextendedinput_holdcue'
    task_name = 'Hatsopoulos2007_sincosextendedinput_holdcue' 
    task_name = 'Hatsopoulos2007_sincosbumpinput_holdcue'
    task_name = 'Hatsopoulos2007_sincosextendedinput_gocueoffset'
    generateINandTARGETOUT = generateINandTARGETOUT_Hatsopoulos2007
    
    np.random.seed(123)# set random seed for reproducible results
    n_T = 300
    T = np.arange(0,n_T)# (n_T,)
    if task_name == 'Hatsopoulos2007_onehotbumpinput_holdcue': n_input = 9# 8 one-hot inputs encode the reach condition + 1 input for the hold-cue
    if task_name == 'Hatsopoulos2007_onehotextendedinput_holdcue': n_input = 9# 8 one-hot inputs encode the reach condition + 1 input for the hold-cue
    if task_name == 'Hatsopoulos2007_sincosextendedinput_holdcue': n_input = 3# 2 inputs (sine(orientation) and cosine(orientation)) encode the reach condition + 1 input for the hold-cue  
    if task_name == 'Hatsopoulos2007_sincosbumpinput_holdcue': n_input = 3# 2 inputs (sine(orientation) and cosine(orientation)) encode the reach condition + 1 input for the hold-cue  
    if task_name == 'Hatsopoulos2007_sincosextendedinput_gocueoffset': n_input = 3# 2 inputs (sine(orientation) and cosine(orientation)) encode the reach condition + 1 input for the go-cue  
    n_trials = 100
    toffsetoutput = 0# number of timesteps EMG output is offset, no change if toffsetoutput=0, if toffsetoutput is negative then target EMG outuput is earlier in the trial
    interval1set=np.array([0]); interval2set=np.array([0])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    #interval1set=np.array([49]); interval2set=np.array([149])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    interval1set=np.arange(0,50); interval2set=np.arange(0,150)
    
    OUTPUTONEHOT=0; OUTPUTXYHANDPOSITION=0; OUTPUTXYHANDVELOCITY=0; OUTPUTSHOULDERELBOWANGLES=1# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES
    n_output = 0
    if OUTPUTONEHOT: n_output = n_output+ 8# one-hot output for each reach condition
    if OUTPUTXYHANDPOSITION: n_output = n_output + 2
    if OUTPUTXYHANDVELOCITY: n_output = n_output + 2
    if OUTPUTSHOULDERELBOWANGLES: n_output = n_output + 2# output shoulder and elbow angle
    
    task_input_dict = {'task_name':task_name, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTSHOULDERELBOWANGLES':OUTPUTSHOULDERELBOWANGLES}
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT_Hatsopoulos2007(task_input_dict)   
    TARGETOUT_icondition = task_output_dict['TARGETOUT_icondition']
    
    plt.figure()# inputs and target outputs 
    for itrial in range(n_trials):
        plt.clf()
        #----colormaps----
        cool = cm.get_cmap('cool', n_input)
        colormap_input = cool(range(n_input))# (n_input, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        copper = cm.get_cmap('copper_r', n_output)
        colormap_output = copper(range(n_output))# (n_output, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        #----plot all inputs and outputs----
        ilabelsinlegend = np.round(np.linspace(0,n_input-1,5,endpoint=True))# if there are many inputs only label 5 of them in the legend
        for i in range(n_input):# 0,1,2,...n_input-1
            if np.isin(i,ilabelsinlegend):
                plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3, label=f'Input {i+1}')# label inputs 1,2,3,..
            else:
                plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3)# don't label these inputs
        ilabelsinlegend = np.round(np.linspace(0,n_output-1,5,endpoint=True))# if there are many outputs only label 5 of them in the legend
        for i in range(n_output):# 0,1,2,...n_output-1
            if np.isin(i,ilabelsinlegend):
                plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3, label=f'Output {i+1}')# label outputs 1,2,3,..
            else:
                plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3)# don't label these outputs
        #---------------------
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.title(f'Trial {itrial}, reach condition {TARGETOUT_icondition[itrial]:.2g}')
        plt.xlim(left=0)
        #plt.savefig('%s/generateINandTARGETOUT_Sussillo2015_trial%g.pdf'%(figure_dir,itrial), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.show()
        input("Press Enter to continue...")# pause the program until the user presses Enter
     



