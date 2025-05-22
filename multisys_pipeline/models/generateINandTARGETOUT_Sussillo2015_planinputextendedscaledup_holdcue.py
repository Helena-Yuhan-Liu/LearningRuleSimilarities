import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import scipy.io as sio# for loading .mat files
import os
root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))# os.path.dirname(__file__) returns the folder path for the file we're currently executing, .. goes up to the parent folder
data_dir = root_dir + '/experimental_data/Sussillo2015/'
parameters = sio.loadmat(data_dir + 'EMGforRNN_Sussillo2015')# print(parameters.keys())
EMG = parameters['EMGforRNN_Sussillo2015']# n_muscles(7) x n_TEMG(66) x n_reachconditions(27) array, this is the nonzero part of the EMG, time [-250:10:400] ms relative to movement onset
TEMG = np.arange(-250,400+10,10)# time [-250:10:400] ms relative to movement onset, this is the nonzero part of the EMG, 66 elements
#parameters = sio.loadmat(data_dir + 'neuralforRNN')# print(parameters.keys())
#neuralforRNN = parameters['neuralforRNN']# n_neurons(180) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
'''
parameters = sio.loadmat(data_dir + 'xyhandpositionforRNN')# print(parameters.keys())
xyhandpositionforRNN = parameters['xyhandpositionforRNN']# xy(2) x n_Txy(106) x n_reachconditions(27) array, time [-250:10:800] ms relative to movement onset, used the same start time as EMG to make things simple in generateINandTARGETOUT, the end time is later because sometimes the hand position doesn't reach the target until later
parameters = sio.loadmat(data_dir + 'xyhandvelocityforRNN')# print(parameters.keys())
xyhandvelocityforRNN = parameters['xyhandvelocityforRNN']# xy(2) x n_Txy(106) x n_reachconditions(27) array, time [-250:10:800] ms relative to movement onset, used the same start time as EMG to make things simple in generateINandTARGETOUT, the end time is later because sometimes the hand position doesn't reach the target until later
'''
Txy = np.arange(-250,800+10,10)# time [-250:10:800] ms relative to movement onset
n_Txy = Txy.size# 106

parameters = sio.loadmat(data_dir + 'goEnvelope')
gocue = np.squeeze(parameters['goEnvelope'])# (36,) array, time [-260:10:90] ms relative to movement onset in Mark Churchland's original demo code 
gocue = gocue[4:-5]# (27,) array, remove the first four and last five element because they are near zero, gocue[13] = 1 is the maximum
#import matplotlib.pyplot as plt; plt.plot(gocue,'k-')

parameters = sio.loadmat(data_dir + 'planinputforRNN')
planinputforRNN = np.squeeze(parameters['planinputforRNN'])# (15, 196, 27) array 
planinputforRNNsteadystate = planinputforRNN[:,100,:].copy()# (15, 27) array
planinputforRNNsteadystate = planinputforRNNsteadystate / np.max(np.abs(planinputforRNNsteadystate))# (15, 27) array. Scale up the input so max(abs(input)) = 1 where the input is across all 27 reach conditions. Note, this doesn't mean that each reach condition has max(abs(input)) = 1.
# planinputforRNN = planinputforRNN[:,60:-69,:]# (15, 67, 27) array, remove timesteps at the beginning and end to obtain the steady state values of planinputforRNN
# for icondition in range(27):# check that planinputforRNN only contains steady state values
#     for i in range(15):
#         #A0 = planinputforRNN[i,0,icondition]
#         A0 = planinputforRNNsteadystate[i,icondition]
#         assert np.allclose(A0, planinputforRNN[i,:,icondition]), "Error: planinputforRNN is not at steady state."
# import matplotlib.pyplot as plt; icondition = 26; plt.plot(planinputforRNN[:,:,icondition].T,'r-')
#import sys; sys.exit()# stop script at current line

'''
n_muscles, n_TEMG, n_conditions = EMG.shape# (7, 66, 27)
n_colors = n_conditions# create red-to-green colormap
colorsredgreen = np.stack((np.linspace(0.9,0,n_colors), np.linspace(0,0.9,n_colors), np.zeros(n_colors), np.ones(n_colors)), axis=-1)# (n_colors, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]      
colormap = colorsredgreen
figure_dir = data_dir

import matplotlib.pyplot as plt
for iteration in range(5):# reach variables, e.g. x hand position, for each of the 8 reach conditions over time
    if iteration==0: A = xyhandpositionforRNN[0,:,:].T; ylabel = 'x hand position'; Tplot = Txy; figurestring = 'xhandposition'; figure_suffix=''# figure_suffix = '_downsample'
    if iteration==1: A = xyhandpositionforRNN[1,:,:].T; ylabel = 'y hand position'; Tplot = Txy; figurestring = 'yhandposition'; figure_suffix=''# figure_suffix = '_downsample'
    if iteration==2: A = xyhandvelocityforRNN[0,:,:].T; ylabel = 'x hand speed'; Tplot = Txy; figurestring = 'xhandspeed'; figure_suffix = ''# figure_suffix = '_smoothposition'# figure_suffix = '_smoothspeed'# figure_suffix = '_downsample'# figure_suffix = ''
    if iteration==3: A = xyhandvelocityforRNN[1,:,:].T; ylabel = 'y hand speed'; Tplot = Txy; figurestring = 'yhandspeed'; figure_suffix = ''# figure_suffix = '_smoothposition'# figure_suffix = '_smoothspeed'# figure_suffix = '_downsample'# figure_suffix = ''
    if iteration==4: A = np.transpose(EMG, (2,1,0)); ylabel = 'EMG'; Tplot = TEMG; figurestring = 'EMG'; figure_suffix = ''# figure_suffix = '_smoothposition'# figure_suffix = '_smoothspeed'# figure_suffix = '_downsample'# figure_suffix = ''
    
    fig, ax = plt.subplots()
    fontsize = 10
    for icondition in range(n_conditions):# reach variables, e.g. x hand position, for each of the 8 reach conditions over time
        ax.plot(Tplot, A[icondition,:], linewidth=1, c=colormap[icondition,:])
    ymin, ymax = ax.get_ylim(); ax.plot(np.zeros(100), np.linspace(ymin, ymax, 100), 'k-', linewidth=1)# vertical line at 0
    ax.set_xlabel('Time relative to start of movement (ms)', fontsize=fontsize)
    ax.set_ylabel(f'{ylabel}', fontsize=fontsize)
    ax.set_title(f'{ylabel} for {n_conditions} reach conditions', fontsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('%s/generateINandTARGETOUT_Sussillo2015_%s%s.pdf'%(figure_dir,figurestring,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
import sys; sys.exit()# stop script at current line
'''


   


#%%###########################################################################
# Generate inputs and target outputs for training a recurrent neural network (RNN)
# The function generateINandTARGETOUT_Sussillo2015 generates the inputs and target outputs for training a RNN on a reaching task to one of 27 targets 
# On each trial the RNN receives inputs that specify the reach condition. When the reach condition input starts to turn off that's the go-cue for the reach to begin. The final hold-cue is redundant.
# After the go-cue the RNN must produce an output. The specific output depends on the binary options selected with OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
# Based on Sussillo, Churchland, Kaufman, Shenoy 2015 "A neural network that finds a naturalistic solution for the production of muscle activity"
##############################################################################
def generateINandTARGETOUT_Sussillo2015_planinputextendedscaledup_holdcue(task_input_dict={}):
#def generateINandTARGETOUT_Sussillo2015(n_input=28, n_output=7, n_T=229, n_trials=100, interval1set=np.arange(0,50), interval2set=np.arange(0,150), toffsetoutput=0, OUTPUTONEHOT=0, OUTPUTXYHANDPOSITION=0, OUTPUTXYHANDVELOCITY=0, OUTPUTEMG=1):
    
    #---------------------------------------------
    #                 INPUTS
    #---------------------------------------------
    # n_input:  number of input units, 15 numbers encode each reach condition + 1 input for the hold-cue. When the reach condition input starts to turn off that's the go-cue for the reach to begin. The final hold-cue is redundant.
    # n_output: number of output units
    # n_T:       number of timesteps in a trial
    # n_trials:  number of trials 
    # interval1set: interval before input specifying the reach condition, if interval1 is 0 then tstartcondition is at the very beginning of the trial (index 0)
    # interval2set: number of timesteps the reach condition input is at steady state, i.e. the number of timesteps after bumpon and before bumpoff
    # toffsetoutput: number of timesteps EMG output is offset, no change if toffsetoutput=0, if toffsetoutput is negative then target EMG output is earlier in the trial
    # select output: output depends on the binary options selected with OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    
    #---------------------------------------------
    #                OUTPUTS
    #---------------------------------------------
    # IN:        n_trials x n_T x n_input tensor
    # TARGETOUT: n_trials x n_T x n_output tensor
    # TARGETOUT_icondition: (n_trials,) elements are 0,1,...n_conditions(27)-1
    # output_mask:  n_trials x n_T x n_output tensor, elements 0(timepoint does not contribute to cost function), 1(timepoint contributes to cost function)
    #
    #---------------------------------------------
    n_input = task_input_dict['n_input']
    n_output = task_input_dict['n_output']
    n_T = task_input_dict['n_T']
    n_trials = task_input_dict['n_trials']
    interval1set = task_input_dict['interval1set']
    interval2set = task_input_dict['interval2set']
    toffsetoutput = task_input_dict['toffsetoutput']
    OUTPUTONEHOT = task_input_dict['OUTPUTONEHOT']
    OUTPUTXYHANDPOSITION = task_input_dict['OUTPUTXYHANDPOSITION']
    OUTPUTXYHANDVELOCITY = task_input_dict['OUTPUTXYHANDVELOCITY']
    OUTPUTEMG = task_input_dict['OUTPUTEMG']
      
       
    IN = np.zeros((n_trials, n_T, n_input))
    TARGETOUT = np.zeros((n_trials, n_T, n_output))
    output_mask = np.ones((n_trials, n_T, n_output))
    n_muscles, n_TEMG, n_reachconditions = EMG.shape
    assert n_input==16, "Error: input should be 15 numbers for each reach condition + 1 input for the hold-cue. When the reach condition input starts to turn off that's the go-cue for the reach to begin. The final hold-cue is redundant."
    #n_musclestrial = n_output
    n_musclestrial = 7# all muscles are outputted
    TARGETOUT_icondition = np.random.randint(0, high=n_reachconditions, size=n_trials)# (n_trials,) array, elements 0,1,...n_reachconditions-1
    if n_trials>=n_reachconditions: TARGETOUT_icondition[0:n_reachconditions] = np.arange(0,n_reachconditions)# first 27 trials contain the first 27 reach conditions
    # convert numbers to one-hot encoding
    #data = TARGETOUT_icondition
    #one_hot = np.zeros((data.size, n_reachconditionstrial))
    #one_hot[np.arange(data.size), data] = 1# (n_trials, n_reachconditionstrial) array
    
    #bumpon = np.array([0.1, 0.6, 0.9])# (3,) make inputs rounded so unit activity doesn't have sharp discontinuities when inputs turn on and off
    #bumpoff = np.flipud(bumpon)# (3,) make inputs rounded so unit activity doesn't have sharp discontinuities when inputs turn on and off
    #bump = np.concatenate((bumpon, np.ones(4), bumpoff))# (10,) make inputs rounded so unit activity doesn't have sharp discontinuities when inputs turn on and off
    
    bumpon = np.array([0.05, 0.1, 0.2,  0.35,  0.65,  0.8, 0.9, 0.95])# (8,) make inputs rounded so unit activity doesn't have sharp discontinuities when inputs turn on and off
    bumpoff = np.flipud(bumpon)# (8,) make inputs rounded so unit activity doesn't have sharp discontinuities when inputs turn on and off
    bump = np.concatenate((bumpon, np.ones(7), bumpoff))# (23,) make inputs rounded so unit activity doesn't have sharp discontinuities when inputs turn on and off
    #plt.figure(); plt.plot(bump)
    
    istartcondition = np.zeros(0)# stacking a vector of np.zeros(0) will not contribute elements
    iendcondition = np.zeros(0)# stacking a vector of np.zeros(0) will not contribute elements
    istartoutput = np.zeros(0)# stacking a vector of np.zeros(0) will not contribute elements
    iendoutput = np.zeros(0)# stacking a vector of np.zeros(0) will not contribute elements
    for itrial in range(0,n_trials):# 0, 1, 2, ... n_trials-1
        # timesteps for important events in the trial, timestep are 1,2,3,...n_T, these ultimately need to be translated to indices 0,1,2,...n_T-1
        interval1 = interval1set[np.random.randint(interval1set.size)]# interval before input specifying the condition, if interval1 is 0 then tstartcondition is at the very beginning of the trial (index 0)
        tstartcondition_startbumpon = interval1 + 1# timestep when tstartcondition starts, there are interval1 timesteps before tstartcondition starts
        tstartcondition_endbumpon = tstartcondition_startbumpon + bumpon.size-1
        interval2 = interval2set[np.random.randint(interval2set.size)]# number of timesteps the reach condition input is at steady state, i.e. the number of timesteps after bumpon and before bumpoff
        tendcondition_startbumpoff = tstartcondition_endbumpon + interval2 + 1
        tendcondition_endbumpoff = tendcondition_startbumpoff + bumpoff.size-1# bumpoff lasts from tendcondition_startbumpoff through tendcondition_endbumpoff (inclusive)
        tstarthold = 1# timestep 1 = index 0
        tendhold = tendcondition_endbumpoff# hold-cue is on from tstarthold through tendhold (inclusive)
        tstartoutput = tendcondition_startbumpoff# timestep when EMG starts (inclusive)
        tstartoutput = tstartoutput + toffsetoutput
        tendemg = np.minimum(n_T, tstartoutput + n_TEMG - 1)# timestep when EMG ends (inclusive), if tendemg isn't truncated then EMG output is on for n_TEMG(66) timesteps
        n_TEMGtrial = tendemg - tstartoutput + 1
        #print(f'tstartoutput = {tstartoutput}, tendemg = {tendemg}')
        
        # all timesteps from 1,2,3,...n_T are translated to indices 0,1,2,...n_T-1
        for i in range(15):# 15 numbers specify each reach condition
            IN[itrial,(tstartcondition_startbumpon-1):tstartcondition_endbumpon,i] = bumpon * planinputforRNNsteadystate[i,TARGETOUT_icondition[itrial]]
            IN[itrial,tstartcondition_endbumpon:(tendcondition_startbumpoff-1),i] = planinputforRNNsteadystate[i,TARGETOUT_icondition[itrial]]
            IN[itrial,(tendcondition_startbumpoff-1):tendcondition_endbumpoff,i] = bumpoff * planinputforRNNsteadystate[i,TARGETOUT_icondition[itrial]]

        IN[itrial,(tstarthold-1):tendhold,n_input-1] = 1# last input is hold cue
        IN[itrial,(tendhold-8):tendhold,n_input-1] = bumpoff# last input is hold cue
        if tstartoutput <= n_T:
            istartoutput = 0# starting index for output
            if OUTPUTONEHOT: 
                tend = n_T
                TARGETOUT[itrial,(tstartoutput-1):tend,istartoutput+TARGETOUT_icondition[itrial]] = 1# one-hot output for each reach condition until end of trial
                istartoutput = istartoutput + 27# one-hot output for each reach condition
            if OUTPUTXYHANDPOSITION: 
                n_Toutput = np.minimum(n_Txy, n_T-tstartoutput+1)# number of timesteps the output is on for
                tend = tstartoutput + n_Toutput - 1# output is on during timesteps [tstart:tend] inclusive
                TARGETOUT[itrial,(tstartoutput-1):tend,istartoutput:(istartoutput+2)] = xyhandpositionforRNN[:,0:n_Toutput,TARGETOUT_icondition[itrial]].transpose()
                output_mask[itrial,tend:,istartoutput:(istartoutput+2)] = 0# the output of the RNN for all timesteps after the xyhandposition are not counted in the loss function
                istartoutput = istartoutput + 2
            if OUTPUTXYHANDVELOCITY: 
                n_Toutput = np.minimum(n_Txy, n_T-tstartoutput+1)# number of timesteps the output is on for
                tend = tstartoutput + n_Toutput - 1# output is on during timesteps [tstart:tend] inclusive
                TARGETOUT[itrial,(tstartoutput-1):tend,istartoutput:(istartoutput+2)] = xyhandvelocityforRNN[:,0:n_Toutput,TARGETOUT_icondition[itrial]].transpose()
                output_mask[itrial,tend:,istartoutput:(istartoutput+2)] = 0# the output of the RNN for all timesteps after the xyhandposition are not counted in the loss function
                istartoutput = istartoutput + 2
            if OUTPUTEMG: 
                tend = tendemg.copy()
                TARGETOUT[itrial,(tstartoutput-1):tend,istartoutput:(istartoutput+n_musclestrial)] = EMG[0:n_musclestrial,0:n_TEMGtrial,TARGETOUT_icondition[itrial]].transpose()
                output_mask[itrial,tend:,istartoutput:(istartoutput+n_musclestrial)] = 0# the output of the RNN for all timesteps after the EMG are not counted in the loss function
                istartoutput = istartoutput + n_musclestrial# output 7 EMG
        
        # store trial info
        istartcondition = np.hstack((istartcondition, tstartcondition_startbumpon-1))# index (not timestep) when input specifying the condition starts
        iendcondition = np.hstack((iendcondition, tendcondition_endbumpoff-1))# index (not timestep) when input specifying the condition ends
        istartoutput = np.hstack((istartoutput, tstartoutput-1))# index (not timestep) when output starts. output is on during indices [istartoutput:iendoutput] inclusive
        iendoutput = np.hstack((iendoutput, tend-1))# index (not timestep) when output ends. output is on during indices [istartoutput:iendoutput] inclusive
    
    # convert to array of integers so we can use as indices
    istartcondition = istartcondition.astype(int)
    iendcondition = iendcondition.astype(int)
    istartoutput = istartoutput.astype(int)
    iendoutput = iendoutput.astype(int)
    
    # convert to pytorch tensors 
    dtype = torch.float32
    #IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype); output_mask = torch.from_numpy(output_mask, dtype=dtype); TARGETOUT_icondition = torch.from_numpy(TARGETOUT_icondition, dtype=dtype);
    IN = torch.tensor(IN, dtype=dtype); TARGETOUT = torch.tensor(TARGETOUT, dtype=dtype); output_mask = torch.tensor(output_mask, dtype=dtype); TARGETOUT_icondition = torch.tensor(TARGETOUT_icondition, dtype=dtype);
    #--------------------------------------------------------------------------
    task_output_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'TARGETOUT_icondition':TARGETOUT_icondition, 'istartcondition':istartcondition, 'iendcondition':iendcondition, 'istartoutput':istartoutput, 'iendoutput':iendoutput}
    return IN, TARGETOUT, output_mask, task_output_dict






#%%############################################################################
#                       test generateINandTARGETOUT
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    figure_dir = os.path.dirname(__file__)# return the folder path for the file we're currently executing

    
    np.random.seed(123); torch.manual_seed(123)# set random seed for reproducible results
    n_T = 229# if interval1 = 60 and interval2 = 80 then tendemg is 229
    n_T = 300
    T = np.arange(0,n_T)# (n_T,)
    
    n_input = 16# don't change, 15 numbers specify each reach condition
    n_trials = 100
    toffsetoutput = 0# number of timesteps EMG output is offset, no change if toffsetoutput=0, if toffsetoutput is negative then target EMG outuput is earlier in the trial
    interval1set=np.array([45]); interval2set=np.array([62])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    interval1set=np.array([49]); interval2set=np.array([149])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    #interval1set=np.array([6]); interval2set=np.array([6])# The way generateINandTARGETOUT_Sussillo2015_planinputextended_holdcue is currently written, if toffsetoutput is very negative and interval1set and interval2set are very small then the target output will try to start at negative timepoints and generate an error. For toffsetoutput = -20 the minimum values for the two intervals are interval1set=np.array([6]); interval2set=np.array([6])
    #interval1set=np.arange(0,50); interval2set=np.arange(0,80)
    
    OUTPUTONEHOT=0; OUTPUTXYHANDPOSITION=0; OUTPUTXYHANDVELOCITY=0; OUTPUTEMG=1# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    n_output = 0
    if OUTPUTONEHOT: n_output = n_output+ 27# one-hot output for each reach condition
    if OUTPUTXYHANDPOSITION: n_output = n_output + 2
    if OUTPUTXYHANDVELOCITY: n_output = n_output + 2
    if OUTPUTEMG: n_output = n_output + 7# output shoulder and elbow angle
    
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT_Sussillo2015_planinputextendedscaledup_holdcue(task_input_dict)   
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
        plt.title(f'EMG production task, trial {itrial}, reach condition {TARGETOUT_icondition[itrial]:.2g}')
        plt.xlim(left=0)
        #plt.savefig('%s/generateINandTARGETOUT_Sussillo2015_planinputextendedscaledup_holdcue_trial%g.pdf'%(figure_dir,itrial), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.show()
        input("Press Enter to continue...")# pause the program until the user presses Enter
     



