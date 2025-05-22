import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import copy
import os
root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))# os.path.dirname(__file__) returns the folder path for the file we're currently executing, .. goes up to the parent folder
data_dir_processed = root_dir + '/experimental_data/Hatsopoulos2007/'

#--------------------------------------------------------------------------
# To process the kinematic data we use the variables tstart, tend, and n_T_downsample
# To process the firing rates we use the variables tstart, tend, tpad, SD, T_downsample, dt_downsample
# The smoothed firing rates are downsampled by averaging across a sliding window of size dt_downsample centered around times in T_downsample
#--------------------------------------------------------------------------
tstart = -800# keep times [tstart:dt_downsample:tend] ms relative to start of movement 
tend = 400# keep times [tstart:dt_downsample:tend] ms relative to start of movement 
dt_downsample = 10# each timestep of the downsampled firing rate is dt_downsample ms apart
T_downsample = np.arange(tstart, tend+dt_downsample, step=dt_downsample)
n_T_downsample = T_downsample.size
SD = 25# standard deviation of Gaussian filter in ms applied to neural activity, this is not used on the kinematic data
tpad = 100# Compute firing rates for tpad=100 ms before and after our interval of interest [tstart, tend] and then truncate to the desired interval. This way the firing rates near the ends of our interval are more accurate.

#data_dir_processed = '/Users/christophercueva/Desktop/neural networks/zzz - data Hatsopoulos/'# load processed data from data_dir_processed
if os.path.exists(f'{data_dir_processed}/data_output_aligntomovementonset_minT{tstart}_maxT{tend}_FRsmoothSD{SD}ms.npy'):
    data_output_dict = np.load(f'{data_dir_processed}/data_output_aligntomovementonset_minT{tstart}_maxT{tend}_FRsmoothSD{SD}ms.npy', allow_pickle='TRUE').item()#np.save(f'{figure_dir}/data_output_aligntomovementonset_minT{tstart}_maxT{tend}_FRsmoothSD{SD}ms.npy', data_output)
else:
    data_dir_matfiles = '/Users/christophercueva/Desktop/neural networks/zzz - data Hatsopoulos/doi_10.5061_dryad.xsj3tx9cm__v2/'# load original Hatsopoulos2007 data from data_dir_matfiles
    data_input_dict = {'tstart':tstart, 'tend':tend, 'T_downsample':T_downsample, 'SD':SD, 'tpad':tpad, 'data_dir_matfiles':data_dir_matfiles, 'data_dir_processed':data_dir_processed}# dictionary
    data_output_dict = load_data_Hatsopoulos2007(data_input_dict)


#firing_rate = data_output_dict['firing_rate']# (n_units_nonzero, n_T_downsample, n_conditions) array
#plt.figure(); plt.plot(T_downsample, np.mean(firing_rate,0))
xhandposition = data_output_dict['xhandposition']# (n_conditions, n_T_downsample) array
yhandposition = data_output_dict['yhandposition']# (n_conditions, n_T_downsample) array
xhandspeed = data_output_dict['xhandspeed']# (n_conditions, n_T_downsample) array
yhandspeed = data_output_dict['yhandspeed']# (n_conditions, n_T_downsample) array
shoulderangle = data_output_dict['shoulderangle']# (n_conditions, n_T_downsample) array
elbowangle = data_output_dict['elbowangle']# (n_conditions, n_T_downsample) array
time_startmovement_to_endmovement_mean = data_output_dict['time_startmovement_to_endmovement_mean']# (n_conditions,) array, time in ms
time_startmovement_to_endmovement_median = data_output_dict['time_startmovement_to_endmovement_median']# (n_conditions,) array, time in ms
reach_direction = np.arange(0,360,45) * np.pi/180# [0:45:315] degrees   

# after the hold-cue turns off, output the ~nonzero part of the kinematic variables from -100 to tend ms relative to the start of the movement, i.e. relative to the times stored in T_downsample
xhandposition = xhandposition[:,T_downsample >= -100]# (8, 51) array
yhandposition = yhandposition[:,T_downsample >= -100]# (8, 51) array
xhandspeed = xhandspeed[:,T_downsample >= -100]# (8, 51) array
yhandspeed = yhandspeed[:,T_downsample >= -100]# (8, 51) array
shoulderangle = shoulderangle[:,T_downsample >= -100]# (8, 51) array
elbowangle = elbowangle[:,T_downsample >= -100]# (8, 51) array
n_Tdata = xhandposition.shape[1]

task_input_dict_default_test = {'n_input':3, 'n_output':2, 'n_T':200, 'n_trials':8, 'interval1set':np.array([25]), 'interval2set':np.array([75]), 'toffsetoutput':0, 'INPUT_ONE_HOT':0, 'INPUT_SINE_COSINE':1, 'OUTPUTONEHOT':0, 'OUTPUTXYHANDPOSITION':0, 'OUTPUTXYHANDVELOCITY':0, 'OUTPUTSHOULDERELBOWANGLES':1}
    

#%%###########################################################################
# Generate inputs and target outputs for training a recurrent neural network (RNN)
# The function generateINandTARGETOUT_Hatsopoulos2007 generates the inputs and target outputs for training a RNN on a center-out reaching task to one of 8 targets spaced at orientations [0:45:360)
# On each trial the RNN receives inputs that specify the reach orientation, and the go-cue when the reach should be initiated. 
# After the go-cue the RNN must produce an output. The specific output depends on the binary options selected with OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES
# Based on Hatsopoulos et al. 2007 "Encoding of Movement Fragments in the Motor Cortex" and Suresh, Goodman et al. 2020 "Neural population dynamics in motor cortex are different for reach and grasp"
##############################################################################
def generateINandTARGETOUT_Hatsopoulos2007_sincosbumpinput_holdcue(task_input_dict={}):
            
    #---------------------------------------------
    #                 INPUTS
    #---------------------------------------------
    # n_input:  number of input units
    # n_output: number of output units
    # n_T:       number of timesteps in a trial
    # n_trials:  number of trials 
    # interval1set: interval before input specifying the reach condition, if interval1 is 0 then tstartcondition is at the very beginning of the trial (index 0)
    # interval2set: interval after the input specifying the reach condition and before the hold-cue turns off
    # toffsetoutput: number of timesteps kinematic output, e.g. hand position, is offset. no change if toffsetoutput=0. if toffsetoutput is negative then target hand position output is earlier in the trial
    # select input: if INPUT_ONE_HOT=1 then 8 one-hot inputs encode the reach condition + 1 input for the hold-cue
    # select input: if INPUT_SINE_COSINE=1 then 2 inputs (sine(orientation) and cosine(orientation)) encode the reach condition + 1 input for the hold-cue
    # select output: output depends on the binary options selected with OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES
    
    #---------------------------------------------
    #                OUTPUTS
    #---------------------------------------------
    # IN:        n_trials x n_T x n_input tensor
    # TARGETOUT: n_trials x n_T x n_output tensor
    # TARGETOUT_icondition: (n_trials,) elements are 0,1,...n_conditions(8)-1
    # output_mask:  n_trials x n_T x n_output tensor, elements 0(timepoint does not contribute to cost function), 1(timepoint contributes to cost function)
    #
    #---------------------------------------------
    if ('use_default_test_parameters' in task_input_dict) and task_input_dict['use_default_test_parameters']==1:
        task_input_dict = copy.deepcopy(task_input_dict_default_test)
    n_input = task_input_dict['n_input']
    n_output = task_input_dict['n_output']
    n_T = task_input_dict['n_T']
    n_trials = task_input_dict['n_trials']
    interval1set = task_input_dict['interval1set']
    interval2set = task_input_dict['interval2set']
    toffsetoutput = task_input_dict['toffsetoutput']
    #INPUT_ONE_HOT = task_input_dict['INPUT_ONE_HOT']
    #INPUT_ONE_HOT = 1
    #INPUT_SINE_COSINE = task_input_dict['INPUT_SINE_COSINE']
    INPUT_SINE_COSINE = 1
    OUTPUTONEHOT = task_input_dict['OUTPUTONEHOT']
    OUTPUTXYHANDPOSITION = task_input_dict['OUTPUTXYHANDPOSITION']
    OUTPUTXYHANDVELOCITY = task_input_dict['OUTPUTXYHANDVELOCITY']
    OUTPUTSHOULDERELBOWANGLES = task_input_dict['OUTPUTSHOULDERELBOWANGLES']
    TEACHER_STUDENT_TRAINING = 0# 0(False) or 1(True), default is 0 
    if ('TEACHER_STUDENT_TRAINING' in task_input_dict):
        TEACHER_STUDENT_TRAINING = task_input_dict['TEACHER_STUDENT_TRAINING']
        
    #assert np.sum(INPUT_ONE_HOT + INPUT_SINE_COSINE) == 1, "Error: one, and only one, input type must be selected"
    #if INPUT_ONE_HOT: assert n_input==9, "Error: 8 one-hot inputs encode the reach condition + 1 input for the hold-cue"
    if INPUT_SINE_COSINE: assert n_input==3, "Error: 2 inputs (sine(orientation) and cosine(orientation)) encode the reach condition + 1 input for the hold-cue"
    
    # OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES can all be 1
    # in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES
    if TEACHER_STUDENT_TRAINING==0:
        n_output_check = 0
        if OUTPUTONEHOT==1: n_output_check = n_output_check + 8# one-hot output for each reach condition
        if OUTPUTXYHANDPOSITION==1: n_output_check = n_output_check + 2
        if OUTPUTXYHANDVELOCITY==1: n_output_check = n_output_check + 2
        if OUTPUTSHOULDERELBOWANGLES==1: n_output_check = n_output_check + 2# output shoulder and elbow angle
        assert n_output==n_output_check, "Error: n_output is not correct."
        
    IN = np.zeros((n_trials,n_T,n_input))
    TARGETOUT = np.zeros((n_trials,n_T,n_output))
    output_mask = np.ones((n_trials,n_T,n_output))
    n_reachconditions, n_Tdata = xhandposition.shape
    
    
    #n_musclestrial = n_output
    #n_musclestrial = 7# all muscles are outputted
    #n_reachconditionstrial = n_input-1
    TARGETOUT_icondition = np.random.randint(0, high=n_reachconditions, size=n_trials)# (n_trials,) array, elements 0,1,...n_reachconditions-1
    if n_trials>=n_reachconditions: TARGETOUT_icondition[0:n_reachconditions] = np.arange(0,n_reachconditions)# first 8 trials contain the first 8 reach conditions
    
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
        tstartcondition = interval1 + 1# timestep when tstartcondition starts, there are interval1 timesteps before tstartcondition starts
        tendcondition = tstartcondition + 22# input specifying the condition is on for 23 timesteps
        interval2 = interval2set[np.random.randint(interval2set.size)]# interval after the input specifying the condition and before the hold-cue turns off
        tstarthold = 1# timestep 1 = index 0
        tendhold = tendcondition + interval2# hold-cue is on from tstarthold through tendhold (inclusive)
        tstartoutput = tendhold + 1# timestep when output starts (inclusive)
        tstartoutput = tstartoutput + toffsetoutput
        tenddata = np.minimum(n_T, tstartoutput + n_Tdata - 1)# timestep when kinematic data ends (inclusive), if tendoutput isn't truncated then kinematic output is on for n_Tdata(51) timesteps
        n_Toutput = tenddata - tstartoutput + 1# output kinematic data for n_Tdatatrial timesteps on this trial
        #print(f'tstartoutput = {tstartoutput}, tendemg = {tendemg}')
        
        # all timesteps from 1,2,3,...n_T are translated to indices 0,1,2,...n_T-1
        IN[itrial,(tstarthold-1):tendhold,n_input-1] = 1# last input is hold cue
        IN[itrial,(tendhold-8):tendhold,n_input-1] = bumpoff# last input is hold cue
        #if INPUT_ONE_HOT: 
        #    IN[itrial,(tstartcondition-1):tendcondition,TARGETOUT_icondition[itrial]] = bump
        if INPUT_SINE_COSINE: 
            IN[itrial,(tstartcondition-1):tendcondition,0] = np.sin(reach_direction[TARGETOUT_icondition[itrial]]) * bump
            IN[itrial,(tstartcondition-1):tendcondition,1] = np.cos(reach_direction[TARGETOUT_icondition[itrial]]) * bump
            
        if tstartoutput<=n_T:
            istartdimoutput = 0# starting index for output
            if OUTPUTONEHOT: 
                tend = n_T
                TARGETOUT[itrial,(tstartoutput-1):tend,istartdimoutput+TARGETOUT_icondition[itrial]] = 1# one-hot output for each reach condition until end of trial
                istartdimoutput = istartdimoutput + 8# one-hot output for each reach condition
            if OUTPUTXYHANDPOSITION: 
                n_Toutput = np.minimum(n_Tdata,n_T-tstartoutput+1)# number of timesteps the output is on for
                tend = tstartoutput + n_Toutput - 1# output is on during timesteps [tstart:tend] inclusive
                TARGETOUT[itrial,(tstartoutput-1):tend,istartdimoutput:(istartdimoutput+1)] = xhandposition[TARGETOUT_icondition[itrial], 0:n_Toutput][:,None]
                TARGETOUT[itrial,(tstartoutput-1):tend,(istartdimoutput+1):(istartdimoutput+2)] = yhandposition[TARGETOUT_icondition[itrial], 0:n_Toutput][:,None]
                output_mask[itrial,tend:,istartdimoutput:(istartdimoutput+2)] = 0# the output of the RNN for all timesteps after the xyhandposition are not counted in the loss function
                istartdimoutput = istartdimoutput + 2
            if OUTPUTXYHANDVELOCITY: 
                n_Toutput = np.minimum(n_Tdata,n_T-tstartoutput+1)# number of timesteps the output is on for
                tend = tstartoutput + n_Toutput - 1# output is on during timesteps [tstart:tend] inclusive
                TARGETOUT[itrial,(tstartoutput-1):tend,istartdimoutput:(istartdimoutput+1)] = xhandspeed[TARGETOUT_icondition[itrial], 0:n_Toutput][:,None]
                TARGETOUT[itrial,(tstartoutput-1):tend,(istartdimoutput+1):(istartdimoutput+2)] = yhandspeed[TARGETOUT_icondition[itrial], 0:n_Toutput][:,None]
                output_mask[itrial,tend:,istartdimoutput:(istartdimoutput+2)] = 0# the output of the RNN for all timesteps after the xyhandspeed are not counted in the loss function
                istartdimoutput = istartdimoutput + 2
            if OUTPUTSHOULDERELBOWANGLES: 
                n_Toutput = np.minimum(n_Tdata,n_T-tstartoutput+1)# number of timesteps the output is on for
                tend = tstartoutput + n_Toutput - 1# output is on during timesteps [tstart:tend] inclusive
                TARGETOUT[itrial,(tstartoutput-1):tend,istartdimoutput:(istartdimoutput+1)] = shoulderangle[TARGETOUT_icondition[itrial], 0:n_Toutput][:,None]
                TARGETOUT[itrial,(tstartoutput-1):tend,(istartdimoutput+1):(istartdimoutput+2)] = elbowangle[TARGETOUT_icondition[itrial], 0:n_Toutput][:,None]
                output_mask[itrial,tend:,istartdimoutput:(istartdimoutput+2)] = 0# the output of the RNN for all timesteps after the arm angles are not counted in the loss function
                istartdimoutput = istartdimoutput + 2
        
        # store trial info
        istartcondition = np.hstack((istartcondition, tstartcondition-1))# index (not timestep) when input specifying the condition starts
        iendcondition = np.hstack((iendcondition, tendcondition-1))# index (not timestep) when input specifying the condition ends
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
    if TEACHER_STUDENT_TRAINING==1:
        output_mask = torch.ones(n_trials,n_T,n_output, dtype=torch.float32)# all timepoints are included in loss function
        # 1. load parameters from previously trained model to serve as a teacher during training
        dir_parameters_teacher = task_input_dict['dir_parameters_teacher']
        n_parameter_updates_loadteacher = task_input_dict['n_parameter_updates_loadteacher']
        model_teacher = torch.load(f'{dir_parameters_teacher}/model.pth')# torch.save(model, f'{figdir}/model.pth')# save entire model, not just model parameters
        p = n_parameter_updates_loadteacher# choose RNN parameters to load
        checkpoint = torch.load(dir_parameters_teacher + f'/model_parameter_update{p}.pth'); model_teacher.load_state_dict(checkpoint['model_state_dict']); 
        
        # 2. change loss function, the goal is for each unit in the student model to match the firing rate of each unit in the teacher model, this only works if the teacher and student models have the same number of units
        activity_noise = task_input_dict['activity_noise']
        model_input_forwardpass_teacher = {'input':IN, 'activity_noise':activity_noise}
        model_output_forwardpass_teacher = model_teacher(model_input_forwardpass_teacher)# model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
        activity_teacher = model_output_forwardpass_teacher['activity']# (n_trials, n_T, n_recurrent)
        TARGETOUT = torch.tensor(activity_teacher.detach().numpy(), dtype=torch.float32)# get rid of graph and don't backprop through teacher. probably can do this more efficiently
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
    n_T = 250# if interval1 = 60 and interval2 = 80 then tendemg is 229
    n_T = 300
    T = np.arange(0,n_T)# (n_T,)
    n_trials = 100
    #toffsetoutput = -23# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
    toffsetoutput = 0# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
    interval1set=np.array([25]); interval2set=np.array([105])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    interval1set=np.array([49]); interval2set=np.array([149])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    #interval1set=np.array([0]); interval2set=np.array([0])# use A=np.array([x]) not A=np.array(x) so A[0] is defined
    #interval1set=np.arange(0,50); interval2set=np.arange(0,150)
    
    INPUT_ONE_HOT = 0; INPUT_SINE_COSINE = 1# select one, and only one, of these options
    #assert np.sum(INPUT_ONE_HOT + INPUT_SINE_COSINE) == 1, "Error: one, and only one, input type must be selected"
    if INPUT_ONE_HOT: n_input = 9# 8 one-hot inputs encode the reach condition + 1 input for the hold-cue
    if INPUT_SINE_COSINE: n_input = 3# 2 inputs (sine(orientation) and cosine(orientation)) encode the reach condition + 1 input for the hold-cue
    
    OUTPUTONEHOT=0; OUTPUTXYHANDPOSITION=0; OUTPUTXYHANDVELOCITY=0; OUTPUTSHOULDERELBOWANGLES=1# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES
    figure_label = ''
    n_output = 0
    if OUTPUTONEHOT: n_output = n_output+ 8; figure_label = figure_label + 'OUTPUTONEHOT'# one-hot output for each reach condition
    if OUTPUTXYHANDPOSITION: n_output = n_output + 2; figure_label = figure_label + 'OUTPUTXYHANDPOSITION'
    if OUTPUTXYHANDVELOCITY: n_output = n_output + 2; figure_label = figure_label + 'OUTPUTXYHANDVELOCITY'
    if OUTPUTSHOULDERELBOWANGLES: n_output = n_output + 2; figure_label = figure_label + 'OUTPUTSHOULDERELBOWANGLES'# output shoulder and elbow angle
    
    #task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'INPUT_ONE_HOT':INPUT_ONE_HOT, 'INPUT_SINE_COSINE':INPUT_SINE_COSINE, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTSHOULDERELBOWANGLES':OUTPUTSHOULDERELBOWANGLES}
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTSHOULDERELBOWANGLES':OUTPUTSHOULDERELBOWANGLES}
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT_Hatsopoulos2007_sincosbumpinput_holdcue(task_input_dict)   
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
        #plt.savefig('%s/generateINandTARGETOUT_Hatsopoulos2007_trial%g_reachcondition%g_%s.pdf'%(figure_dir,itrial,TARGETOUT_icondition[itrial],figure_label), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.show()
        input("Press Enter to continue...")# pause the program until the user presses Enter
     



