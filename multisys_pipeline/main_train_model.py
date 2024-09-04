
import os
root_dir = './' #' os.getcwd() + '/' #os.path.dirname(__file__) + '/'# return the folder path for the file we're currently executing
print('root_dir = ' + root_dir)
os.chdir(root_dir)# print(f'current working direction is {os.getcwd()}')
import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib
if 'allen' in os.getcwd():
    matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

#import testpipeline
#import sys; sys.exit()# stop script at current line

#import testpipeline
from multisys_pipeline.models.model_architectures import CTRNN, effective_weight, get_Wab # relative import, from file import function
# from multisys_pipeline.models.model_architectures import LowPassCTRNN 
from multisys_pipeline.utils.compute_normalized_error import compute_normalized_error
#from multisys_pipeline.models.generateINandTARGETOUT_Sussillo2015_EMG import generateINandTARGETOUT_Sussillo2015_EMG
#---
# from multisys_pipeline.models.generateINandTARGETOUT_Sussillo2015_planinputextendedscaledup_holdcue import generateINandTARGETOUT_Sussillo2015_planinputextendedscaledup_holdcue
from torch.optim.lr_scheduler import StepLR
#---
#import sys; sys.exit()# stop script at current line


#%%###########################################################################
# Use ArgumentParser to load parameters from command line or from launch_jobs_cluster.py
# To run from the command line >> python3 main.py  -N 33 --string_variable 'testcommandline'
# A - or -- added to the argument name turns it into an optional argument, which can always be omitted at the command line. To make an option required, True can be specified for the required= keyword argument to add_argument(): parser.add_argument('--foo', required=True)
# The default= parameter allows you to specify a default value if the option is not specified
# https://tacc.github.io/ctls2017/docs/intro_to_python/intro_to_python_101_argparse.html
from argparse import ArgumentParser
parser = ArgumentParser()# if type is not specified then by default the parser reads command-line arguments in as simple strings
parser.add_argument('--task_name', default='Mante13', choices=['Mante13'], type=str)# task_name determines which version of generateINandTARGETOUT is used, task_name is also used to name the folder where parameters are stored
parser.add_argument('--optimizer_name', default='Adam', choices=['Adam', 'AdamW'], help="Options: 'Adam', 'AdamW'", type=str)# optimizer_name determines which optimizer to use when updating the model parameters
parser.add_argument('--learning_rate', default=1e-3, help='learning rate for Adam optimizer', type=float)# learning_rate = 1e-3 default
parser.add_argument('--CLIP_GRADIENT_NORM', default=0, help='if CLIP_GRADIENT_NORM = 1 then clip the norm of the gradient', type=int)
parser.add_argument('--max_gradient_norm', default=0.4, help='if CLIP_GRADIENT_NORM = 1 then the norm of the gradient is clipped to have a maximum value of max_gradient_norm', type=float)# this is only used if CLIP_GRADIENT_NORM = 1
parser.add_argument('--n_parameter_updates', default=3000, help='number of parameter updates', type=int)# set required=True for arguments that are required
parser.add_argument('--model_class', default='CTRNN', help='RNN model architecture', type=str)# 'CTRNN', 'LowPassCTRNN'
parser.add_argument('--activation_function', default='retanh', help='RNN activation function', type=str)# set required=True for arguments that are required
parser.add_argument('--n_recurrent', default=400, help='number of units in RNN', type=int)# set required=True for arguments that are required
parser.add_argument('--regularization_activityL2', default=0.0, help='L2 regularization on h - "firing rate" of units, larger regularization_activityL2 = more regularization = smaller absolute firing rates', type=float)# set required=True for arguments that are required
parser.add_argument('--L2_Wrec', default=0.0, help='L2 regularization on off block diagonal Wrec', type=float)
parser.add_argument('--activity_noise_std', default=0.1, help='added noise', type=float)# use 0.0 instead of 0 so folder names are consistent when using nonzero noise, e.g. activity_noise_std = 0.1
parser.add_argument('--ini_gain', default=1.0, help='initial weight gain', type=float)
parser.add_argument('--tau_m', default=10.0, help='membrane time constant', type=float)
parser.add_argument('--conn_density', default=-1, help='sparsity of connection, -1 to turn off', type=float)
parser.add_argument('--learning_mode', default=1, help='learning rule mode; 0: BPTT, 1:e-prop, 2:ModProp, -1:TBPTT', type=int)
parser.add_argument('--dale_constraint', default=False, help='constrain with Dale or not', type=bool)
parser.add_argument('--random_seed', default=1, help='random seed', type=int)# set required=True for arguments that are required
#-----------------------------------------------------------------------------
# the following arguments are used if task_name is 'Sussillo2015'
parser.add_argument('--toffsetoutput', default=10, help="toffsetoutput is only used if task_name is 'Hatsopoulos2007' or 'Sussillo2015'. Number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial", type=int)# set required=True for arguments that are required
parser.add_argument('--OUTPUTONEHOT', default=0, type=int)# set required=True for arguments that are required
parser.add_argument('--OUTPUTEMG', default=1, type=int)# set required=True for arguments that are required
parser.add_argument('--folder_comment', default='', help="sub folder to load store data", type=str)
#-----------------------------------------------------------------------------
args = vars(parser.parse_args())# args is a dictionary with keys args.keys()
# example: task_name = args["task_name"]
for key,val in args.items():# https://stackoverflow.com/questions/18090672/convert-dictionary-entries-into-variables
    exec(key + '=val')# if we have a dictionary with keys that are all strings, unpack these keys into local variables with the corresponding values. example: keyname = dictionary["keyname"]

#import sys; sys.exit()# stop script at current line

# comment out if don't want to fix seed 
set_seed = True
if set_seed:
    np.random.seed(args["random_seed"]); torch.manual_seed(args["random_seed"])# set random seed for reproducible results 

assert learning_mode < 3 or (activity_noise_std == 0.0), \
        "If learning_mode > 3, then recurrent noise is not supported"

##############################################################################
#%% initialize network
task_name = args["task_name"]# task_name determines which version of generateINandTARGETOUT is used, task_name is also used to name the folder where parameters are stored
if task_name == 'Sussillo2015_planinputextendedscaledup_holdcue':
    short_task_name = 'Sussillo2015'
else:
    short_task_name = task_name


if task_name[0:12]=='Sussillo2015':# this assumes task_name starts with the phrase Sussillo2015
    #-----
    if task_name=='Sussillo2015_planinputextendedscaledup_holdcue':
        generateINandTARGETOUT = generateINandTARGETOUT_Sussillo2015_planinputextendedscaledup_holdcue; n_input = 16# DON'T CHANGE, 15 numbers encode each reach condition + 1 input for the hold-cue
        interval1set=np.arange(0,50); interval2set=np.arange(0,150)
    #-----
    n_T = 229# if interval1 = 60 and interval2 = 80 then tendemg is 229
    n_T = 300# increase n_T so target outputs are not truncated when interval1 and interval2 are large
    n_trials = 100
    toffsetoutput = args["toffsetoutput"]# toffsetoutput = -6 for all outputs for Sussillo2015 dataset
    # OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1
    # in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    OUTPUTONEHOT = args["OUTPUTONEHOT"]# 0 or 1
    OUTPUTXYHANDPOSITION = 0# args["OUTPUTXYHANDPOSITION"]# 0 or 1
    OUTPUTXYHANDVELOCITY = 0# args["OUTPUTXYHANDVELOCITY"]# 0 or 1
    OUTPUTEMG = args["OUTPUTEMG"]# 0 or 1
    n_output = 0
    if OUTPUTONEHOT==1: n_output = n_output + 27# one-hot output for each reach condition
    if OUTPUTXYHANDPOSITION==1: n_output = n_output + 2
    if OUTPUTXYHANDVELOCITY==1: n_output = n_output + 2
    if OUTPUTEMG==1: n_output = n_output + 7# output 7 EMG
    
    folder_suffix = ''
    if OUTPUTONEHOT: folder_suffix = folder_suffix + '_outputonehot'
    if OUTPUTXYHANDPOSITION: folder_suffix = folder_suffix + '_outputxyhandposition'
    if OUTPUTXYHANDVELOCITY: folder_suffix = folder_suffix + '_outputxyhandvelocity'
    if OUTPUTEMG: folder_suffix = folder_suffix + '_outputEMG'
    folder_suffix = folder_suffix + f'_toffsetoutput{toffsetoutput}'
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}

elif task_name == 'Mante13':
    Mante13_path = root_dir + 'Mante13_data/' + folder_comment
    neuralforRNN = np.load(Mante13_path + 'neuralforRNN.npy') # generated from Katheryn's code 
    with open(Mante13_path + 'data_synthetic.pkl', 'rb') as f:
        data_synthetic = pickle.load(f)
        IN = data_synthetic['IN']
        TARGETOUT = data_synthetic['TARGETOUT']
        output_mask = data_synthetic['output_mask']
        batch_trial_info = data_synthetic['batch_trial_info']    
    n_trials, n_T, n_input = IN.shape
    n_output = 3 # went into Katheryn's code and printed this 
    folder_suffix = ''
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials}

   
n_recurrent = args["n_recurrent"]# (200)
#-------------------------------------------  
activation_function = args["activation_function"]# (retanh)
regularization_activityL2 = args["regularization_activityL2"]# (0 or 1 are reasonable) # L2 regularization on h - "firing rate" of units, larger regularization_activityL2 = more regularization = smaller absolute firing rates
n_parameter_updates = args["n_parameter_updates"]# 7000
# During training sample pset_saveparameters more densely during the first 1500 parameter updates, e.g. np.round(np.linspace(200,1500,num=50,endpoint=True)) because this is where most of the gains occur. This will help if I want to compare models at some performance cutoff and actually find models with a performance near this cutoff.
pset_saveparameters = np.unique(np.concatenate((np.arange(0,6), np.array([50, 100, 150, 200]), np.round(np.linspace(0,n_parameter_updates,num=20,endpoint=True)), np.round(np.linspace(200,np.minimum(700,n_parameter_updates),num=40,endpoint=True)), np.round(np.linspace(700,np.minimum(1500,n_parameter_updates),num=30,endpoint=True)) ))).astype(int)# save parameters when parameter update p is a member of pset_saveparameters, save as int so we can use elements to load model parameters: for example, model_parameter_update211.pth versus model_parameter_update211.0.pth   
pset_saveparameters = np.unique(np.concatenate((np.arange(0,6), np.array([50, 100, 150, 200]), np.round(np.linspace(25,150,num=20,endpoint=True)), np.round(np.linspace(0,n_parameter_updates,num=20,endpoint=True)), np.round(np.linspace(200,np.minimum(700,n_parameter_updates),num=40,endpoint=True)), np.round(np.linspace(700,np.minimum(1500,n_parameter_updates),num=30,endpoint=True)) ))).astype(int)# save parameters when parameter update p is a member of pset_saveparameters, save as int so we can use elements to load model parameters: for example, model_parameter_update211.pth versus model_parameter_update211.0.pth   
#------------------------------------------------------------------------------
pset_saveparameters = np.delete(pset_saveparameters, pset_saveparameters>n_parameter_updates)


# constants that are not learned: dt, Tau, activity_noise
activity_noise_std = args["activity_noise_std"]# (0.1) standard deviation of firing rate noise, activity_noise = activity_noise_std*torch.randn(n_trials, n_T, n_recurrent)
activity_noise = activity_noise_std*torch.randn(n_trials, n_T, n_recurrent)# (n_trials, n_T, n_recurrent) tensor

# parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional)
ah0 = torch.zeros(n_recurrent)
bah = torch.zeros(n_recurrent)
by = torch.zeros(n_output)


# Sussillo et al. 2015 "A neural network that finds a naturalistic solution for the production of muscle activity"
if set_seed: 
    np.random.seed(args["random_seed"]+2); torch.manual_seed(args["random_seed"]+2)# set random seed for reproducible results 
Wahx = torch.randn(n_recurrent,n_input) / np.sqrt(n_input)
Wahh = 1.5 * torch.randn(n_recurrent,n_recurrent) / np.sqrt(n_recurrent); initname = '_initWrecsussillo'# initname = '_initWahhsussillo'
Wyh = None # torch.zeros(n_output,n_recurrent)



LEARN_OUTPUTWEIGHT = True
LEARN_OUTPUTBIAS = True
#LEARN_OUTPUTWEIGHT = False; LEARN_OUTPUTBIAS = False; np.random.seed(args["random_seed"]+2); torch.manual_seed(args["random_seed"]+2); Wyh = torch.randn(n_output,n_recurrent) / np.sqrt(n_recurrent); by = torch.zeros(n_output)
if args["model_class"]=='CTRNN':
    tau_m_ = tau_m 
    # if False: 
    #     part1 = np.random.uniform(5, 15, int(n_recurrent/2))         # Half of the elements between 5 and 15
    #     part2 = np.random.uniform(1, 5, int(n_recurrent/4))       # 1/4 of the elements between 1 and 3
    #     part3 = np.random.uniform(15, 40, int(n_recurrent/4))     # 1/4 of the elements between 20 and 50
    #     tau_array = np.concatenate([part1, part2, part3])
    #     np.random.shuffle(tau_array)
    #     tau_m_ = tau_array 
    model = CTRNN(n_input, n_recurrent, n_output, activation_function=activation_function, ah0=ah0, LEARN_ah0=True, Wahx=Wahx, Wahh=Wahh, Wyh=Wyh, bah=bah, by=by, LEARN_OUTPUTWEIGHT=LEARN_OUTPUTWEIGHT, LEARN_OUTPUTBIAS=LEARN_OUTPUTBIAS, gain_Wh2h=ini_gain, learning_mode=learning_mode, dale_constraint=dale_constraint, Tau=tau_m_, conn_density=conn_density); model_class = 'CTRNN'; 
# if args["model_class"]=='LowPassCTRNN':
#     model = LowPassCTRNN(n_input, n_recurrent, n_output, Wrx=Wahx, Wrr=Wahh, Wyr=Wyh, br=bah, by=by, activation_function=activation_function, r0=ah0, LEARN_r0=True, LEARN_OUTPUTWEIGHT=LEARN_OUTPUTWEIGHT, LEARN_OUTPUTBIAS=LEARN_OUTPUTBIAS); model_class = 'LowPassCTRNN'
     


#---------------check number of learned parameters---------------
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)# model.parameters include those defined in __init__ even if they are not used in forward pass
assert np.allclose(model.n_parameters, n_parameters), "Number of learned parameters don't match!"
#import sys; sys.exit()# stop script at current line

#---------------make folder to store files---------------
#figure_dir = root_dir + f'store_trained_models/{task_name}_{model_class}_{activation_function}_nin{n_input}nout{n_output}nrecurrent{n_recurrent}_{n_parameters}parameters_{n_parameter_updates}parameterupdates{initname}_activitynoisestd{activity_noise_std}_rng{random_seed}{folder_suffix}'
#figure_dir = root_dir + f'store_trained_models/{task_name}_{model_class}_{activation_function}_nin{n_input}nout{n_output}nrecurrent{n_recurrent}_{n_parameter_updates}parameterupdates{initname}_regularizationactivityL2{regularization_activityL2}_activitynoisestd{activity_noise_std}_rng{random_seed}{folder_suffix}'

#figure_dir = root_dir + f'store_trained_models/{short_task_name}_{model_class}_{activation_function}_nin{n_input}nout{n_output}nrecurrent{n_recurrent}_activitynoisestd{activity_noise_std}_rng{random_seed}{folder_suffix}_g0{ini_gain}_lm{learning_mode}'
if 0 < conn_density < 1:    
    figure_dir = root_dir + f'store_trained_models/{folder_comment}{short_task_name}_{activation_function}_nrecurrent{n_recurrent}_activitynoisestd{activity_noise_std}_rng{random_seed}{folder_suffix}_lm{learning_mode}_lr{learning_rate}_l2{regularization_activityL2}_g0{ini_gain}_sp{conn_density}'
elif L2_Wrec > 0:
    figure_dir = root_dir + f'store_trained_models/{folder_comment}{short_task_name}_{activation_function}_nrecurrent{n_recurrent}_activitynoisestd{activity_noise_std}_rng{random_seed}{folder_suffix}_lm{learning_mode}_lr{learning_rate}_l2{regularization_activityL2}_g0{ini_gain}_l2W{L2_Wrec}'
else: 
    figure_dir = root_dir + f'store_trained_models/{folder_comment}{short_task_name}_{activation_function}_nrecurrent{n_recurrent}_activitynoisestd{activity_noise_std}_rng{random_seed}{folder_suffix}_lm{learning_mode}_lr{learning_rate}_l2{regularization_activityL2}_g0{ini_gain}'

if not os.path.exists(figure_dir):# if folder doesn't exist then make folder
    os.makedirs(figure_dir, exist_ok=True)

#---------------save model info---------------
# save info into key-value pairs of dictionary
model_info_dictionary = {}# empty dictionary
model_info_dictionary['task_name'] = task_name
model_info_dictionary['optimizer_name'] = optimizer_name
model_info_dictionary['n_input'] = n_input
model_info_dictionary['n_output'] = n_output
model_info_dictionary['n_recurrent'] = n_recurrent
model_info_dictionary['n_T'] = n_T
model_info_dictionary['n_trials'] = n_trials
model_info_dictionary['model_class'] = model_class
model_info_dictionary['activation_function'] = activation_function
model_info_dictionary['initname'] = initname
model_info_dictionary['n_parameter_updates'] = n_parameter_updates
model_info_dictionary['regularization_activityL2'] = regularization_activityL2# L2 regularization on h - "firing rate" of units, larger regularization_activityL2 = more regularization = smaller absolute firing rates
model_info_dictionary['activity_noise_std'] = activity_noise_std
model_info_dictionary['figure_dir'] = figure_dir
model_info_dictionary['random_seed'] = random_seed
model_info_dictionary['learning_rate'] = learning_rate
model_info_dictionary['CLIP_GRADIENT_NORM'] = CLIP_GRADIENT_NORM
model_info_dictionary['max_gradient_norm'] = max_gradient_norm
model_info_dictionary['ini_gain'] = ini_gain
# parameters that may or may not exist depending on the task
if task_name[0:12]=='Sussillo2015':# this assumes task_name starts with the phrase Sussillo2015
    model_info_dictionary['toffsetoutput'] = toffsetoutput# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
    model_info_dictionary['OUTPUTONEHOT'] = OUTPUTONEHOT# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    model_info_dictionary['OUTPUTXYHANDPOSITION'] = OUTPUTXYHANDPOSITION# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    model_info_dictionary['OUTPUTXYHANDVELOCITY'] = OUTPUTXYHANDVELOCITY# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    model_info_dictionary['OUTPUTEMG'] = OUTPUTEMG# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
   

np.save(f'{figure_dir}/model_info_dictionary.npy', model_info_dictionary)
#model_info_dictionary = np.load(f'{figure_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
#import sys; sys.exit()# stop script at current line

#---------------save pset_saveparameters---------------
np.save(f'{figure_dir}/pset_saveparameters.npy', pset_saveparameters)#pset_saveparameters = np.load(f'{figure_dir}/pset_saveparameters.npy')

#---------------save entire model, not just model parameters---------------
torch.save(model, f'{figure_dir}/model.pth')# model = torch.load(f'{figure_dir}/model.pth')# save entire model, not just model parameters
# This save/load process uses the most intuitive syntax and involves the least amount of code. 
# Saving a model in this way will save the entire module using Python’s pickle module. 
# The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

#---------------plot initial eigenvalue spectrum---------------
plt.figure()# initial eigenvalue spectrum of recurrent weight matrix  
if model_class=='CTRNN': W = model.fc_h2ah.weight.detach().numpy() 
# if model_class=='LowPassCTRNN': W = model.fc_r2r.weight.detach().numpy()
plt.clf()
eigVal = np.linalg.eigvals(W)
plt.plot(eigVal.real, eigVal.imag, 'k.', markersize=10)
plt.xlabel('real(eig(W))')
plt.ylabel('imag(eig(W))')
plt.title(model_class)
plt.axis('equal')# plt.axis('scaled') 
plt.savefig('%s/eigenvaluesW_beforelearning_%s.pdf'%(figure_dir,model_class.replace(" ", "")), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
print('Warning: plotted e-val not for effective weight!')
#import sys; sys.exit()# stop script at current line


# Recurrent weight regularization
# Punish off-block diagonal weights, i.e. long range connections across types 
def regularization_L2Wrec(W, K=2):
    """
    Calculate the regularization term for the weight matrix W.

    Parameters:
    - W: The recurrent weight tensor
    - K: int, the number of blocks (cell types).

    Returns:
    - reg_term: float, the calculated regularization term.
    """
    block_size = n_recurrent // K  # Size of each block
    reg_term = 0.0  # Initialize the regularization term

    # Iterate over blocks
    for i in range(K):
        for j in range(K):
            # Extract the block
            block = W[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if i == j:
                # Diagonal blocks (same cell type), no penalty
                continue
            else:
                # Off-diagonal blocks (different cell types), add penalty
                reg_term += L2_Wrec * torch.sum(block**2)  # L2 penalty for simplicity
    return reg_term

def exp_convolve(tensor, decay):
    # tensor shape: (trial, time, neuron)
    tensor_time_major = tensor.transpose(0, 1)  # Change to (time, trial, neuron)
    initializer = torch.zeros_like(tensor_time_major[0])

    # Initialize a list to hold the filtered tensor slices
    filtered_tensor_slices = [initializer]

    # Manually iterate through the time dimension
    for t in range(1, tensor_time_major.shape[0]):
        # Apply the exponential convolution formula
        a = filtered_tensor_slices[-1]  # Previous state
        x = tensor_time_major[t]  # Current input
        filtered_slice = a * decay + (1 - decay) * x
        filtered_tensor_slices.append(filtered_slice)

    # Stack the filtered slices along the time dimension and transpose back
    filtered_tensor = torch.stack(filtered_tensor_slices, dim=0).transpose(0, 1)

    return filtered_tensor


##############################################################################
#%% train network
if set_seed:
    np.random.seed(args["random_seed"]+3); torch.manual_seed(args["random_seed"]+3)# set random seed for reproducible results 
if args['optimizer_name']=='Adam': optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)# learning_rate = 1e-3 default
if args['optimizer_name']=='AdamW': optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)# learning_rate = 1e-3 default
if learning_mode > 3:
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8) 
error_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
errormain_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
error_activityL2_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
error_dimensionality_store = -700*np.ones(n_parameter_updates+1)# error_store[0] is the error before any parameter updates have been made, error_store[j] is the error after j parameter updates
gradient_norm = -700*np.ones(n_parameter_updates+1)# gradient_norm[0] is norm of the gradient before any parameter updates have been made, gradient_norm[j] is the norm of the gradient after j parameter updates
figure_suffix = ''
#pcounter = 0
for p in range(n_parameter_updates+1):# 0, 1, 2, ... n_parameter_updates
    activity_noise = activity_noise_std*torch.randn(n_trials, n_T, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
    if task_name[0:12]=='Sussillo2015':
        IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
    
    model_output_forwardpass = model({'input':IN, 'activity_noise':activity_noise, 'conn_density':conn_density})# model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
    output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
    if task_name[0:12]=='Sussillo2015':
        errormain = torch.sum((output[output_mask==1] - TARGETOUT[output_mask==1])**2) / torch.sum(output_mask==1)# output_mask: n_trials x n_T x n_output tensor, elements 0(timepoint does not contribute to this term in the error function), 1(timepoint contributes to this term in the error function) 
    elif task_name == 'Mante13':
        # Reshape or transpose output to match the expected shape for cross_entropy: (batch*time, num_classes)
        output_flat = output.reshape(-1, output.size(2))  # Now shape is [batch*time, neuron]        
        # Flatten TARGETOUT and output_mask to align with the reshaped output tensor
        TARGETOUT_flat = TARGETOUT.reshape(-1).long()  # Flatten TARGETOUT
        output_mask_flat = output_mask.reshape(-1)  # Flatten output_mask        
        # Select the entries that are marked by the output_mask
        selected_outputs = output_flat[output_mask_flat.nonzero().squeeze(), :]  # Outputs selected by the mask
        selected_targets = TARGETOUT_flat[output_mask_flat.nonzero().squeeze()].long()  # Corresponding targets        
        # Compute cross-entropy loss for the selected entries
        errormain = torch.nn.functional.cross_entropy(selected_outputs, selected_targets)
    error_activityL2 = regularization_activityL2*torch.mean(activity.flatten()**2) 
    
    error = errormain + error_activityL2 + regularization_L2Wrec(model.fc_h2ah.weight)
    
    if error_activityL2 >= errormain/2: regularization_activityL2 = regularization_activityL2*2/3# bound error_activityL2 from getting too large
    error_store[p] = error.item(); errormain_store[p] = errormain.item(); error_activityL2_store[p] = error_activityL2.item()# error.item() gets the scalar value held in error
    
    
    #if p<=3 or p==n_parameter_updates-1 or np.remainder(p,np.ceil(n_parameter_updates/20))==0:
    if np.isin(p,pset_saveparameters):# model_parameter_update{p}.pth stores the parameters after p parameter updates, model_parameter_update0.pth are the initial parameters
        print(f'{p} parameter updates: error = {error.item():.4g}')
        torch.save({'model_state_dict':model.state_dict(), 'figure_suffix':figure_suffix, 'regularization_activityL2':regularization_activityL2}, figure_dir + f'/model_parameter_update{p}.pth')# save the trained model’s learned parameters
    
    if p==10 or np.isin(p, np.linspace(0,n_parameter_updates,num=4,endpoint=True).astype(int)):# only plot and save 5 times during training
        plt.figure()# training error vs number of parameter updates
        plt.semilogy(np.arange(0,p+1), errormain_store[0:p+1], 'k-', linewidth=1, label=f'{model_class} main {errormain.item():.4g}')
        plt.semilogy(np.arange(0,p+1), error_activityL2_store[0:p+1], 'k--', linewidth=1, label=f'{model_class} hL2 {error_activityL2.item():.4g}')
        plt.xlabel('Number of parameter updates')
        plt.ylabel('Error during training')
        plt.legend()
        plt.title(f"{p} parameter updates, error = {error_store[p]:.4g}, errormain = {errormain_store[p]:.4g}\n"
                  f"error_activityL2 = {error_activityL2_store[p]:.4g}")
        plt.xlim(left=0)
        #plt.ylim(bottom=0) 
        plt.savefig('%s/error_trainingerrorVSparameterupdates%s_.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.pause(0.05)# https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    
    
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the Tensors it will update (which are the learnable weights of the model)
    optimizer.zero_grad()
    
    # Backward pass: compute gradient of the error with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    error.backward() # error.backward(retain_graph=True)
    
    # clip the norm of the gradient
    if args['CLIP_GRADIENT_NORM']:
        #max_gradient_norm = 1; figure_suffix = '_maxgradientnorm1'
        #max_gradient_norm = 0.4; figure_suffix = '_maxgradientnorm0.4'
        max_gradient_norm = args['max_gradient_norm']; figure_suffix = f'_maxgradientnorm{max_gradient_norm}'
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    
    # Calling the step function on an Optimizer makes an update to its parameters
    if p<=n_parameter_updates-1:# only update the parameters n_parameter_updates times. note that the parameters are first updated when p is 0
        optimizer.step()# parameters that have a gradient of zero may still be updated due to weight decay or momentum (if previous gradients were nonzero)
    if learning_mode > 3:
        scheduler.step()
 
    gradient = []# store all gradients
    for param in model.parameters():# model.parameters include those defined in __init__ even if they are not used in forward pass
        if param.requires_grad is True:# model.parameters include those defined in __init__ even if param.requires_grad is False (in this case param.grad is None)
            gradient.append(param.grad.detach().flatten().numpy())
    gradient = np.concatenate(gradient)# gradient = torch.cat(gradient)
    assert np.allclose(gradient.size,model.n_parameters), "size of gradient and number of learned parameters don't match!"
    gradient_norm[p] = np.sqrt(np.sum(gradient**2))
    #print(f'{p} parameter updates: gradient norm = {gradient_norm[p]:.4g}')
    
    # update Wab, assume ModProp run when both are true 
    if (activation_function == 'ReLU' and dale_constraint):
        model.Wab = get_Wab(effective_weight(model.fc_h2ah.weight, model.mask), model.tp_idx)

        
#import sys; sys.exit()# stop script at current line    
##############################################################################
#%%
# normalized error, if RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
# outputforerror = output(output_mask==1)
# TARGETOUTforerror = TARGETOUT(output_mask==1)
# errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
if task_name != 'Mante13':
    errornormalized = compute_normalized_error(TARGETOUT.detach().numpy(), output.detach().numpy(), output_mask.detach().numpy())# all inputs are arrays with shape (n_trials, n_T, n_output)

# plot training figures
plt.figure()# norm of the gradient vs number of parameter updates
plt.plot(np.arange(0,n_parameter_updates+1), gradient_norm, 'k-', label=model_class)
plt.xlabel('Number of parameter updates')
plt.ylabel('Gradient norm during training')
plt.legend()
if task_name == 'Mante13':
    plt.title(f"{model_class}\n{n_parameter_updates} parameter updates, error = {error_store[-1]:.4g} \nmax = {np.max(gradient_norm):.2g}, min = {np.min(gradient_norm):.2g}, median = {np.median(gradient_norm):.2g}")
else: 
    plt.title(f"{model_class}\n{n_parameter_updates} parameter updates, error = {error_store[-1]:.4g}, normalized error = {errornormalized:.4g}\nmax = {np.max(gradient_norm):.2g}, min = {np.min(gradient_norm):.2g}, median = {np.median(gradient_norm):.2g}")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.savefig('%s/gradient_norm%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')


plt.figure()# training error vs number of parameter updates
plt.plot(np.arange(0,n_parameter_updates+1), error_store, 'k-', linewidth=1, label=f'{model_class} {error_store[n_parameter_updates]:.4g}')
plt.xlabel('Number of parameter updates')
plt.ylabel('Mean squared error during training')
plt.legend()
if task_name == 'Mante13': 
    plt.title('%s\n%.4g parameter updates, error = %.4g \nerror i%g = %.4g, i%g = %.4g, i%g = %.4g'\
              %(model_class,n_parameter_updates,error_store[-1],5,error_store[5],round(n_parameter_updates/2),error_store[round(n_parameter_updates/2)],n_parameter_updates,error_store[n_parameter_updates]))
else: 
    plt.title('%s\n%.4g parameter updates, error = %.4g, normalized error = %.4g\nerror i%g = %.4g, i%g = %.4g, i%g = %.4g'\
              %(model_class,n_parameter_updates,error_store[-1],errornormalized,5,error_store[5],round(n_parameter_updates/2),error_store[round(n_parameter_updates/2)],n_parameter_updates,error_store[n_parameter_updates]))
plt.xlim(left=0)
plt.ylim(bottom=0)  
plt.savefig('%s/error_trainingerrorVSparameterupdates%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


plt.figure()# training error vs number of parameter updates, semilogy
plt.semilogy(np.arange(0,n_parameter_updates+1), error_store, 'k-', linewidth=1, label=f'{model_class} {error_store[n_parameter_updates]:.4g}')
plt.xlabel('Number of parameter updates')
plt.ylabel('Mean squared error during training')
plt.legend()
if task_name == 'Mante13':
    plt.title('%s\n%.4g parameter updates, error = %.4g, \nerror i%g = %.4g, i%g = %.4g, i%g = %.4g'\
              %(model_class,n_parameter_updates,error_store[-1],5,error_store[5],round(n_parameter_updates/2),error_store[round(n_parameter_updates/2)],n_parameter_updates,error_store[n_parameter_updates]))
else: 
    plt.title('%s\n%.4g parameter updates, error = %.4g, normalized error = %.4g\nerror i%g = %.4g, i%g = %.4g, i%g = %.4g'\
              %(model_class,n_parameter_updates,error_store[-1],errornormalized,5,error_store[5],round(n_parameter_updates/2),error_store[round(n_parameter_updates/2)],n_parameter_updates,error_store[n_parameter_updates]))
plt.xlim(left=0); 
plt.savefig('%s/error_trainingerrorVSparameterupdates_semilogy%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

  
plt.figure()# final eigenvalue spectrum of recurrent weight matrix 
if model_class=='CTRNN': W = model.fc_h2ah.weight.detach().numpy(); 
# if model_class=='LowPassCTRNN': W = model.fc_r2r.weight.detach().numpy(); 
plt.clf()
eigVal = np.linalg.eigvals(W)
plt.plot(eigVal.real, eigVal.imag, 'k.', markersize=10)
plt.xlabel('real(eig(W))')
plt.ylabel('imag(eig(W))')
if task_name == 'Mante13': 
    plt.title(f"{model_class}\n{n_parameter_updates} parameter updates, error = {error_store[-1]:.4g}")
else:
    plt.title(f"{model_class}\n{n_parameter_updates} parameter updates, error = {error_store[-1]:.4g}, normalized error = {errornormalized:.4g}")
plt.axis('equal')# plt.axis('scaled')
plt.savefig('%s/eigenvaluesW_%gparameterupdates_%s%s.pdf'%(figure_dir,n_parameter_updates,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

if task_name == 'Mante13':
    np.save(f'{figure_dir}/error_store.npy', error_store[pset_saveparameters])
    pset = pset_saveparameters[pset_saveparameters>=0]
    np.save(f'{figure_dir}/pset.npy', pset)

#%%############################################################################
#                            Test data
##############################################################################

if task_name != 'Mante13':

    if set_seed:
        np.random.seed(random_seed); torch.manual_seed(random_seed)# set random seed for reproducible results
    n_trials_test = 1000
    n_T_test = n_T
    activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
    task_input_dict['n_trials'] = n_trials_test# method2: task_input_dict.update({'n_trials':n_trials_test})
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
    #n_T_test = task_output_dict['n_T']
    #n_trials_test = task_output_dict['n_trials']
    TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy()
    # IN:        (n_trials_test, n_T_test, n_input) tensor
    # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
    # activity:  (n_trials_test, n_T_test, n_recurrent) tensor
    
    
    ##############################################################################
    #%% percent normalized test error as a function of training iteration
    pset = pset_saveparameters[pset_saveparameters>=0]
    #pset = pset_saveparameters[pset_saveparameters>=0]
    errornormalized_store = -700*np.ones((pset.shape[0]))
    for ip, p in enumerate(pset):
        # load models, first re-create the model structure and then load the state dictionary into it
        checkpoint = torch.load(figure_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
        
        model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise, 'conn_density': conn_density}
        model_output_forwardpass = model(model_input_forwardpass)
        output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
        output = output.detach().numpy(); activity = activity.detach().numpy()
        # output:   (n_trials_test, n_T_test, n_output) tensor
        # activity: (n_trials_test, n_T_test, n_recurrent) tensor
    
        # normalized error, if RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
        # outputforerror = output(output_mask==1)
        # TARGETOUTforerror = TARGETOUT(output_mask==1)
        # errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
        errornormalized = compute_normalized_error(TARGETOUT, output, output_mask)# all inputs are arrays with shape (n_trials, n_T, n_output)
        errornormalized_store[ip] = errornormalized 
    np.save(f'{figure_dir}/pset.npy', pset)# pset = np.load(f'{figure_dir}/pset.npy')
    np.save(f'{figure_dir}/errornormalized_store.npy', errornormalized_store)# errornormalized_store = np.load(f'{figure_dir}/errornormalized_store.npy')
    
    fig, ax = plt.subplots()# normalized error versus number of parameter updates
    fontsize = 14
    handle = ax.plot(pset, errornormalized_store, 'k-', linewidth=3)  
    ax.legend(handles=handle, labels=[f'{model_class} {errornormalized_store[-1]:.6g}'], loc='best', frameon=True)
    ax.set_xlabel('Number of parameter updates', fontsize=fontsize)
    ax.set_ylabel('Normalized error', fontsize=fontsize)
    imin = np.argmin(errornormalized_store)# index of minimum normalized error
    if imin==(pset.size-1):# if the minimum normalized error occurs after the last parameter update only put the error after parameter updates pset[0] and pset[-1] in the title, remember pset[pset.size-1] gives the last element of pset 
        ax.set_title(f'{n_trials_test} test trials, {n_T_test} timesteps, {n_parameter_updates} parameter updates\nError after {pset[0]} parameter updates = {errornormalized_store[0]:.6g}\nError after {pset[imin]} parameter updates = {errornormalized_store[imin]:.6g}', fontsize=fontsize)
    else:
        ax.set_title(f'{n_trials_test} test trials, {n_T_test} timesteps, {n_parameter_updates} parameter updates\nError after {pset[0]} parameter updates = {errornormalized_store[0]:.6g}\nError after {pset[imin]} parameter updates = {errornormalized_store[imin]:.6g}\nError after {pset[-1]} parameter updates = {errornormalized_store[-1]:.6g}', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    fig.savefig('%s/errornormalized_test_nTtest%g%s.pdf'%(figure_dir,n_T_test,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
      
      
    ##############################################################################
    #%% load model with lowest test error
    # load models, first re-create the model structure and then load the state dictionary into it
    n_parameter_updates_model = n_parameter_updates
    n_parameter_updates_model = pset[np.argmin(errornormalized_store)]
    checkpoint = torch.load(figure_dir + f'/model_parameter_update{n_parameter_updates_model}.pth'); model.load_state_dict(checkpoint['model_state_dict']); figure_suffix = checkpoint['figure_suffix']
    
    ##############################################################################
    #%% compute loss on test set
    model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise, 'conn_density': conn_density}
    model_output_forwardpass = model(model_input_forwardpass)
    output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
    output = output.detach().numpy(); activity = activity.detach().numpy()
    # output:   (n_trials_test, n_T_test, n_output) tensor
    # activity: (n_trials_test, n_T_test, n_recurrent) tensor
    #loss = loss_fn(output[output_mask==1], TARGETOUT[output_mask==1])
    
    
    # normalized error, if RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
    # outputforerror = output(output_mask==1)
    # TARGETOUTforerror = TARGETOUT(output_mask==1)
    # errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
    errornormalized = compute_normalized_error(TARGETOUT, output, output_mask)# all inputs are arrays with shape (n_trials, n_T, n_output)
    
    
    fontsize = 14
    T = np.arange(0,n_T_test)# (n_T_test,)
    
    plt.figure()# RNN input and output on test trials
    #----colormaps----
    cool = cm.get_cmap('cool', n_input)
    colormap_input = cool(range(n_input))# (n_input, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
    #-----------------
    n_curves = n_output# number of curves to plot
    blacks = cm.get_cmap('Greys', n_curves+3) 
    colormap = blacks(range(n_curves+3));# (n_curves+3, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha
    # colormap[0,:] = white, first row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[0,:], linewidth=3)
    # colormap[-1,:] = black, last row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[-1,:], linewidth=3)
    colormap = colormap[3:,:]# (n_curves, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha
    colormap_outputtarget = colormap
    #-----------------
    n_curves = n_output# number of curves to plot
    reds = cm.get_cmap('Reds', n_curves+3) 
    colormap = reds(range(n_curves+3));# (n_curves+3, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha
    # colormap[0,:] = almost white, first row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[0,:], linewidth=3)
    # colormap[-1,:] = dark red, last row, check: plt.figure(); plt.plot(np.arange(0,10), -13*np.ones(10), c=colormap[-1,:], linewidth=3)
    colormap = colormap[2:,:]# (n_curves+1, 4) array, remove first two rows because they are too light
    colormap = colormap[:-1,:]# (n_curves, 4) array, remove last row because it is too similar to black
    colormap_outputrnn = colormap
    if n_curves==1: colormap_outputrnn = np.array([1, 0, 0, 1])[None,:]# (1, 4) array, red
    #-----------------
    #for itrial in range(n_trials_test):
    for itrial in range(5):  
        plt.clf()
        #----plot single input and output for legend----
        plt.plot(T, IN[itrial,:,0], c=colormap_input[0,:], linewidth=3, label='Input'); 
        plt.plot(T[output_mask[itrial,:,0]==1], TARGETOUT[itrial,output_mask[itrial,:,0]==1,0], '-', c=colormap_outputtarget[-1,:], linewidth=3, label='Output: target'); 
        plt.plot(T, output[itrial,:,0], '--', c=colormap_outputrnn[-1,:], linewidth=3, label='Output: RNN')# for legend
        #----plot all inputs and outputs----
        for i in range(n_input):
            plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3)
        for i in range(n_output):
            plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_outputtarget[i,:], linewidth=3)# black
            plt.plot(T, output[itrial,:,i], '--', c=colormap_outputrnn[i,:], linewidth=3)# red
        plt.xlabel('Timestep', fontsize=fontsize)
        plt.legend()
        plt.title(f'{model_class}, trial {itrial}\n{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
        plt.xlim(left=0)
        #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
        plt.savefig('%s/testtrial%g_nTtest%g_%gparameterupdates_%s%s.pdf'%(figure_dir,itrial,n_T_test,n_parameter_updates_model,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    
        
    plt.figure()# firing of all hidden units on a single trial
    #for itrial in range(n_trials_test):
    for itrial in range(5): 
        plt.clf()
        plt.plot(T, activity[itrial,:,:])
        plt.xlabel('Timestep', fontsize=fontsize)
        plt.ylabel('Firing of hidden units', fontsize=fontsize)
        #plt.legend()
        plt.title(f'{model_class}, trial {itrial}\n{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
        plt.xlim(left=0)
        #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
        plt.savefig('%s/testtrial%g_nTtest%g_%gparameterupdates_h_%s%s.pdf'%(figure_dir,itrial,n_T_test,n_parameter_updates_model,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    
        
    plt.figure()# firing of a single hidden unit across all trials
    #for iunit in range(n_recurrent):
    for iunit in range(10): 
        plt.clf()
        plt.plot(T, activity[:,:,iunit].T)
        plt.xlabel('Timestep', fontsize=fontsize)
        plt.ylabel(f'Firing rate of unit {iunit}', fontsize=fontsize)
        #plt.legend()
        plt.title(f'{model_class}, unit {iunit}\n{n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
        plt.xlim(left=0)
        #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
        plt.savefig('%s/unit%g_nTtest%g_%gparameterupdates_h_%s%s.pdf'%(figure_dir,iunit,n_T_test,n_parameter_updates_model,model_class.replace(" ", ""),figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    




