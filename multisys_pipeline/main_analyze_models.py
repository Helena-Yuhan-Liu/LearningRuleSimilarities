import os
root_dir = './' #' os.getcwd() + '/' #root_dir = os.path.dirname(__file__) + '/'# return the folder path for the file we're currently executing
#----
os.chdir(root_dir)# print(f'current working direction is {os.getcwd()}')
import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm# for heatmap
import matplotlib.colors as mcolors
import scipy 
import re 
import shutil
import pickle 
#import colorcet as cc# perceptually uniform colour maps https://colorcet.holoviz.org/user_guide/, redwhiteblue = cm.get_cmap('cet_CET_D1', n_curves)  or cmap=cc.cm.CET_D1 


random_seed = 1234; np.random.seed(random_seed); torch.manual_seed(random_seed)# set random seed for reproducible results 

if 'allen' in os.getcwd():
    matplotlib.use('Agg') 

# for heatmap plots
#from matplotlib import colors
#import colorcet as cc# perceptually uniform colour maps https://colorcet.holoviz.org/user_guide/, redwhiteblue = cm.get_cmap('cet_CET_D1', n_curves)  or cmap=cc.cm.CET_D1 
from mpl_toolkits.axes_grid1 import make_axes_locatable



# from multisys_pipeline.models.generateINandTARGETOUT_Sussillo2015 import generateINandTARGETOUT_Sussillo2015


from multisys_pipeline.utils.computedistance_normalizedwithdata import computedistance_normalizedwithdata
from multisys_pipeline.utils.computedistance import computedistance_crossconditionvalidation
from multisys_pipeline.utils.computedistance import computedistance
from multisys_pipeline.utils.compute_normalized_error import compute_normalized_error



from sklearn.manifold import MDS, TSNE, Isomap
import umap
import openTSNE# from openTSNE import TSNE

from multisys_pipeline.utils.computecrossconditionaveragebaseline import computecrossconditionaveragebaseline# from file import function
from multisys_pipeline.utils.computecrossconditionaveragebaseline import computecrossconditionaveragebaseline_crossconditionvalidation# from file import function
from multisys_pipeline.utils.computePCA import computePCA# from file import function
from multisys_pipeline.utils.computeDPCA import computeDPCA# from file import function

#%%############################################################################
#                       Some important settings!!!!  
###############################################################################

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', default='Mante13', type=str, help='Mante13')
parser.add_argument('--anal_mode', default='compLM', type=str, help='compLR = compare different learning rates')
parser.add_argument('--load_stored_dist', default=False, type=bool, help='load stored distance if it exists')
args = parser.parse_args()

comment = '' 

LOAD_STORED_DIST = args.load_stored_dist # if stored dist exists for the given folder name, simply load it

VERBOSE = False 
plotPCA = False 

anal_mode = args.anal_mode  
dataset_name = args.dataset_name 


#%%############################################################################
#                       pick models to analyze
##############################################################################

n_models = 4
data_dir0 = root_dir + 'store_trained_models/Mante13_retanh_nrecurrent400_activitynoisestd0.1_rng1_lm0_lr0.001_l20.0_g01.0'
legend_label0 = 'BPTT'; color0='k'
data_dir1 = root_dir + 'store_trained_models/Mante13_retanh_nrecurrent400_activitynoisestd0.1_rng2_lm0_lr0.001_l20.0_g01.0'
legend_label1 = 'BPTT'; color1='k'
data_dir2 = root_dir + 'store_trained_models/Mante13_retanh_nrecurrent400_activitynoisestd0.1_rng1_lm1_lr0.001_l20.0_g01.0'
legend_label2 = 'magenta'; color2='m'
data_dir3 = root_dir + 'store_trained_models/Mante13_retanh_nrecurrent400_activitynoisestd0.1_rng2_lm1_lr0.001_l20.0_g01.0'
legend_label3 = 'magenta'; color3='m'
folder_suffix = '' # can be anything you want, for the purpose of naming stored analysis folders  


#%%###########################################################################
# loop through folders and check if a parameter is missing (may need to relaunch jobs on the cluster!)
for i in range(n_models):
    data_dir = eval(f"data_dir{i}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    if not os.path.exists(data_dir):# if folder doesn't exist then print the name of the folder
        print(f'data_dir does not exist: {data_dir}\n')
    else:
        pset_saveparameters = np.load(f'{data_dir}/pset_saveparameters.npy')
        for j in range(pset_saveparameters.size):
            p = pset_saveparameters[j]
            if (not os.path.exists(f'{data_dir}/model_parameter_update{p}.pth')):
                print(f'model_parameter_update{p}.pth does not exist in {data_dir}\n')# if file doesn't exist then print the name of the file
#import sys; sys.exit()# stop script at current line
 
   
     
# store info for the models in a list
xtick_label = (n_models+1)*[None]# initialize list with n_models elements
xtick_color = (n_models+1)*[None]# initialize list with n_models elements
for imodel in range(n_models):   
    data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    #xtick_label[imodel] = data_dir[len(root_dir):]# remove common folder prefix from string
    xtick_label[imodel] = data_dir.rsplit('/', 1)[1]# the label is the string after the last /
    xtick_color[imodel] = eval(f"color{imodel}")
xtick_color[-1] = 'orange' # data
xtick_label[-1] = 'neural data'
exec(f"color{n_models}=xtick_color[-1]")
exec(f"legend_label{n_models}=xtick_label[-1]")
 
# save info into key-value pairs of dictionary
distance_info_dictionary = {}# empty dictionary
distance_info_dictionary['n_models'] = n_models
#distance_info_dictionary['n_output'] = n_output
for i in range(n_models):
    data_dir = eval(f"data_dir{i}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
    #exec(f"n_recurrent{i} = model_info_dictionary['n_recurrent']")# https://stackoverflow.com/questions/6181935/how-do-you-create-different-variable-names-while-in-a-loop
    #distance_info_dictionary[f'n_recurrent{i}'] = eval(f'n_recurrent{i}')# use eval[f'stuff{0}'] get the numerical value of stuff0 and not the string 'stuff0'
    model_class_forforwardpass = model_info_dictionary['model_class']
    n_recurrent = model_info_dictionary['n_recurrent']
    distance_info_dictionary[f'data_dir{i}'] = eval(f'data_dir{i}')# use eval[f'stuff{0}'] get the numerical value of stuff0 and not the string 'stuff0'
    distance_info_dictionary[f'model_class_forforwardpass{i}'] = model_class_forforwardpass# use eval[f'stuff{0}'] get the numerical value of stuff0 and not the string 'stuff0'
    distance_info_dictionary[f'color{i}'] = eval(f'color{i}')# use eval[f'stuff{0}'] get the numerical value of stuff0 and not the string 'stuff0'
    distance_info_dictionary[f'legend_label{i}'] = eval(f'legend_label{i}')# use eval[f'stuff{0}'] get the numerical value of stuff0 and not the string 'stuff0'
#np.save(f'{figure_dir}/distance_info_dictionary.npy', distance_info_dictionary)
#distance_info_dictionary = np.load(f'{figure_dir}/distance_info_dictionary.npy', allow_pickle='TRUE').item()
  


#%%###########################################################################
#                    compute model/data similarities
# convert all numbers to distances so CCA distance = 1 - CCA similarity, varies between 0(most similar) and 1
# convert all numbers to distances so CKA distance = 1 - CKA similarity, varies between 0(most similar) and 1
# convert all numbers to distances, metricCKA distance varies between 0(most similar) and pi/2
# convert all numbers to distances, Williams2021 distance varies between 0(most similar) and pi/2
##############################################################################
if comment != '_DSA': 
    similarityname = 'Williams2021' # 'CCA', 'CKA', 'metricCKA', 'Williams2021'
else:
    similarityname = 'DSA'
NORMALIZE_BETWEEN0AND1 = 1# If NORMALIZE_BETWEEN0AND1 = 1 scale distance to be between 0(most similar) and 1
CROSS_CONDITION_VALIDATION = 0# If CROSS_CONDITION_VALIDATION = 1 compute train distance across n_conditions-1 conditions and test distance on the heldout condition. Repeat until all n_conditions have been in the test set. This is slower than computing the distance once across all n_conditions conditions (CROSS_CONDITION_VALIDATION = 0).
# 1) load model parameters at each point during training
# 2) compute firing rates 
# 3) compute model/data similarity at each point during training
VARIANCEEXPLAINED_KEEP = None# 0.9
n_PCs_KEEP = None
#n_PCs_KEEP = 20
if similarityname=='CCA': n_PCs_KEEP = 20
distance_info_dictionary['similarityname'] = similarityname
distance_info_dictionary['VARIANCEEXPLAINED_KEEP'] = VARIANCEEXPLAINED_KEEP
distance_info_dictionary['n_PCs_KEEP'] = n_PCs_KEEP



if dataset_name == 'Sussillo2015':# load neural data and pick timepoints to analyze in neural data and RNN models
    generateINandTARGETOUT = generateINandTARGETOUT_Sussillo2015
    #----
    # load neural and EMG data
    import scipy.io as sio# for loading .mat files
    data_dir_neuralforRNN = root_dir + 'experimental_data/Sussillo2015/'
    parameters = sio.loadmat(data_dir_neuralforRNN + 'EMGforRNN_Sussillo2015')
    print(parameters.keys())
    EMG = parameters['EMGforRNN_Sussillo2015']# n_muscles(7) x n_TEMG(66) x n_reachconditions(27) array, this is the nonzero part of the EMG, time [-250:10:400] ms relative to movement onset
    parameters = sio.loadmat(data_dir_neuralforRNN + 'neuralforRNN_Sussillo2015')
    print(parameters.keys())
    neuralforRNN = parameters['neuralforRNN_Sussillo2015']# n_neurons(180) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
    [n_neurons, n_Tneural, n_conditions] = neuralforRNN.shape
    #---------------
    n_conditions = 27
    Elabelneural = 'movement onset'# itimekeep_neural are relative to Elabelneural (ms)
    Tneural = np.arange(-1550,400+10,10)# time [-1550:10:400] ms relative to movement onset, 196 elements
    itimekeep_neural = np.logical_and(-400<=Tneural, Tneural<=400)# (n_Tneural,) array of True/False, timesteps to analyze for CCA, -400 to 400 ms relative to movement onset, interval used in Sussillo et al. 2015 "A neural network that finds a naturalistic solution for the production of muscle activity"
    itimekeep_neural = np.logical_and(-1450<=Tneural, Tneural<=400)# (n_Tneural,) array of True/False, timesteps to analyze for CCA, -1550 to 400 ms relative to movement onset
    #itimekeep_neural = np.logical_and(-1550<=Tneural, Tneural<=400)# (n_Tneural,) array of True/False, timesteps to analyze for CCA, -1550 to 400 ms relative to movement onset
    n_Tkeep = np.sum(itimekeep_neural)# number of timesteps for PCA analysis, number of True values in itimePCA
    distance_info_dictionary['itimekeep_neural'] = itimekeep_neural
    distance_info_dictionary['n_Tkeep'] = n_Tkeep
    #---------------
    ElabelRNN = 'movement onset'# itimekeep_RNN are relative to ElabelRNN (ms)
    itimekeep_RNN = itimekeep_neural
    n_T_test = 196# if interval1 = 45 and interval2 = 62 then tendemg is 196, there are 196 timesteps in the neural data from times [-1550:10:400] ms
    n_trials_test = 27# there are 27 different reach conditions
    
    interval1set = np.array([37])# use A=np.array([x]) not A=np.array(x) so A[0] is defined. interval before input specifying the reach condition, if interval1 is 0 then tstartcondition is at the very beginning of the trial (index 0)
    interval2set = np.array([69])# use A=np.array([x]) not A=np.array(x) so A[0] is defined. interval after the input specifying the reach condition and before the hold-cue turns off
    
    #---------------
    # create red-to-green colormap
    n_colors = n_conditions
    colorsredgreen = np.stack((np.linspace(0.9,0,n_colors), np.linspace(0,0.9,n_colors), np.zeros(n_colors), np.ones(n_colors)), axis=-1)# (n_colors, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]      
    ineuron = 27# sort colors based on mean activity of ineuron 27, largest mean activity is colored red
    Tkeep_ = np.arange(-870,-240+10,10)
    ikeep_ = np.isin(Tneural,Tkeep_)# (n_Tneural,) boolean indices of timepoints to keep
    meandata = np.mean(neuralforRNN[ineuron,ikeep_,:],0)# 27 x 1 matrix
    isortdata = np.flip(np.argsort(meandata))# largets element first, isortdata(1) = red, isortdata(27) = green
    isortcolormap = np.argsort(isortdata)
    colorsredgreen = colorsredgreen[isortcolormap,:]# (27, 4) matrix, sort colormap
    colormapforconditions = colorsredgreen
    #import sys; sys.exit()# stop script at current line
    
elif dataset_name == 'Mante13':
    Mante13_path = root_dir + 'Mante13_data/' 
    neuralforRNN = np.load(Mante13_path + 'neuralforRNN.npy') # generated from Katheryn's code 
    with open(Mante13_path + 'data_synthetic.pkl', 'rb') as f:
        data_synthetic = pickle.load(f)
        IN = data_synthetic['IN']
        TARGETOUT = data_synthetic['TARGETOUT']
        output_mask = data_synthetic['output_mask']
        batch_trial_info = data_synthetic['batch_trial_info']   
        recording_start_index = data_synthetic['recording_start_index']
        recording_stop_index = data_synthetic['recording_stop_index']
    n_trials, n_T, n_input = IN.shape
    n_trials_test, n_T_test = n_trials, n_T
    n_output = 3 # went into Katheryn's code and printed this 
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials}
    
    [n_neurons, n_Tneural, n_conditions] = neuralforRNN.shape
    n_colors = n_conditions
    colorsredgreen = np.stack((np.linspace(0.9,0,n_colors), np.linspace(0,0.9,n_colors), np.zeros(n_colors), np.ones(n_colors)), axis=-1)# (n_colors, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]      
    colormapforconditions = colorsredgreen
    Tneural = np.arange(n_Tneural)
    itimekeep_neural = np.ones_like(np.arange(n_Tneural)).astype(bool)
    itimekeep_RNN = np.zeros_like(np.arange(n_T))
    itimekeep_RNN[recording_start_index:recording_stop_index] = 1
    itimekeep_RNN = itimekeep_RNN.astype(bool)
    n_Tkeep = np.sum(itimekeep_neural)
    distance_info_dictionary['itimekeep_neural'] = itimekeep_neural
    distance_info_dictionary['n_Tkeep'] = n_Tkeep

assert np.sum(itimekeep_neural) == np.sum(itimekeep_RNN), "Error: the same number of timesteps must be compared in the neural data and RNN model"



#%%############################################################################
# if folder storing computed distances exists then just load these precomputed quantitites
# else make folder to store files  
#---------------make folder to store files---------------
import os
similarityinfo = similarityname
if (n_PCs_KEEP is not None):
    similarityinfo = similarityinfo + f'_nPCs{n_PCs_KEEP}'
if (VARIANCEEXPLAINED_KEEP is not None):
    similarityinfo = similarityinfo + f'_VARIANCEEXPLAINED{VARIANCEEXPLAINED_KEEP}'
figure_dir = root_dir + f'store_completed_analyses/{dataset_name}_nTkeep{n_Tkeep}_{n_models}models_{similarityinfo}_crossconditionvalidation{CROSS_CONDITION_VALIDATION}{folder_suffix}'
if not os.path.exists(figure_dir):# if folder storing computed distances doesn't exist, then make it
        os.makedirs(figure_dir)




#%%############################################################################
#         plot normalized error vs number of parameter updates
#              load data from the RNN training folders
###############################################################################
PLOTLEGEND = 1; figure_suffix ='_legend'
# for iteration in range(2):
#     if iteration==0: PLOTLEGEND = 0; figure_suffix = ''
#     if iteration==1: PLOTLEGEND = 1; figure_suffix ='_legend'

fig, ax = plt.subplots()# normalized error vs number of parameter updates
fontsize = 13
p80_list = []; ip80_list = []; p60_list = []; ip60_list = []; p40_list = []; ip40_list = []
for imodel in range(n_models):
    data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    pset = np.load(f'{data_dir}/pset.npy')
    if os.path.exists(f'{data_dir}/normalizederror_store.npy'): errornormalized_store = np.load(f'{data_dir}/normalizederror_store.npy')
    if os.path.exists(f'{data_dir}/errornormalized_store.npy'): errornormalized_store = np.load(f'{data_dir}/errornormalized_store.npy')
    if os.path.exists(f'{data_dir}/error_store.npy'): error_store = np.load(f'{data_dir}/error_store.npy')
    if dataset_name == 'Sussillo2015':
        ip80 = np.abs(errornormalized_store - (1-0.8)).argmin(); p80 = pset[ip80]
        p80_list.append(p80); ip80_list.append(ip80)
        ip60 = np.abs(errornormalized_store - (1-0.6)).argmin(); p60 = pset[ip60]
        p60_list.append(p60); ip60_list.append(ip60)
        ip40 = np.abs(errornormalized_store - (1-0.4)).argmin(); p40 = pset[ip40]
        p40_list.append(p40); ip40_list.append(ip40)
    if dataset_name == 'Mante13':
        ax.plot(pset, error_store, '-', color=eval(f"color{imodel}"), linewidth=1, label=eval(f"legend_label{imodel}")) 
    else:
        ax.plot(pset, errornormalized_store, '-', color=eval(f"color{imodel}"), linewidth=1, label=eval(f"legend_label{imodel}"))     
if PLOTLEGEND: ax.legend(frameon=True, loc='best')   
ax.set_xlabel('Number of parameter updates', fontsize=fontsize)
if dataset_name == 'Mante13':
    ax.set_ylabel('Error', fontsize=fontsize)
    # ax.set_title('Error loaded from training folders', fontsize=fontsize)
else:
    ax.set_ylabel('Normalized error', fontsize=fontsize)
    # ax.set_title('Normalized error loaded from training folders', fontsize=fontsize)
ax.set_ylim([0.0, 1.0])
ax.set_xlim(xmin=0, xmax=None); ax.set_ylim(ymin=0, ymax=None)
ax.tick_params(axis='both', labelsize=fontsize)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
fig.savefig('%s/errornormalized_vs_numberofparameterupdates_loadfromtrainingfolders%s.pdf'%(figure_dir,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
#import sys; sys.exit()# stop script at current line


#%%############################################################################
#     average neural activity across all neurons (for each reach condition)
###############################################################################
average_neuralforRNN = np.mean(neuralforRNN[:,itimekeep_neural,:].copy(),0)# (itimekeep_neural.size, n_conditions) array
meanovertime = np.mean(average_neuralforRNN.copy(),1)# (itimekeep_neural.size,) array
average_neuralforRNN_minusmeanovertime = average_neuralforRNN.copy() - meanovertime[:,np.newaxis] @ np.ones((1,n_conditions))# (itimekeep_neural.size, n_conditions) array

fig, ax = plt.subplots()# average neural activity across all neurons (for each reach condition)
fontsize = 12
for icondition in range(n_conditions):
    ax.plot(Tneural[itimekeep_neural], average_neuralforRNN[:,icondition], '-', linewidth=3, c = colormapforconditions[icondition,:])
#ax.plot(Tneural[itimekeep_neural], meanovertime, 'k-', linewidth=2)
if dataset_name == 'Mante13':
    ax.set_xlabel('Time points', fontsize=fontsize)
else:
    ax.set_xlabel(f'Time relative to {ElabelRNN} (ms)', fontsize=fontsize)
ax.set_ylabel(f'Average neural activity\nacross {n_neurons} neurons', fontsize=fontsize)
ax.set_title(f'Average neural activity for {n_conditions} conditions', fontsize=fontsize)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); 
ax.tick_params(axis='both', labelsize=fontsize)
fig.savefig('%s/average_neuralforRNN_across%gneurons_nTkeep%g_%gconditions.pdf'%(figure_dir,n_neurons,itimekeep_neural.size,n_conditions), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff

if VERBOSE: 
    fig, ax = plt.subplots()# average neural activity across all neurons (for each reach condition) minus mean activity (across all neurons and conditions) over time
    fontsize = 12
    for icondition in range(n_conditions):
        ax.plot(Tneural[itimekeep_neural], average_neuralforRNN_minusmeanovertime[:,icondition], '-', linewidth=3, c = colormapforconditions[icondition,:])
    if dataset_name == 'Mante13':
        ax.set_xlabel('Time points', fontsize=fontsize)
    else:
        ax.set_xlabel(f'Time relative to {ElabelRNN} (ms)', fontsize=fontsize)
    ax.set_ylabel(f'Average neural activity\nacross {n_neurons} neurons', fontsize=fontsize)
    ax.set_title(f'Average neural activity for {n_conditions} conditions', fontsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); 
    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('%s/average_neuralforRNN_across%gneurons_minusmeanovertime_nTkeep%g_%gconditions.pdf'%(figure_dir,n_neurons,itimekeep_neural.size,n_conditions), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


#%%############################################################################
#                           dPCA on neural data
###############################################################################
if plotPCA:
    data = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
    if dataset_name != 'Mante13':
        figure_xlabel = f'Time relative to {Elabelneural} (ms)'
    else:        
        figure_xlabel = 'Time points'
    figure_suffix = '_neuraldata'
    figure_title = ''# examples: figure_title = '' or figure_title = f'first line\nsecond line'
    computeDPCA(data, Tneural[itimekeep_neural], figure_xlabel, figure_dir, figure_suffix, figure_title, colormapforconditions, n_figuresplotforeachcomponent=1)# data has shape n_neurons x n_T x n_conditions
    if VERBOSE: 
        computeDPCA(data, Tneural[itimekeep_neural], figure_xlabel, figure_dir, figure_suffix+'_flip', figure_title, colormapforconditions, n_figuresplotforeachcomponent=1, FLIPAXES=1)# data has shape n_neurons x n_T x n_conditions
#import sys; sys.exit()# stop script at current line
   


#%%############################################################################
#                           PCA on neural data
###############################################################################
# if plotPCA:
data = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
figure_suffix = '_neuraldata'
if dataset_name == 'Mante13':
    figure_xlabel = 'Time points'
else:
    figure_xlabel = f'Time relative to {ElabelRNN} (ms)'
figure_title = ''# examples: figure_title = '' or figure_title = f'first line\nsecond line'
#PCAoutput_dictionary = computePCA(data, figure_suffix, figure_title, Elabelneural)# data has shape n_neurons x n_T x n_conditions
PCAinput_dictionary = {'Tdata':Tneural[itimekeep_neural], 'figure_xlabel':figure_xlabel, 'figure_dir':figure_dir, 'figure_suffix':figure_suffix, 'figure_title':figure_title, 'colormapforconditions':colormapforconditions}
PCAoutput_dictionary = computePCA(data, PLOTFIGURES=plotPCA, PCAinput_dictionary=PCAinput_dictionary)
data_dimensionality = PCAoutput_dictionary['dimensionality']
#import sys; sys.exit()# stop script at current line

plt.plot()


#%%############################################################################
#np.save(f'{figure_dir}/distance_info_dictionary.npy', distance_info_dictionary)# distance_info_dictionary = np.load(f'{figure_dir}/distance_info_dictionary.npy', allow_pickle='TRUE').item()
#import sys; sys.exit()# stop script at current line
if LOAD_STORED_DIST and os.path.exists(figure_dir) and os.path.exists(f'{figure_dir}/distance_train_store.npy'):# I should check that all of the files below exist but I'm just checking for one of them
# if False:  
    distance_info_dictionary = np.load(f'{figure_dir}/distance_info_dictionary.npy', allow_pickle='TRUE').item()
    pset = np.load(f'{figure_dir}/pset.npy')
    dimensionality_store = np.load(f'{figure_dir}/dimensionality_store.npy')# pset.size x n_models array, PCA participation ratio dimensionality
    distance_errornormalized_store = np.load(f'{figure_dir}/distance_errornormalized_store.npy')# pset.size x n_models array, how is error related to similarity?
    if CROSS_CONDITION_VALIDATION==0:
        distance_train_store = np.load(f'{figure_dir}/distance_train_store.npy')# pset.size x n_models array, store model/data similarity across n_conditions iterations of cross-validation
        distance_train_cross_condition_average_baseline_store = np.load(f'{figure_dir}/distance_train_cross_condition_average_baseline_store.npy')# a single number
    else:
        distance_train_store = np.load(f'{figure_dir}/distance_train_store.npy')# pset.size x n_models x n_conditions array, store model/data similarity across n_conditions iterations of cross-validation
        distance_test_store = np.load(f'{figure_dir}/distance_test_store.npy')# pset.size x n_models x n_conditions array, store model/data similarity across n_conditions iterations of cross-validation
        distance_train_cross_condition_average_baseline_store = np.load(f'{figure_dir}/distance_train_cross_condition_average_baseline_store.npy')# (n_conditions,) array
        distance_test_cross_condition_average_baseline_store = np.load(f'{figure_dir}/distance_test_cross_condition_average_baseline_store.npy')# (n_conditions,) array
            
    #distance_train_TARGETOUTvsneuralforRNN_baseline_store = np.load(f'{figure_dir}/distance_train_TARGETOUTvsneuralforRNN_baseline_store.npy')# (n_conditions,) array
    #distance_test_TARGETOUTvsneuralforRNN_baseline_store = np.load(f'{figure_dir}/distance_test_TARGETOUTvsneuralforRNN_baseline_store.npy')# (n_conditions,) array
else:
    #--------------------------------------------------------------------------
    if CROSS_CONDITION_VALIDATION==0:
        # cross-condition average baseline for model/data similarity is computed using only neural data
        # compare original neural dataset to one in which each neuron's firing rate across conditions is replaced by a single time varying firing rate that is obtained after averaging across conditions (each neuron can have a different time varying firing rate in this baseline dataset)
        # Train baseline: compute the distance between a concatenation of the original data over all N conditions, and the new average-over-N-conditions dataset concatenated over all N conditions (in the new dataset each condition is the same and is just repeated N times). 
        data = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset   
        distance_train_cross_condition_average_baseline_store = computecrossconditionaveragebaseline(data, similarityname, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, NORMALIZE_BETWEEN0AND1)# a single number
    else:
        # cross-condition average baseline for model/data similarity is computed using only neural data
        # compare original neural dataset to one in which each neuron's firing rate across conditions is replaced by a single time varying firing rate that is obtained after averaging across conditions (each neuron can have a different time varying firing rate in this baseline dataset)
        # Return N=n_conditions numbers. For each heldout condition compute the following:
        # Train baseline: compute the similarity between a concatenation of the original data over N-1 conditions, and the new average-over-all-N-conditions dataset concatenated over N-1 conditions (in the new dataset each condition is the same and is just repeated N-1 times)
        # Test baseline: apply parameters learned from train baseline to compute similarity on the Nth condition of the original dataset and the new average-over-all-N-conditions dataset
        data = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset   
        distance_train_cross_condition_average_baseline_store, distance_test_cross_condition_average_baseline_store = computecrossconditionaveragebaseline_crossconditionvalidation(data, similarityname, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, NORMALIZE_BETWEEN0AND1)# (n_conditions,) array
    #--------------------------------------------------------------------------
          
   
    
    # the following 7 lines assume that pset is the same for all models, in other words, the parameters are saved at the same points for all models
    data_dir = eval(f"data_dir{0}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    pset_saveparameters = np.load(f'{data_dir}/pset_saveparameters.npy')
    pset = pset_saveparameters[pset_saveparameters>=0]
    if similarityname == 'DSA':
        pset_idx=np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, len(pset)-1])
        # pset_idx=np.array([0, 20, 50, 80, len(pset)-1])
        pset = pset[pset_idx]
        pset_saveparameters = pset_saveparameters[pset_idx]
    
    dimensionality_store = -700*np.ones((pset.size, n_models))# PCA participation ratio dimensionality
    distance_errornormalized_store = -700*np.ones((pset.size, n_models))# how is error related to similarity?
    if CROSS_CONDITION_VALIDATION==0:
        distance_train_store = -700*np.ones((pset.shape[0], n_models))# store model/data similarity 
    else:
        distance_train_store = -700*np.ones((pset.shape[0], n_models, n_conditions))# store model/data similarity across n_conditions iterations of cross-validation
        distance_test_store = -700*np.ones((pset.shape[0], n_models, n_conditions))# store model/data similarity across n_conditions iterations of cross-validation
         
    
    for imodel in range(n_models):   
        data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
        model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
        # This save/load process uses the most intuitive syntax and involves the least amount of code. 
        # Saving a model in this way will save the entire module using Python’s pickle module. 
        # The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html

        #--------------------------
        # generateINandTARGETOUT can be different for each model
        # inputs and target outputs for RNN
        np.random.seed(random_seed); torch.manual_seed(random_seed)# set random seed for reproducible results
        model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
        model_class = model_info_dictionary['model_class']
        task_name_model = model_info_dictionary['task_name']
        n_input = model_info_dictionary['n_input']
        n_recurrent = model_info_dictionary['n_recurrent']
        n_output = model_info_dictionary['n_output']
        
        if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG} 
        #--------------------------
        activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        #--------------------------
        if dataset_name != 'Mante13':
            IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
            TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy();
        # IN:        (n_trials_test, n_T_test, n_input) tensor
        # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
        #--------------------------
    
        for ip, p in enumerate(pset):
            #activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
            checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
            if comment == '_sparse':
                model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise, 'conn_density':conn_density}
            else:
                model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
            model_output_forwardpass = model(model_input_forwardpass)
            output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']
            if dataset_name != 'Mante13':
                output = output.detach().numpy()
            activity = activity.detach().numpy()
            # output:   (n_trials_test, n_T_test, n_output) tensor
            # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        
            Y = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
            X = np.transpose(activity[:,itimekeep_RNN,:].copy(), axes=[2, 1, 0])# n_recurrent x n_Tkeep x n_trials_test, permute dimensions of array
            # X: (p, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets
            # Y: (q, n_datapoints, n_conditions) array, p and q can be different but n_datapoints and n_conditions must be the same for both datasets    
            
            if CROSS_CONDITION_VALIDATION==0:
                p, n_datapoints, n_conditions = X.shape
                q, n_datapoints, n_conditions = Y.shape
                if similarityname != 'DSA':
                    X_reshape = X.reshape(p,n_datapoints*n_conditions, order='F')# (p, n_datapoints*n_conditions) array
                    Y_reshape = Y.reshape(q,n_datapoints*n_conditions, order='F')# (q, n_datapoints*n_conditions) array
                else:
                    X_reshape = np.transpose(X.copy(), axes=[2, 1, 0]) # b x T x N
                    Y_reshape = np.transpose(Y.copy(), axes=[2, 1, 0])
                distance_train, distance_test = computedistance(X_reshape, Y_reshape, X_reshape, Y_reshape, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)# returns a single number, model/data distance 
                distance_train_store[ip,imodel] = distance_train# a single number, model/data distance
                print(f"Model {imodel}, parameters {ip}/{pset.size-1}, mean distance train = {distance_train:.4g}, mean baseline train = {distance_train_cross_condition_average_baseline_store:.4g}")
            else:
                distance_train, distance_test = computedistance_crossconditionvalidation(X, Y, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)# (n_conditions,) arrays, model/data distance across n_conditions iterations of cross-validation
                distance_train_store[ip,imodel,:] = distance_train# (n_conditions,) array, model/data similarity across n_conditions iterations of cross-validation
                distance_test_store[ip,imodel,:] = distance_test# (n_conditions,) array, model/data similarity across n_conditions iterations of cross-validation
                print(f"Model {imodel}, parameters {ip}/{pset.size-1}, mean distance train/test = {np.mean(distance_train):.4g}/{np.mean(distance_test):.4g}, mean baseline train/test = {np.mean(distance_train_cross_condition_average_baseline_store):.4g}/{np.mean(distance_test_cross_condition_average_baseline_store):.4g}")
            
            #-------------------------------------------------------------------------
            # normalized error, if RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
            # outputforerror = output(output_mask==1)
            # TARGETOUTforerror = TARGETOUT(output_mask==1)
            # errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
            if dataset_name == 'Mante13':
                TARGETOUT_flat = TARGETOUT.reshape(-1).long()  # Ensure TARGETOUT is Long      
                output_flat = output.reshape(-1, output.size(2))
                # Flatten output_mask to align with the reshaped output tensor
                output_mask_flat = output_mask.reshape(-1)  # Flatten output_mask                
                # Select the entries that are marked by the output_mask
                selected_outputs = output_flat[output_mask_flat.nonzero().squeeze(), :]  # Outputs selected by the mask
                selected_targets = TARGETOUT_flat[output_mask_flat.nonzero().squeeze()]  # Ensure this is Long                
                # Compute cross-entropy loss for the selected entries
                errormain = torch.nn.functional.cross_entropy(selected_outputs, selected_targets)
                distance_errornormalized_store[ip,imodel] = errormain.detach().numpy()
            else:
                errornormalized = compute_normalized_error(TARGETOUT, output, output_mask)# all inputs are arrays with shape (n_trials, n_T, n_output)
                distance_errornormalized_store[ip,imodel] = errornormalized 
            #-------------------------------------------------------------------------
            PCAoutput_dictionary = computePCA(data=X, PLOTFIGURES=0, PCAinput_dictionary={})# # data: (n_neurons, n_T, n_conditions) array, this is the only information that enters into the PCA computation, the rest of the inputs are for the figures
            dimensionality_store[ip,imodel] = PCAoutput_dictionary['dimensionality']
            #-------------------------------------------------------------------------
    distance_info_dictionary['n_T_test'] = n_T_test
    distance_info_dictionary['n_trials_test'] = n_trials_test
    np.save(f'{figure_dir}/distance_info_dictionary.npy', distance_info_dictionary)# distance_info_dictionary = np.load(f'{figure_dir}/distance_info_dictionary.npy', allow_pickle='TRUE').item()
    np.save(f'{figure_dir}/pset.npy', pset)
    np.save(f'{figure_dir}/dimensionality_store.npy', dimensionality_store)
    np.save(f'{figure_dir}/distance_errornormalized_store.npy', distance_errornormalized_store)
    if CROSS_CONDITION_VALIDATION==0:
        np.save(f'{figure_dir}/distance_train_store.npy', distance_train_store)
        np.save(f'{figure_dir}/distance_train_cross_condition_average_baseline_store.npy', distance_train_cross_condition_average_baseline_store)
    else:
        np.save(f'{figure_dir}/distance_train_store.npy', distance_train_store)
        np.save(f'{figure_dir}/distance_test_store.npy', distance_test_store)
        np.save(f'{figure_dir}/distance_train_cross_condition_average_baseline_store.npy', distance_train_cross_condition_average_baseline_store)
        np.save(f'{figure_dir}/distance_test_cross_condition_average_baseline_store.npy', distance_test_cross_condition_average_baseline_store)
    

       
#%%############################################################################
#       plot activity over training         
##############################################################################   
    
# # the following 7 lines assume that pset is the same for all models, in other words, the parameters are saved at the same points for all models
# data_dir = eval(f"data_dir{0}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
# pset_saveparameters = np.load(f'{data_dir}/pset_saveparameters.npy')

# for imodel in [11]: # the model index to plot # range(n_models):   
#     data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
#     model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
#     # This save/load process uses the most intuitive syntax and involves the least amount of code. 
#     # Saving a model in this way will save the entire module using Python’s pickle module. 
#     # The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.
#     # https://pytorch.org/tutorials/beginner/saving_loading_models.html

#     #--------------------------
#     # generateINandTARGETOUT can be different for each model
#     # inputs and target outputs for RNN
#     np.random.seed(random_seed); torch.manual_seed(random_seed)# set random seed for reproducible results
#     model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
#     model_class = model_info_dictionary['model_class']
#     task_name_model = model_info_dictionary['task_name']
#     n_input = model_info_dictionary['n_input']
#     n_recurrent = model_info_dictionary['n_recurrent']
#     n_output = model_info_dictionary['n_output']
    
#     if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
#         toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
#         OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
#         task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG} 
#     #--------------------------
#     activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
#     #--------------------------
#     IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
#     TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy();
#     # IN:        (n_trials_test, n_T_test, n_input) tensor
#     # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
#     #--------------------------

#     for ip, p in enumerate(pset):
#         #activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
#         checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
#         model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
#         model_output_forwardpass = model(model_input_forwardpass)
#         output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']
#         output = output.detach().numpy(); activity = activity.detach().numpy()
#         # output:   (n_trials_test, n_T_test, n_output) tensor
#         # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        
#         if (p >= 100) and ((ip%10)==0): 
#             plt.figure()
#             plt.pcolor(np.mean(activity[:,150:],axis=1), vmin=0.0, vmax=1.0)
#             plt.xlabel('Neuron #')
#             plt.ylabel('Condition')
#             plt.title('Average activity across last few dt, train step:' + str(p))
#             plt.colorbar() 
            
#             # plt.figure()
#             # for cond in range(output.shape[0]): 
#             #     plt.plot(TARGETOUT[cond], '-')
#             #     plt.plot(output[cond],'--')
        

#%%###########################################################################
#              plot normalized error vs number of parameter updates
##############################################################################
if VERBOSE: 
    n_T_test = distance_info_dictionary['n_T_test']
    n_trials_test = distance_info_dictionary['n_trials_test']
    
    fig, ax = plt.subplots()# normalized error vs number of parameter updates
    fontsize = 13
    for imodel in range(n_models):
        ax.plot(pset, distance_errornormalized_store[:,imodel], '-', color=eval(f"color{imodel}"), linewidth=1, label=eval(f"legend_label{imodel}"))     
    ax.legend(frameon=True, loc='best')   
    ax.set_xlabel('Number of parameter updates', fontsize=fontsize)
    if dataset_name == 'Mante13':
        ax.set_ylabel('Error', fontsize=fontsize)
    else: 
        ax.set_ylabel('Normalized error', fontsize=fontsize)
    ax.set_xlim(xmin=0, xmax=None); ax.set_ylim(ymin=0, ymax=None)
    ax.set_title(f'{n_trials_test} test trials, {n_T_test} timesteps', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('%s/main_errornormalized_vs_numberofparameterupdates_nT%g.pdf'%(figure_dir,n_T_test), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
             

#%%###########################################################################
#            plot dimensionality vs number of parameter updates
##############################################################################
if PLOTLEGEND==1: figure_suffix = ''
# for PLOTLEGEND in range(2):# 0,1
#     if PLOTLEGEND==0: figure_suffix = '_nolegend'
#     if PLOTLEGEND==1: figure_suffix = ''
    
fig, ax = plt.subplots()# dimensionality versus number of parameter updates  
fontsize = 13
for imodel in range(n_models):
    ax.plot(pset, dimensionality_store[:,imodel], '-', color=eval(f"color{imodel}"), linewidth=4, label=eval(f"legend_label{imodel}"))
# ax.axhline(y=data_dimensionality, color='grey', linewidth=6, linestyle='--', label="neural data") # mark neural data dimensionality 
if PLOTLEGEND: ax.legend(frameon=True, framealpha=1, loc='best')
ax.set_xlabel('Number of parameter updates', fontsize=fontsize)
ax.set_ylabel('Participation ratio dimensionality', fontsize=fontsize)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize)
fig.savefig('%s/dimensionalityvsnumberofparameterupdates_nTkeep%g%s.pdf'%(figure_dir,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
#import sys; sys.exit()# stop script at current line
     

#%%###########################################################################
#                     choose model/data similarity to plot
##############################################################################
xplot = np.arange(0,n_models+1)# each model gets one bar showing distance for the best parameters
if CROSS_CONDITION_VALIDATION==0:
    yplot_train = -700*np.ones(n_models)
    yplot_train_cross_condition_average_baseline = distance_train_cross_condition_average_baseline_store# cross-condition average baseline for model/data similarity is computed using only neural data
else:  
    yplot_train = -700*np.ones(n_models)
    yplot_test = -700*np.ones(n_models)
    ystd_train = -700*np.ones(n_models)
    ystd_test = -700*np.ones(n_models)
    yplot_train_cross_condition_average_baseline = np.mean(distance_train_cross_condition_average_baseline_store)# cross-condition average baseline for model/data similarity is computed using only neural data
    yplot_test_cross_condition_average_baseline = np.mean(distance_test_cross_condition_average_baseline_store)# cross-condition average baseline for model/data similarity is computed using only neural data
ipbest = -700*np.ones(n_models)# index of best model parameters, pset[ipbest[imodel]] are the best parameters for the model
pbest = -700*np.ones(n_models)# best model parameters, pbest[imodel] = pset[ipbest[imodel]] are the best parameters for the model
for imodel in range(n_models):
    data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    if CROSS_CONDITION_VALIDATION==0:
        distance_train = distance_train_store[:,imodel]# (pset.size,) array
    else:
        mean_distance_train = np.mean(distance_train_store[:,imodel,:],1)# (pset.size,) array, mean across n_conditions iterations of cross-validation
        mean_distance_test = np.mean(distance_test_store[:,imodel,:],1)# (pset.size,) array, mean across n_conditions iterations of cross-validation
        std_distance_train = np.std(distance_train_store[:,imodel,:],1)# (pset.size,) array, std across n_conditions iterations of cross-validation
        std_distance_test = np.std(distance_test_store[:,imodel,:],1)# (pset.size,) array, std across n_conditions iterations of cross-validation
    
    '''
    # plot maximum model/data test similarity across training for each model, OLD CAN DELETE!!
    imax_test = np.argmax(mean_distance_test)# index of maximum model/data test similarity
    ipbest[imodel] = imax_test; pbest[imodel] = pset[imax_test]
    yplot_train[imodel] = mean_distance_train[imax_test]
    yplot_test[imodel] = mean_distance_test[imax_test]
    ystd_train[imodel] = std_distance_train[imax_test]
    ystd_test[imodel] = std_distance_test[imax_test]
    pbestname = 'RNNparametersatmaxtestsimilarity'
    '''
    '''
    # plot maximum model/data train similarity across training for each model, worst model
    imax_train = np.argmax(mean_distance_train)# index of maximum model/data train similarity
    ipbest[imodel] = imax_train; pbest[imodel] = pset[imax_train]
    yplot_train[imodel] = mean_distance_train[imax_train]
    yplot_test[imodel] = mean_distance_test[imax_train]
    ystd_train[imodel] = std_distance_train[imax_train]
    ystd_test[imodel] = std_distance_test[imax_train]
    pbestname = 'RNNparametersatmaxtraindistance'
    '''
    '''
    # plot minimum model/data test distance across training for each model
    imin_test = np.nanargmin(mean_distance_test)# index of minimum model/data test distance
    ipbest[imodel] = imin_test; pbest[imodel] = pset[imin_test]
    yplot_train[imodel] = mean_distance_train[imin_test]
    yplot_test[imodel] = mean_distance_test[imin_test]
    ystd_train[imodel] = std_distance_train[imin_test]
    ystd_test[imodel] = std_distance_test[imin_test]
    pbestname = 'RNNparametersatmintestdistance'
    '''
    
    # plot minimum model/data train distance across training for each model
    if CROSS_CONDITION_VALIDATION==0:
        imin_train = np.nanargmin(distance_train)# index of minimum model/data train distance
        ipbest[imodel] = imin_train; pbest[imodel] = pset[imin_train]
        yplot_train[imodel] = distance_train[imin_train]
        pbestname = 'RNNparametersatmintraindistance'
    else:
        imin_train = np.nanargmin(mean_distance_train)# index of minimum model/data train distance
        ipbest[imodel] = imin_train; pbest[imodel] = pset[imin_train]
        yplot_train[imodel] = mean_distance_train[imin_train]
        yplot_test[imodel] = mean_distance_test[imin_train]
        ystd_train[imodel] = std_distance_train[imin_train]
        ystd_test[imodel] = std_distance_test[imin_train]
        pbestname = 'RNNparametersatmintraindistance'
   
    '''
    # plot model/data similarity at the training iteration when errornormalized first drops below some cutoff
    cutoff = 0.15# 0=perfect, 1=RNN outputting mean
    icutoff = np.flatnonzero(distance_errornormalized_store[:,imodel] <= cutoff)
    icutoff = icutoff[0]# index of first time errornormalized drops below cutoff
    ipbest[imodel] = icutoff; pbest[imodel] = pset[icutoff]
    yplot_train[imodel] = mean_distance_train[icutoff]
    yplot_test[imodel] = mean_distance_test[icutoff]
    ystd_train[imodel] = std_distance_train[icutoff]
    ystd_test[imodel] = std_distance_test[icutoff]
    pbestname = f'RNNparametersaterror{cutoff}'
    '''
    '''
    # plot model/data similarity at the training iteration when errornormalized is a minimum
    imin = np.nanargmin(distance_errornormalized_store[:,imodel])
    ipbest[imodel] = imin; pbest[imodel] = pset[imin]
    yplot_train[imodel] = mean_distance_train[imin]
    yplot_test[imodel] = mean_distance_test[imin]
    ystd_train[imodel] = std_distance_train[imin]
    ystd_test[imodel] = std_distance_test[imin]
    pbestname = 'RNNparametersatminerrornormalized'
    '''
    '''
    # plot initial parameters before training
    ikeep = 0# index of parameters before training
    ipbest[imodel] = ikeep; pbest[imodel] = pset[ikeep]
    yplot_train[imodel] = mean_distance_train[ikeep]
    yplot_test[imodel] = mean_distance_test[ikeep]
    ystd_train[imodel] = std_distance_train[ikeep]
    ystd_test[imodel] = std_distance_test[ikeep]
    pbestname = 'RNNparametersbeforetraining'
    '''
ipbest = ipbest.astype(int)# convert to integers so we can use as index
pbest = pbest.astype(int)# save as int so we can use elements to load model parameters: for example, model_parameter_update211.pth versus model_parameter_update211.0.pth   
desiredorder = np.arange(n_models)# plot model 0 first
xtick_label_desiredorder = xtick_label.copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)
'''  
# sort bars by height, plot smallest model/data distance score first
indices = np.argsort(yplot_test)# np.argsort returns index of smallest element first 
desiredorder = indices# desiredorder[0] is the model that is plotted first
xtick_label_desiredorder = xtick_label.copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)
xtick_label_desiredorder = [xtick_label_desiredorder[i] for i in desiredorder]# order list by indices xtick_label = xtick_label[indices]
'''
'''   
# sort bars by height, plot largest model/data similarity score first, OLD CAN DELETE!!
indices = np.argsort(yplot_test)[::-1]# np.argsort returns index of smallest element first so reverse the order with [::-1]
desiredorder = indices# desiredorder[0] is the model that is plotted first
xtick_label_desiredorder = xtick_label.copy()# use B = A.copy() so changing B doesn't change A (also changing A doesn't change B)
xtick_label_desiredorder = [xtick_label_desiredorder[i] for i in desiredorder]# order list by indices xtick_label = xtick_label[indices]
'''
# find indices of models with lowest yplot_train
i5best_models_train = np.argsort(yplot_train)# sort yplot_train in increasing order, so smallest is first
i5best_models_train = i5best_models_train[0:5]# 5 best models
if CROSS_CONDITION_VALIDATION==1:
    i5best_models_test = np.argsort(yplot_test)# sort yplot_train in increasing order, so smallest is first
    i5best_models_test = i5best_models_test[0:5]# 5 best models
#yplot_train[i5best_models_train]
#yplot_test[i5best_models_test]
#import sys; sys.exit()# stop script at current line


#%%###########################################################################
#                    RNN input and outputs over time
##############################################################################
# if CROSS_CONDITION_VALIDATION==0: imodels_plot = np.array([i5best_models_train[0]])
# if CROSS_CONDITION_VALIDATION==1: imodels_plot = np.array([i5best_models_train[0], i5best_models_test[0]])   
# Warning('Only plotting the input output for the best model(s)')   
if VERBOSE: 
    
    imodels_plot = np.arange(n_models)                                                   
    for imodel in imodels_plot:
        data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
        model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
        model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
        n_input = model_info_dictionary['n_input']
        n_output = model_info_dictionary['n_output']
        n_recurrent = model_info_dictionary['n_recurrent']
        model_class = model_info_dictionary['model_class']# model name for forward pass
        task_name_model = model_info_dictionary['task_name']   
        
        #--------------------------
        # generateINandTARGETOUT can be different for each model
        # inputs and target outputs for RNN
        if task_name_model[0:15]=='Hatsopoulos2007':# this assumes task_name_model starts with the phrase Hatsopoulos2007
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTSHOULDERELBOWANGLES=model_info_dictionary['OUTPUTSHOULDERELBOWANGLES']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTSHOULDERELBOWANGLES
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTSHOULDERELBOWANGLES':OUTPUTSHOULDERELBOWANGLES}
        if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
        #--------------------------
        if dataset_name != 'Mante13':
            IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
            TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy(); 
        # IN:        (n_trials_test, n_T_test, n_input) tensor
        # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
        #--------------------------
        
        p = pbest[imodel]# choose RNN parameters to load (at the update step where the best distance is achieved) 
        n_parameter_updates_model = p
        checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
        activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        if comment == '_sparse':
            model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise, 'conn_density':conn_density}
        else:
            model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
        model_output_forwardpass = model(model_input_forwardpass)
        output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']
        if dataset_name != 'Mante13':
            output = output.detach().numpy() 
        activity = activity.detach().numpy()
        # output:   (n_trials_test, n_T_test, n_output) tensor
        # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        
        
        # normalized error, if RNN output is constant for each n_output (each n_output can be a different constant) then errornormalized = 1
        # outputforerror = output(output_mask==1)
        # TARGETOUTforerror = TARGETOUT(output_mask==1)
        # errornormalized = ((outputforerror(:) - TARGETOUTforerror(:))' @ (outputforerror(:) - TARGETOUTforerror(:))) / ((mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))' @ (mean(TARGETOUTforerror(:)) - TARGETOUTforerror(:))), normalized error when using outputs for which output_mask = 1
        if dataset_name != 'Mante13':
            errornormalized = compute_normalized_error(TARGETOUT, output, output_mask)# all inputs are arrays with shape (n_trials, n_T, n_output)
        else:
            TARGETOUT_flat = TARGETOUT.reshape(-1).long()  # Ensure TARGETOUT is Long      
            output_flat = output.reshape(-1, output.size(2))
            # Flatten output_mask to align with the reshaped output tensor
            output_mask_flat = output_mask.reshape(-1)  # Flatten output_mask                
            # Select the entries that are marked by the output_mask
            selected_outputs = output_flat[output_mask_flat.nonzero().squeeze(), :]  # Outputs selected by the mask
            selected_targets = TARGETOUT_flat[output_mask_flat.nonzero().squeeze()]  # Ensure this is Long                
            # Compute cross-entropy loss for the selected entries
            errornormalized = torch.nn.functional.cross_entropy(selected_outputs, selected_targets)
        
        figure_suffix = f'_model{imodel}_{pbestname}'
        fontsize = 13
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
            plt.legend(loc='best', fontsize=fontsize)
            if CROSS_CONDITION_VALIDATION==0:
                plt.title(f'{n_recurrent} unit {model_class}, {similarityname} train = {yplot_train[imodel]:.4g}\n'
                          f'trial {itrial}, {n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
            else:
                plt.title(f'{n_recurrent} unit {model_class}, {similarityname} train/test = {yplot_train[imodel]:.4g}/{yplot_test[imodel]:.4g}\n'
                          f'trial {itrial}, {n_trials_test} test trials, {n_T_test} timesteps in simulation\n{n_parameter_updates_model} parameter updates, normalized error = {errornormalized:.6g}', fontsize=fontsize)
            plt.xlim(left=0)
            plt.tick_params(axis='both', labelsize=fontsize)
            #plt.show(); input("Press Enter to continue...")# pause the program until the user presses Enter, https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file
            plt.savefig('%s/testtrial%g_nTtest%g_%gparameterupdates_model%g_%s.pdf'%(figure_dir,itrial,n_T_test,n_parameter_updates_model,imodel,pbestname), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff



#%%###########################################################################
#                           PCA and dPCA of model 
##############################################################################
if plotPCA: 
    for imodel in range(n_models):  
    #for imodel in np.array([i5best_models_train[0]]):
    #for imodel in range(0): 
    #for imodel in np.array([76]):
        data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
        model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
        model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
        n_input = model_info_dictionary['n_input']
        n_output = model_info_dictionary['n_output']
        n_recurrent = model_info_dictionary['n_recurrent']
        model_class = model_info_dictionary['model_class']# model name for forward pass
        task_name_model = model_info_dictionary['task_name']   
        
        #--------------------------
        # generateINandTARGETOUT can be different for each model
        # inputs and target outputs for RNN
        if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
        #--------------------------
        activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        #--------------------------
        if dataset_name != 'Mante13':
            IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
            TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy(); 
        # IN:        (n_trials_test, n_T_test, n_input) tensor
        # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
        #--------------------------
        
        p = pbest[imodel]# choose RNN parameters to load
        checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
        #activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        if comment == '_sparse':
            model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise, 'conn_density':conn_density}
        else:
            model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
        model_output_forwardpass = model(model_input_forwardpass)
        output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']
        output = output.detach().numpy(); activity = activity.detach().numpy()
        # output:   (n_trials_test, n_T_test, n_output) tensor
        # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        
        #----------
        data = np.transpose(activity[:,itimekeep_RNN,:].copy(), axes=[2, 1, 0])# n_recurrent x n_T_test x n_trials_test, permute dimensions of array  
        Tdata = Tneural[itimekeep_neural]
        if dataset_name == 'Mante13':
            figure_xlabel = 'Time points'
        else:
            figure_xlabel = f'Time relative to {ElabelRNN} (ms)'
        figure_suffix = f'_model{imodel}_{pbestname}'
        #figure_suffix = f'_model{imodel}_{pbestname}_flip'
        #figure_suffix = f'_model{imodel}_{pbestname}_interval2is57'
        if CROSS_CONDITION_VALIDATION==0: titlestring = f'{n_recurrent} unit {model_class}, {similarityname} train = {yplot_train[imodel]:.4g}\n'# examples: titlestring = '' or titlestring = f'first line\n'
        if CROSS_CONDITION_VALIDATION==1: titlestring = f'{n_recurrent} unit {model_class}, {similarityname} train/test = {yplot_train[imodel]:.4g}/{yplot_test[imodel]:.4g}\n'# examples: titlestring = '' or titlestring = f'first line\n'
        n_figuresplot = 1# plot data over time projected onto PC1, PC2, PC3,...,PCn_figuresplot
        if imodel == i5best_models_train[0]:
            n_figuresplot = 5# plot data over time projected onto PC1, PC2, PC3,...,PCn_figuresplot
        PCAinput_dictionary = {'n_figuresplot':n_figuresplot, 'Tdata':Tdata, 'figure_xlabel':figure_xlabel, 'figure_dir':figure_dir, 'figure_suffix':figure_suffix, 'figure_title':titlestring, 'colormapforconditions':colormapforconditions}
        PCAoutput_dictionary = computePCA(data, PLOTFIGURES=1, PCAinput_dictionary=PCAinput_dictionary)# data has shape n_neurons x n_T x n_conditions
        #----------
        n_figuresplotforeachcomponent = 1# for each of the three components (time, condition, time x condition) plot n_figuresplotforeachcomponent figures
        computeDPCA(data, Tdata, figure_xlabel, figure_dir, figure_suffix, titlestring, colormapforconditions, n_figuresplotforeachcomponent)# data has shape n_neurons x n_T x n_conditions
        if VERBOSE: 
            if CROSS_CONDITION_VALIDATION==0 and (imodel == i5best_models_train[0]):
                computeDPCA(data, Tdata, figure_xlabel, figure_dir, figure_suffix+'_flip', titlestring, colormapforconditions, n_figuresplotforeachcomponent, FLIPAXES=1)# data has shape n_neurons x n_T x n_conditions   
            if CROSS_CONDITION_VALIDATION==1 and ((imodel == i5best_models_train[0]) or (imodel == i5best_models_test[0])):
                computeDPCA(data, Tdata, figure_xlabel, figure_dir, figure_suffix+'_flip', titlestring, colormapforconditions, n_figuresplotforeachcomponent, FLIPAXES=1)# data has shape n_neurons x n_T x n_conditions   
        #----------
#import sys; sys.exit()# stop script at current line


#%%###########################################################################
#     for each model compute normalized-with-data distance using best parameters 
##############################################################################
if similarityname != 'DSA':
    n_iterations = 100# n_iterations: compute normalized distance n_iterations times with different samples from X and Y on each iteration 
else:
    n_iterations = 5
# if False:
if LOAD_STORED_DIST and os.path.exists(figure_dir) and os.path.exists(f'{figure_dir}/distance_normalizedwithdata_train.npy'):# I should check that all of the files below exist but I'm just checking for one of them 
    n_sample_for_distance_normalizedwithdata = np.load(f'{figure_dir}/n_sample_for_distance_normalizedwithdata.npy')# n_sample: all normalized-with-data distances are computed between between n_sample neurons/units
    distance_normalizedwithdata_train = np.load(f'{figure_dir}/distance_normalizedwithdata_train.npy')
    distance_normalizedwithdata_train_cross_condition_average_baseline = np.load(f'{figure_dir}/distance_normalizedwithdata_train_cross_condition_average_baseline.npy')
    distance_neuraltoneuralsubsampled_between0and1 = np.load(f'{figure_dir}/distance_neuraltoneuralsubsampled_between0and1.npy')
    distance_modeltoneuralsubsampled_between0and1 = np.load(f'{figure_dir}/distance_modeltoneuralsubsampled_between0and1.npy')
    distance_crossconditionaverageneuraltoneuralsubsampled_between0and1 = np.load(f'{figure_dir}/distance_crossconditionaverageneuraltoneuralsubsampled_between0and1.npy')
else:     
    distance_normalizedwithdata_train = -700*np.ones((n_iterations, n_models))
    distance_normalizedwithdata_train_cross_condition_average_baseline = -700*np.ones((n_iterations, n_models))
    distance_neuraltoneuralsubsampled_between0and1 = -700*np.ones((n_iterations, n_models))
    distance_modeltoneuralsubsampled_between0and1 = -700*np.ones((n_iterations, n_models))
    distance_crossconditionaverageneuraltoneuralsubsampled_between0and1 = -700*np.ones((n_iterations, n_models))   
    for imodel in range(n_models):# index of model with lowest training distance
        data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
        model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
        model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
        n_input = model_info_dictionary['n_input']
        n_output = model_info_dictionary['n_output']
        n_recurrent = model_info_dictionary['n_recurrent']
        model_class = model_info_dictionary['model_class']# model name for forward pass
        task_name_model = model_info_dictionary['task_name']   
        
        #--------------------------
        # generateINandTARGETOUT can be different for each model
        # inputs and target outputs for RNN
        if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
        #--------------------------
        if dataset_name != 'Mante13':
            IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
            TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy(); 
        # IN:        (n_trials_test, n_T_test, n_input) tensor
        # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
        #--------------------------
        
        p = pbest[imodel]# choose RNN parameters to load
        #p = 0# initial parameters for model before training
        checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
        activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        if comment == '_sparse':
            model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise, 'conn_density':conn_density}
        else:
            model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
        model_output_forwardpass = model(model_input_forwardpass)
        output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']# (n_trials, n_T, n_output/n_recurrent)
        output = output.detach().numpy(); activity = activity.detach().numpy()
        # output:   (n_trials_test, n_T_test, n_output) tensor
        # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        X_neuraldata = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
        Y = np.transpose(activity[:,itimekeep_RNN,:].copy(), axes=[2, 1, 0])# n_recurrent x n_Tkeep x n_trials_test, permute dimensions of array
        
        #n_sample_max = np.minimum(np.floor(X_neuraldata.shape[0]/2), np.floor(Y.shape[0]/2)).astype(int)# convert to integer so we can use as index
        n_sample_max = np.floor(X_neuraldata.shape[0]/2).astype(int)# convert to integer so we can use as index
        n_sample_for_distance_normalizedwithdata = n_sample_max# n_sample: all normalized-with-data distances are computed between between n_sample neurons/units
        distance_normalizedwithdata_output_dictionary = computedistance_normalizedwithdata(X_neuraldata, Y, similarityname, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, n_sample_for_distance_normalizedwithdata, n_iterations, random_seed=imodel)# (n_iterations,) arrays
        distance_normalizedwithdata = distance_normalizedwithdata_output_dictionary['distance_normalizedwithdata']# (n_iterations,) array
        distance_normalizedwithdata_baseline = distance_normalizedwithdata_output_dictionary['distance_normalizedwithdata_baseline']# (n_iterations,) array
        distance_neuraltoneuralsubsampled_between0and1_ = distance_normalizedwithdata_output_dictionary['distance_neuraltoneuralsubsampled_between0and1']# (n_iterations,) array
        distance_modeltoneuralsubsampled_between0and1_ = distance_normalizedwithdata_output_dictionary['distance_modeltoneuralsubsampled_between0and1']# (n_iterations,) array
        distance_crossconditionaverageneuraltoneuralsubsampled_between0and1_ = distance_normalizedwithdata_output_dictionary['distance_crossconditionaverageneuraltoneuralsubsampled_between0and1']# (n_iterations,) array
        #--------------------------
        distance_normalizedwithdata_train[:,imodel] = distance_normalizedwithdata
        distance_normalizedwithdata_train_cross_condition_average_baseline[:,imodel] = distance_normalizedwithdata_baseline
        distance_neuraltoneuralsubsampled_between0and1[:,imodel] = distance_neuraltoneuralsubsampled_between0and1_
        distance_modeltoneuralsubsampled_between0and1[:,imodel] = distance_modeltoneuralsubsampled_between0and1_
        distance_crossconditionaverageneuraltoneuralsubsampled_between0and1[:,imodel] = distance_crossconditionaverageneuraltoneuralsubsampled_between0and1_
        print(f'model{imodel}: mean(distance_normalizedwithdata)+-2*std = {np.mean(distance_normalizedwithdata):.3g}+-{2*np.std(distance_normalizedwithdata):.3g}')
    np.save(f'{figure_dir}/n_sample_for_distance_normalizedwithdata.npy', n_sample_for_distance_normalizedwithdata)
    np.save(f'{figure_dir}/distance_normalizedwithdata_train.npy', distance_normalizedwithdata_train)
    np.save(f'{figure_dir}/distance_normalizedwithdata_train_cross_condition_average_baseline.npy', distance_normalizedwithdata_train_cross_condition_average_baseline)
    np.save(f'{figure_dir}/distance_neuraltoneuralsubsampled_between0and1.npy', distance_neuraltoneuralsubsampled_between0and1)
    np.save(f'{figure_dir}/distance_modeltoneuralsubsampled_between0and1.npy', distance_modeltoneuralsubsampled_between0and1)
    np.save(f'{figure_dir}/distance_crossconditionaverageneuraltoneuralsubsampled_between0and1.npy', distance_crossconditionaverageneuraltoneuralsubsampled_between0and1)
'''
for iteration in range(2):# scatterplot showing model/neural and neural/neural distances
    #PLOTBASELINE = 1# if 1 plot cross condition average baseline
    if iteration==0: PLOTLEGEND = 1# if 1 plot legend
    if iteration==1: PLOTLEGEND = 0# if 1 plot legend
    similaritynametoplot = similarityname + 'between0and1' + f'on{n_sample_for_distance_normalizedwithdata}neurons'; figure_suffix = pbestname + '_train' + '_crossconditionaveragebaseline' + '_noisefloor'; yplot_baseline = np.mean(distance_crossconditionaverageneuraltoneuralsubsampled_between0and1)
    fontsize = 10 
    fig, ax = plt.subplots()
    for imodel in range(n_models):
        iposition = np.flatnonzero(desiredorder == imodel)[0]# model imodel is plotted at position iposition
        ax.plot(iposition*np.ones(n_iterations), distance_neuraltoneuralsubsampled_between0and1[:,imodel], '.', markersize=5, color='k', label='neural to neural distance'if imodel == 0 else "")
        ax.plot(iposition*np.ones(n_iterations), distance_modeltoneuralsubsampled_between0and1[:,imodel], '.', markersize=5, color=eval(f"color{imodel}"), label=eval(f"legend_label{imodel}"))   
    ax.plot([0-0.5, n_models-0.5], [yplot_baseline, yplot_baseline], "k--", linewidth=1);# figure_suffix = figure_suffix + '_crossconditionaveragebaseline'; # horizontal line indicating the cross condition average baseline
    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    ax.set_xticks(xplot); 
    ax.set_xticklabels(xtick_label_desiredorder, rotation = 45, ha="right", fontsize=1)
    ax.set_ylabel(f'{similaritynametoplot}\ntrain distance scaled between 0 and 1', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0)
    if PLOTLEGEND==1: ax.legend(frameon=True, framealpha=1, loc='lower right')
    if PLOTLEGEND==0: figure_suffix = figure_suffix + '_nolegend'
    if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_scatterplot%s_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_scatterplot%s_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_scatterplot%s_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
#import sys; sys.exit()# stop script at current line    
''' 

'''
#%%###########################################################################
#    vary interval2set and see if this improves model/data similarity
##############################################################################
#imodel_set = np.array([21, 62, 66, 68, 76, 38, 39]); interval2set_new = 62 + np.arange(-10, 6)# interval after the input specifying the condition and before the hold-cue turns off
imodel_set = np.array([0, n_models-1]); interval2set_new = interval2set + np.arange(-20, 20)# interval after the input specifying the condition and before the hold-cue turns off
imodel_set = i5best_models_train; interval2set_new = interval2set + np.arange(-20, 21)# interval after the input specifying the condition and before the hold-cue turns off

distance_train = -700*np.ones((imodel_set.size, interval2set_new.size))
distance_test = -700*np.ones((imodel_set.size, interval2set_new.size))
for ii, imodel in enumerate(imodel_set):
    for jj in range(interval2set_new.size):
        data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
        model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
        model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
        n_input = model_info_dictionary['n_input']
        n_output = model_info_dictionary['n_output']
        n_recurrent = model_info_dictionary['n_recurrent']
        model_class = model_info_dictionary['model_class']# model name for forward pass
        task_name_model = model_info_dictionary['task_name']   
            
        #--------------------------
        # generateINandTARGETOUT can be different for each model
        # inputs and target outputs for RNN
        if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set_new[jj][None], 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
        #--------------------------
        IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
        TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy()
        # IN:        (n_trials_test, n_T_test, n_input) tensor
        # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
        #--------------------------
        p = pbest[imodel]# choose RNN parameters to load
        #p = 0# initial parameters for model before training
        checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
        activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
        model_output_forwardpass = model(model_input_forwardpass)
        output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']
        output = output.detach().numpy(); activity = activity.detach().numpy()
        # output:   (n_trials_test, n_T_test, n_output) tensor
        # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        X_neuraldata = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
        Y = np.transpose(activity[:,itimekeep_RNN,:].copy(), axes=[2, 1, 0])# n_recurrent x n_Tkeep x n_trials_test, permute dimensions of array
        
        distance_train_, distance_test_ = computedistance_crossconditionvalidation(X_neuraldata, Y, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)# (n_conditions,) arrays, model/data distance across n_conditions iterations of cross-validation
        distance_train[ii,jj] = np.mean(distance_train_)
        distance_test[ii,jj] = np.mean(distance_test_)
  
for iteration in range(2):
    PLOTBASELINE = 0# if 1 plot cross condition average baseline
    PLOTLEGEND = 1# if 1 plot legend
    if iteration==0: distance_plot = distance_train; figure_suffix = pbestname + '_train'; ylabel = 'train';
    if iteration==1: distance_plot = distance_test; figure_suffix = pbestname + '_test'; ylabel = 'test';
            
    fig, ax = plt.subplots()# model/data distance versus interval2 (interval after the input specifying the condition and before the hold-cue turns off)
    fontsize = 8
    yplot_baseline = yplot_train_cross_condition_average_baseline
    for ii, imodel in enumerate(imodel_set):
        min = np.min(distance_plot[ii,:])
        indices = distance_plot[ii,:] == min# (n_iterations,) array of bool, indices where both conditions are true
        indices = np.arange(indices.size)[indices==True]# indices where condition is True
        imin = indices[0]
        ax.plot(interval2set_new, distance_plot[ii,:], '-', color=eval(f"color{imodel}"), linewidth=2, label=f'model{imodel}, min={min:.4g} when interval2={interval2set_new[imin]}')# label=eval(f"legend_label{imodel}")
        ax.plot(interval2set_new[imin], distance_plot[ii,imin], '.', color=eval(f"color{imodel}"), markersize=20)
    if PLOTBASELINE: xmin, xmax = ax.get_xlim(); ax.plot([xmin, xmax], [yplot_baseline, yplot_baseline], "k--", linewidth=1)# horizontal line indicating the cross condition average baseline
    if PLOTLEGEND: ax.legend(frameon=True, framealpha=1, fontsize=fontsize, loc='best'); figure_suffix = figure_suffix + '_legend'
    ax.set_xlabel('interval2\ninterval after the input specifying the condition and before the hold-cue turns off', fontsize=fontsize)
    ax.set_ylabel(f'{similarityname} {ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.grid(axis='both', color='0.95')
    if VARIANCEEXPLAINED_KEEP is not None: ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {100*VARIANCEEXPLAINED_KEEP}% variance', fontsize=fontsize)
    if n_PCs_KEEP is not None:             ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {n_PCs_KEEP} PCs', fontsize=fontsize)
    if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname} (no PCA)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/models_%svsinterval2_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similarityname,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if n_PCs_KEEP is not None:                                    fig.savefig('%s/models_%svsinterval2_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similarityname,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/models_%svsinterval2_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similarityname,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff


#%%###########################################################################
#    vary interval1set and interval2set and see if this improves model/data similarity
##############################################################################
imodel = i5best_models_train[0]
interval1set_new = interval1set + np.arange(-15, 16)# interval before input specifying the reach condition, if interval1 is 0 then tstartcondition is at the very beginning of the trial (index 0)
interval2set_new = interval2set + np.arange(-15, 16)# interval after the input specifying the condition and before the hold-cue turns off
#interval1set_new = interval1set + np.arange(-25, 26)# interval before input specifying the reach condition, if interval1 is 0 then tstartcondition is at the very beginning of the trial (index 0)
#interval2set_new = interval2set + np.arange(-25, 26)# interval after the input specifying the condition and before the hold-cue turns off
interval1set_new = np.delete(interval1set_new, interval1set_new<0)# remove negative indices, interval1set >= 0


distance_train = -700*np.ones((interval1set_new.size, interval2set_new.size))
distance_test = -700*np.ones((interval1set_new.size, interval2set_new.size))
for ii in range(interval1set_new.size):
    for jj in range(interval2set_new.size):
        data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
        model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
        model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
        n_input = model_info_dictionary['n_input']
        n_output = model_info_dictionary['n_output']
        n_recurrent = model_info_dictionary['n_recurrent']
        model_class = model_info_dictionary['model_class']# model name for forward pass
        task_name_model = model_info_dictionary['task_name']   
            
        #--------------------------
        # generateINandTARGETOUT can be different for each model
        # inputs and target outputs for RNN
        if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
            toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
            OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
            task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set_new[ii][None], 'interval2set':interval2set_new[jj][None], 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
        #--------------------------
        IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
        TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy()
        # IN:        (n_trials_test, n_T_test, n_input) tensor
        # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
        #--------------------------
        
        p = pbest[imodel]# choose RNN parameters to load
        #p = 0# initial parameters for model before training
        checkpoint = torch.load(data_dir + f'/model_parameter_update{p}.pth'); model.load_state_dict(checkpoint['model_state_dict']); 
        activity_noise = 0*torch.randn(n_trials_test, n_T_test, n_recurrent)# (n_trials, n_T, n_recurrent) tensor
        model_input_forwardpass = {'input':IN, 'activity_noise':activity_noise}
        model_output_forwardpass = model(model_input_forwardpass)
        output = model_output_forwardpass['output']; activity = model_output_forwardpass['activity']
        output = output.detach().numpy(); activity = activity.detach().numpy()
        # output:   (n_trials_test, n_T_test, n_output) tensor
        # activity: (n_trials_test, n_T_test, n_recurrent) tensor
        X_neuraldata = neuralforRNN[:,itimekeep_neural,:].copy()# neuralforRNN: n_neurons(161) x n_Tneural(196) x n_reachconditions(27) array, time [-1550:10:400] ms relative to movement onset
        Y = np.transpose(activity[:,itimekeep_RNN,:].copy(), axes=[2, 1, 0])# n_recurrent x n_Tkeep x n_trials_test, permute dimensions of array
        
        distance_train_, distance_test_ = computedistance_crossconditionvalidation(X_neuraldata, Y, VARIANCEEXPLAINED_KEEP, n_PCs_KEEP, similarityname, NORMALIZE_BETWEEN0AND1)# (n_conditions,) arrays, model/data distance across n_conditions iterations of cross-validation
        distance_train[ii,jj] = np.mean(distance_train_)
        distance_test[ii,jj] = np.mean(distance_test_)
    print(f"interval1set_new {ii+1}/{interval1set_new.size}")


for iteration in range(2):
    if iteration==0: data = distance_train; figure_suffix = pbestname + '_train'; ylabel = 'train';
    if iteration==1: data = distance_test; figure_suffix = pbestname + '_test'; ylabel = 'test';
        
    imin = np.where(data == np.min(data))
    
    fig, ax = plt.subplots()# model/data distance versus interval1 and interval2 
    fontsize = 5
    colormap = 'viridis'
    #colormap = cm.get_cmap('cet_gouldian')# The colorcet colormaps are all available through matplotlip.cm.get_cmap by prepending cet_to the colormap name. https://colorcet.holoviz.org/user_guide/index.html
    im = ax.imshow(data, cmap=colormap,  origin='upper')# use colorcet for perceptually uniform colormaps https://colorcet.holoviz.org/getting_started/index.html
    # create an axes on the right side of ax. The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch. https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(similarityname + f' {ylabel} distance', size=fontsize); cbar.ax.tick_params(labelsize=fontsize) 
    ax.set_xlabel('interval2\ninterval after the input specifying the condition and before the hold-cue turns off', fontsize=fontsize)
    ax.set_ylabel('interval1\ninterval before input specifying the condition', fontsize=fontsize)
    ax.set_xticks(np.arange(0,data.shape[1])); ax.set_xticklabels(interval2set_new)
    ax.set_yticks(np.arange(0,data.shape[0])); ax.set_yticklabels(interval1set_new)
    if VARIANCEEXPLAINED_KEEP is not None: ax.set_title(f'model{imodel}, min={np.min(data):.4g} when interval1={interval1set_new[imin[0][0]]}, interval2={interval2set_new[imin[1][0]]}\n{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {100*VARIANCEEXPLAINED_KEEP}% variance', fontsize=fontsize)
    if n_PCs_KEEP is not None:             ax.set_title(f'model{imodel}, min={np.min(data):.4g} when interval1={interval1set_new[imin[0][0]]}, interval2={interval2set_new[imin[1][0]]}\n{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {n_PCs_KEEP} PCs', fontsize=fontsize)
    if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): ax.set_title(f'model{imodel}, min={np.min(data):.4g} when interval1={interval1set_new[imin[0][0]]}, interval2={interval2set_new[imin[1][0]]}\n{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname} (no PCA)', fontsize=fontsize)
    #ax.tick_params(axis='both', labelsize=fontsize)
    ax.tick_params(axis='both', labelsize=2)
    if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/models_%svsinterval1and2_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similarityname,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if n_PCs_KEEP is not None:                                    fig.savefig('%s/models_%svsinterval1and2_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similarityname,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/models_%svsinterval1and2_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similarityname,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
'''






#%%###########################################################################
#                             Figures
##############################################################################
PLOTBASELINE = False # if 1 plot cross condition average baseline
PLOTLEGEND = 0# if 1 plot legend

if VERBOSE: 

    if CROSS_CONDITION_VALIDATION==0: n_iteration = 2; ystd_train = -700# define ystd_train so the code below doesn't break
    if CROSS_CONDITION_VALIDATION==1: n_iteration = 3
    xplot = np.arange(0,n_models)# each model gets one bar showing distance for the best parameters
    for iteration in range(n_iteration):# bar plot showing model/data distance for both train and test data
        if iteration==0: similaritynametoplot = similarityname; figure_suffix = pbestname + '_train'; ylabel = 'train'; yplot = yplot_train; ystd = ystd_train; yplot_baseline = yplot_train_cross_condition_average_baseline# train
        if iteration==2: similaritynametoplot = similarityname; figure_suffix = pbestname + '_test'; ylabel = 'test'; yplot = yplot_test; ystd = ystd_test; yplot_baseline = yplot_test_cross_condition_average_baseline# test
        if iteration==1: similaritynametoplot = similarityname + 'normalizedwithdata'; figure_suffix = pbestname + '_train'; ylabel = 'train'; yplot = np.mean(distance_normalizedwithdata_train,0); ystd = np.std(distance_normalizedwithdata_train,0); yplot_baseline = np.mean(distance_normalizedwithdata_train_cross_condition_average_baseline)# normalized-with-data train distance
        #if iteration==3: similaritynametoplot = similarityname + 'between0and1' + f'on{n_sample_for_distance_normalizedwithdata}neurons'; figure_suffix = pbestname + '_train'; ylabel = 'train'; yplot = np.mean(distance_modeltoneuralsubsampled_between0and1,0); ystd = np.std(distance_modeltoneuralsubsampled_between0and1,0); yplot_baseline = np.mean(distance_crossconditionaverageneuraltoneuralsubsampled_between0and1)# train distance between 0 and 1
    
        fontsize = 12# bar plot showing model/data distance
        fig, ax = plt.subplots()
        for imodel in range(n_models):
            iposition = np.flatnonzero(desiredorder == imodel)# model imodel is plotted at position iposition
            if CROSS_CONDITION_VALIDATION==0: plt.bar(iposition, yplot[imodel],                    color=eval(f"color{imodel}"), ecolor="#F0EBD1", width=0.92, error_kw={'lw': 1}, label=eval(f"legend_label{imodel}"))
            if CROSS_CONDITION_VALIDATION==1: plt.bar(iposition, yplot[imodel], yerr=ystd[imodel], color=eval(f"color{imodel}"), ecolor="#F0EBD1", width=0.92, error_kw={'lw': 1}, label=eval(f"legend_label{imodel}"))
        if PLOTBASELINE: ax.plot([0-0.5, n_models-0.5], [yplot_baseline, yplot_baseline], "k--", linewidth=1); figure_suffix = figure_suffix + '_crossconditionaveragebaseline'; # horizontal line indicating the cross condition average baseline
        plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
        ax.set_xticks(xplot); 
        ax.set_xticklabels(xtick_label_desiredorder, rotation = 45, ha="right", fontsize=1)
        ax.set_ylabel(f'{similaritynametoplot}\n{ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', which='both', length=0)
        if PLOTLEGEND==1: ax.legend(frameon=True, framealpha=1, loc='lower right')
        if PLOTLEGEND==0: figure_suffix = figure_suffix + '_nolegend'
        try: 
            if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_barplot%s_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_barplot%s_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_barplot%s_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        except:
            print('passed')
    
        fontsize = 12# scatter plot showing model/data distance (exactly the same data as the bar plot above)
        fig, ax = plt.subplots()
        for imodel in range(n_models):
            iposition = np.flatnonzero(desiredorder == imodel)# model imodel is plotted at position iposition
            #plt.bar(iposition, yplot[imodel], yerr=ystd[imodel], color=eval(f"color{imodel}"), ecolor="#F0EBD1", width=0.92, error_kw={'lw': 1}, label=eval(f"legend_label{imodel}"))
            ax.plot(iposition, yplot[imodel], '.', markersize=28, color=eval(f"color{imodel}"), label=eval(f"legend_label{imodel}"))
        if PLOTBASELINE: ax.plot([0-0.5, n_models-0.5], [yplot_baseline, yplot_baseline], "k--", linewidth=1);# figure_suffix = figure_suffix + '_crossconditionaveragebaseline'; # horizontal line indicating the cross condition average baseline
        plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
        ax.set_xticks(xplot); 
        ax.set_xticklabels(xtick_label_desiredorder, rotation = 45, ha="right", fontsize=1)
        ax.set_ylabel(f'{similaritynametoplot}\n{ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', which='both', length=0)
        if PLOTLEGEND==1: ax.legend(frameon=True, framealpha=1, loc='lower right')
        #if PLOTLEGEND==0: figure_suffix = figure_suffix + '_nolegend'
        try:
            if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_scatterplot%s_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_scatterplot%s_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_scatterplot%s_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        except:
            print('passed')
    
        fontsize = 12# scatter plot showing model/data distance vs PCA participationa ratio dimensionality
        fig, ax = plt.subplots()
        for imodel in range(n_models):
            ax.plot(dimensionality_store[ipbest[imodel],imodel], yplot[imodel], '.', markersize=28, color=eval(f"color{imodel}"), label=eval(f"legend_label{imodel}"))
        if PLOTBASELINE: minxlim, maxxlim = ax.get_xlim(); ax.plot([minxlim, maxxlim], [yplot_baseline, yplot_baseline], "k--", linewidth=1);# figure_suffix = figure_suffix + '_crossconditionaveragebaseline'; # horizontal line indicating the cross condition average baseline
        ax.set_xlabel('Participation ratio dimenstionality', fontsize=fontsize)
        ax.set_ylabel(f'{similaritynametoplot}\n{ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if PLOTLEGEND==1: ax.legend(frameon=True, framealpha=1, loc='lower right')
        #if PLOTLEGEND==0: figure_suffix = figure_suffix + '_nolegend'
        ax.tick_params(axis='both', labelsize=fontsize)
        try:
            if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_barplot%sVSdimensionality_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_barplot%sVSdimensionality_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_barplot%sVSdimensionality_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        except:
            print('passed')
            
        fig, ax = plt.subplots()# scatter plot showing model/data distance vs normalized accuracy = (1 - normalized error), scatter plot shows only best models at parameters pbest
        fontsize = 13
        for imodel in range(n_models):
            ax.plot(1-distance_errornormalized_store[ipbest[imodel],imodel], yplot[imodel], '.', markersize=28, color=eval(f"color{imodel}"), label=eval(f"legend_label{imodel}")) 
        if PLOTBASELINE: ax.plot([np.min(1-distance_errornormalized_store[ipbest,np.arange(n_models)]), np.max(1-distance_errornormalized_store[ipbest,np.arange(n_models)])], [yplot_baseline, yplot_baseline], "k--", linewidth=1)# horizontal line indicating the cross condition average baseline
        if PLOTLEGEND: ax.legend(frameon=True, framealpha=1, loc='best')   
        ax.set_xlabel('Normalized accuracy\n(1 - normalized error)', fontsize=fontsize)
        ax.set_ylabel(f'{similaritynametoplot}\n{ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        try: 
            if VARIANCEEXPLAINED_KEEP is not None: ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similaritynametoplot}, keep {100*VARIANCEEXPLAINED_KEEP}% variance', fontsize=fontsize)
            if n_PCs_KEEP is not None:            ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similaritynametoplot}, keep {n_PCs_KEEP} PCs', fontsize=fontsize)
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname} (no PCA)', fontsize=fontsize)
        except:
            print('passed')
        ax.tick_params(axis='both', labelsize=fontsize)
        try: 
            if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_scatterplot%svsaccuracy_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_scatterplot%svsaccuracy_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_scatterplot%svsaccuracy_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similaritynametoplot,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        except:
            print('passed')
    #import sys; sys.exit()# stop script at current line
       

if CROSS_CONDITION_VALIDATION==1:
    fontsize = 12# scatter plot showing model/data train vs test distance (exactly the same data as the bar plots above)
    fig, ax = plt.subplots()
    figure_suffix = pbestname + '_trainVStest';
    for imodel in range(n_models):
        ax.plot(yplot_train[imodel], yplot_test[imodel], '.', markersize=28, color=eval(f"color{imodel}"), label=eval(f"legend_label{imodel}"))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    #lineplot = np.linspace(ymin, np.minimum(xmax,ymax), 100)
    lineplot = np.linspace(xmin, xmax, 100)
    ax.plot(lineplot, lineplot, 'k-', linewidth=1)# y = x
    ax.set_xlabel(f'{similarityname} train distance across {n_conditions} conditions', fontsize=fontsize)
    ax.set_ylabel(f'{similarityname}\ntest distance across {n_conditions} conditions', fontsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    if PLOTLEGEND==1: ax.legend(frameon=True, framealpha=1, loc='upper left')
    if PLOTLEGEND==0: figure_suffix = figure_suffix + '_nolegend'
    #ax.axis('equal')
    try: 
        if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_scatterplot%s_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similarityname,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_scatterplot%s_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similarityname,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_scatterplot%s_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similarityname,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    except:
        print('passed')

 

if CROSS_CONDITION_VALIDATION==0: 
    n_iteration = 1
    mean_distance_train = distance_train_store# (pset.size, n_models) array
if CROSS_CONDITION_VALIDATION==1: 
    n_iteration = 2
    mean_distance_train = np.mean(distance_train_store,2)# (pset.size, n_models) array, mean across n_conditions iterations of cross-validation
    mean_distance_test = np.mean(distance_test_store,2)# (pset.size, n_models) array, mean across n_conditions iterations of cross-validation

for iteration in range(n_iteration):# curves showing model/data similarity for both train and test data
    if iteration==0: figure_suffix = 'train'; ylabel = 'train'; yplot = mean_distance_train; yplot_baseline = yplot_train_cross_condition_average_baseline# train
    if iteration==1: figure_suffix = 'test'; ylabel = 'test'; yplot = mean_distance_test; yplot_baseline = yplot_test_cross_condition_average_baseline# test
    if PLOTBASELINE: figure_suffix = figure_suffix + '_crossconditionaveragebaseline'# horizontal line indicating the cross condition average baseline
    if PLOTLEGEND==0: figure_suffix = figure_suffix + '_nolegend'
    
    if VERBOSE: 
        fig, ax = plt.subplots()# model/data distance versus number of parameter updates  
        fontsize = 13
        for imodel in range(n_models):
            ax.plot(pset, yplot[:,imodel], '-', color=eval(f"color{imodel}"), linewidth=2, label=eval(f"legend_label{imodel}"))
            #ax.plot(pset[pset<=5], yplot[pset<=5,imodel], 'o', color=eval(f"color{imodel}"), label='_nolegend_')
        if PLOTBASELINE: ax.plot([np.min(pset), np.max(pset)], [yplot_baseline, yplot_baseline], "k--", linewidth=1)# horizontal line indicating the cross condition average baseline
        if PLOTLEGEND: ax.legend(frameon=True, framealpha=1, loc='best')
        ax.set_xlabel('Number of parameter updates', fontsize=fontsize)
        ax.set_ylabel(f'{similarityname}\n{ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        try: 
            if VARIANCEEXPLAINED_KEEP is not None: ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {100*VARIANCEEXPLAINED_KEEP}% variance', fontsize=fontsize)
            if n_PCs_KEEP is not None:            ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {n_PCs_KEEP} PCs', fontsize=fontsize)
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname} (no PCA)', fontsize=fontsize)
        except:
            print('passed')
        ax.tick_params(axis='both', labelsize=fontsize)
        try: 
            if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_%svsnumberofparameterupdates_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similarityname,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_%svsnumberofparameterupdates_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similarityname,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
            if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_%svsnumberofparameterupdates_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similarityname,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        except:
            print('passed')   
    
    ### 
    fig, ax = plt.subplots() # distance versus normalized accuracy = (1 - normalized error)
    fontsize = 16
    for imodel in range(n_models):
        ax.plot(1-distance_errornormalized_store[:,imodel], yplot[:,imodel], '-', color=eval(f"color{imodel}"), linewidth=2, label=eval(f"legend_label{imodel}"))         
        ax.plot(1-distance_errornormalized_store[pset<=5,imodel], yplot[pset<=5,imodel], 'o', color=eval(f"color{imodel}"), linewidth=2, label='_nolegend_') 
    if similarityname != 'DSA':
        if PLOTBASELINE: ax.plot([np.min(1-distance_errornormalized_store.flatten()), np.max(1-distance_errornormalized_store.flatten())], [yplot_baseline, yplot_baseline], "k--", linewidth=1)# horizontal line indicating the cross condition average baseline
    if PLOTLEGEND: ax.legend(frameon=True, framealpha=1, loc='best')  
    if dataset_name == 'Mante13':
        ax.set_xlabel('(1 - error)', fontsize=fontsize)
    else:
        ax.set_xlabel('Normalized accuracy\n(1 - normalized error)', fontsize=fontsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylabel('Distance to neural data', fontsize=fontsize)
    # ax.set_ylabel(f'{similarityname}\n{ylabel} distance across {n_conditions} conditions', fontsize=fontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);# ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    # if VARIANCEEXPLAINED_KEEP is not None: ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {100*VARIANCEEXPLAINED_KEEP}% variance', fontsize=fontsize)
    # if n_PCs_KEEP is not None:            ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname}, keep {n_PCs_KEEP} PCs', fontsize=fontsize)
    # if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): ax.set_title(f'{n_conditions} conditions, {n_Tkeep} timesteps for {similarityname} (no PCA)', fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize)
    fig.savefig('%s/main_%svsaccuracy.pdf'%(figure_dir,similarityname), bbox_inches='tight')
    # if VARIANCEEXPLAINED_KEEP is not None:                        fig.savefig('%s/main_%svsaccuracy_nTkeep%g_variance%g_%s.pdf'%(figure_dir,similarityname,n_Tkeep,VARIANCEEXPLAINED_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    # if n_PCs_KEEP is not None:                                    fig.savefig('%s/main_%svsaccuracy_nTkeep%g_%gPCs_%s.pdf'%(figure_dir,similarityname,n_Tkeep,n_PCs_KEEP,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
    # if (VARIANCEEXPLAINED_KEEP is None) and (n_PCs_KEEP is None): fig.savefig('%s/main_%svsaccuracy_nTkeep%g_noPCA_%s.pdf'%(figure_dir,similarityname,n_Tkeep,figure_suffix), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff 




#import sys; sys.exit()# stop script at current line  

#%%############################################################################
# check that all models take the same inputs before running the following code
# it's ok for models to have different target outputs because the following code clusters models based on the forward pass (for this code don't use random intervals in the input)
# this check is likely only relevant if the task_name is not all the same for the models
###############################################################################
imodel = 0# get IN for first model
data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
n_input = model_info_dictionary['n_input']
n_output = model_info_dictionary['n_output']
n_recurrent = model_info_dictionary['n_recurrent']
model_class = model_info_dictionary['model_class']# model name for forward pass
task_name_model = model_info_dictionary['task_name']   
#--------------------------
# generateINandTARGETOUT can be different for each model
# inputs and target outputs for RNN
if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
    toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
    OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
    task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}

    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
    TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy();
# IN:        (n_trials_test, n_T_test, n_input) tensor
# TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
#--------------------------
INold = IN.detach().numpy().copy()# IN for first model
#------------------------------------------------------------------------------
for imodel in range(n_models):      
    data_dir = eval(f"data_dir{imodel}")# use eval so data_dir is the value of data_dir0 and not the string 'data_dir0'
    model = torch.load(f'{data_dir}/model.pth')# torch.save(model, f'{figure_dir}/model.pth')# save entire model, not just model parameters
    model_info_dictionary = np.load(f'{data_dir}/model_info_dictionary.npy', allow_pickle='TRUE').item()
    n_input = model_info_dictionary['n_input']
    n_output = model_info_dictionary['n_output']
    n_recurrent = model_info_dictionary['n_recurrent']
    model_class = model_info_dictionary['model_class']# model name for forward pass
    task_name_model = model_info_dictionary['task_name']   
    #--------------------------
    # generateINandTARGETOUT can be different for each model
    # inputs and target outputs for RNN
    if task_name_model[0:12]=='Sussillo2015':# this assumes task_name_model starts with the phrase Sussillo2015
        toffsetoutput = model_info_dictionary['toffsetoutput']# number of timesteps EMG output is offset, no change if toffsetEMG=0, if toffsetEMG is negative then target EMG outuput is earlier in the trial
        OUTPUTONEHOT=model_info_dictionary['OUTPUTONEHOT']; OUTPUTXYHANDPOSITION=model_info_dictionary['OUTPUTXYHANDPOSITION']; OUTPUTXYHANDVELOCITY=model_info_dictionary['OUTPUTXYHANDVELOCITY']; OUTPUTEMG=model_info_dictionary['OUTPUTEMG']# OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG can all be 1, in this case the target output is a concatenation of each output in the order OUTPUTONEHOT, OUTPUTXYHANDPOSITION, OUTPUTXYHANDVELOCITY, OUTPUTEMG
        task_input_dict = {'task_name':task_name_model, 'n_input':n_input, 'n_output':n_output, 'n_T':n_T_test, 'n_trials':n_trials_test, 'interval1set':interval1set, 'interval2set':interval2set, 'toffsetoutput':toffsetoutput, 'OUTPUTONEHOT':OUTPUTONEHOT, 'OUTPUTXYHANDPOSITION':OUTPUTXYHANDPOSITION, 'OUTPUTXYHANDVELOCITY':OUTPUTXYHANDVELOCITY, 'OUTPUTEMG':OUTPUTEMG}
        
        IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT(task_input_dict)
        TARGETOUT = TARGETOUT.detach().numpy(); output_mask = output_mask.detach().numpy();
    # IN:        (n_trials_test, n_T_test, n_input) tensor
    # TARGETOUT: (n_trials_test, n_T_test, n_output) tensor
    #--------------------------
    if (INold.shape != IN.detach().numpy().shape) or (not np.allclose(INold, IN.detach().numpy())):# if the arrays don't have the same shape or are not close then stop script
        print("stopped script before clustering visualizations because all models don't have the same inputs")
        import sys; sys.exit()# stop script at current line 
    INold = IN.detach().numpy().copy()
    
    



