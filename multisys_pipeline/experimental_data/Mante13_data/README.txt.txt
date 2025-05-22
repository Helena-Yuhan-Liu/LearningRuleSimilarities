./neuralforRNN.npy
This file contains preprocessed data based on the publicly available Mante et al. (2013) dataset. The tensor is shaped as (n_neurons, n_steps, n_conditions), where spiking activity has been converted to firing rates and averaged across trials within each condition. 

./data_synthetic.pkl
This file contains synthetic data for RNN training, generated via NeuroGym, formatted as (n_conditions, n_steps, n_neurons).

./data_processing/
This folder contains code snippets for data preprocessing and NeuroGym-based synthetic data generation. Parts of the preprocessing were modified from code written by Nathan Cloos and Katheryn Zhou, originally based on scripts provided on Valerio Mante’s website (https://www.ini.uzh.ch/en/research/groups/mante/data.html). At this time, we include illustrative code snippets for neural data preprocessing and synthetic data generation, but not the full pipeline. These are intended for reference for preprocessing details rather than direct execution.

 
