# LearningRuleSimilarities

This repository implements RNN models trained with biologically plausible learning rules (e.g., e-prop) to study how well they reproduce neural activity observed in primate datasets. We benchmark neural similarity against BPTT-trained models using Procrustes and other similarity measures. Code and analysis follow the setup described in our paper [1]. 

Please be advised that the included code supports the training and analysis pipeline for the publicly available Mante 2013 dataset (accessible at https://www.ini.uzh.ch/en/research/groups/mante/data.html) and Hatsopoulos2007 dataset (https://datadryad.org/dataset/doi:10.5061/dryad.xsj3tx9cm). The Sussillo 2015 data is not included as we do not have permission to redistribute it. The pipeline utilizes preprocessed neural data, where spikes are converted to rate data and each trial represents a condition average. For RNN training, synthetic data is generated with the shape of the input tensor being (trials, time, number of inputs); the shape of the desired target output tensor is (trials, time, number of outputs). 

## Usage 

**Step 1** - Setup:
   Run the command `python3 setup.py`.
   Refer to the comments in setup.py for troubleshooting if any issues occur.

**Step 2** - Train a Model:
   Execute `python3 multisys_pipeline/main_train_model.py`.
   You can modify the command line arguments to specify hyperparameters and select the learning rule (e.g., `learning_mode=0` for BPTT and `learning_mode=1` for e-prop).

   Example: 

   `python3 multisys_pipeline/main_train_model.py --learning_mode=0 --random_seed=1`

   `python3 multisys_pipeline/main_train_model.py --learning_mode=0 --random_seed=2`

   `python3 multisys_pipeline/main_train_model.py --learning_mode=1 --random_seed=1`

   `python3 multisys_pipeline/main_train_model.py --learning_mode=1 --random_seed=2`

**Step 3** - Run Analysis:
   Execute `python3 multisys_pipeline/main_analyze_models.py`.
   Modify the block of code below `line 72` to specify which models to include in the analysis.

## Credits 

Helena Liu and Chris Cueva both contributed to the code development. We thank Nathan Cloos and Katheryn Zhou for their preprocessing code for the Mante 2013 dataset, which builds on Valerio Mante�s original scripts. We are also grateful to Mante, Hatsopoulos, and Churchland for sharing their neural datasets. 

## Reference 

[1] Liu Y.H., Yang R.G., Cueva C.J., �Can Biologically Plausible Temporal Credit Assignment Rules Match BPTT for Neural Similarity? E-prop as an Example�, International Conference on Machine Learning (ICML), 2025 (to appear). 
