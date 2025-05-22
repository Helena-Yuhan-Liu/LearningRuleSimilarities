from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as spio
import numpy as np
from urllib.request import urlopen
from zipfile import ZipFile

from multisys.neural_datasets.trial_dataset import TrialDataset
from multisys.neural_datasets.preprocessing import ConditionAverage, DatasetAggregation, DatasetSplit, downsample
from multisys.utils.compute_fns import compute_cross_condition_avg
# from multisys.utils import download_and_unzip

# file_path = Path(__file__).parent.resolve()
# DOWNLOAD_DIR = file_path / Path("data/mante13")
DOWNLOAD_DIR = Path(__file__).parent.resolve()
DATA_DIR = DOWNLOAD_DIR / Path("PFC data/PFC data 1")

def download_and_unzip(url, extract_to='.'):
    # https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
    print("Downloading (this may take a few minutes): {}".format(url))
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    print(f'FINISHED DOWNLOADING TO {extract_to}')

def download(save_path):
    url = "https://www.ini.uzh.ch/dam/jcr:ca4213cf-1692-4c3d-8aeb-d4c5081d2fd1/PFC%20data.zip"
    download_and_unzip(url, extract_to=save_path)

MANTE13_DT = 15 # ms (similar results with 50ms too)

class Mante13Dataset(TrialDataset):
    def __init__(self, dt, data_dir: str | Path = None):
        print('Initializing Mante13 neural dataset, this will take a few seconds')
        stim_onset = 350 # ms
        # information about the start and stop times from this brain dataset
        self.record_config = {
        "start_time": stim_onset, # 0, 
        "stop_time": stim_onset+750 #350+750+300+300
        }
        self.dt = dt
        self.recording_start_index = int(self.record_config['start_time']/self.dt)
        self.recording_stop_index = int(self.record_config['stop_time']/self.dt)

        # download the data if data_dir doesn't exist
        data_dir = DATA_DIR if data_dir is None else data_dir
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        if not data_dir.exists():
            download(DOWNLOAD_DIR)

        activity, trial_info, init_unit_dts, dt, regions = self.process_data(Path(data_dir))
        activity = activity.transpose(1, 0, 2) # (timesteps, trial, neurons) --> (trials, timesteps, neurons)
        self.init_unit_dts = init_unit_dts
        super().__init__(activity, trial_info, regions=regions, dt=self.dt)
        self.cross_condition_avg = {}
        self.cross_condition_avg['PFC'] = compute_cross_condition_avg(neuraldata=activity, normalize=True)

        num_regions = len(self.get_region_names())
        cmap = plt.get_cmap('gnuplot')
        self.region_colormap = [cmap(i) for i in np.linspace(0, 1, num_regions)]
        print('Finished initializing Mante13 neural dataset')

    def _process_unit(self, unit):
        """
        Args:
            unit: struct load from .mat file
        Returns:
            dt: sampling timestep
            cond_avg_activity: n_conditions x n_steps array, condition-averaged activity of the unit
            trials_info: list of dict containing the condition info for each row of activity
        """
        dt = (unit['time'][1] - unit['time'][0])*1000 # ms
        init_unit_dt = dt
        task_var = unit['task_variable']
        activity = unit['response'] # n_total_trials x n_recording_steps = 2043, 751

        # mante_var_names = ['stim_dir', 'stim_col', 'context']
        # standardized_names = ['coh_mod1', 'coh_mod2', 'context']
        mante_var_names = ['targ_dir', 'stim_dir', 'stim_col2dir', 'context']
        standardized_names = ['gt_choice', 'coh_mod1', 'coh_mod2', 'context']

        # cond_keys = ['stim_dir', 'stim_col2dir', 'context']

        trials_by_condition = defaultdict(list)# trials sorted by condition
        # loop through all trials
        for trial in range(task_var['stim_trial'][-1]):
            condition = tuple([task_var[name][trial] for name in mante_var_names])
            # condition = tuple([task_var[name][trial] for name in cond_keys])
            trials_by_condition[condition].append(trial)
        conditions, cond_trials = list(trials_by_condition.keys()), list(trials_by_condition.values())
        # sort conditions to have the same order for all units
        k = np.empty(len(trials_by_condition), dtype=object)
        k[:] = conditions # array of tuple
        indices = np.argsort(k, axis=0)

        cond_avg_activity = []  # n_conditions x n_steps
        trials_info = []
        for idx in indices:
            cond, trials = conditions[idx], cond_trials[idx]
            # print(cond, len(trials), task_var['correct'][trials])
            if not np.all(task_var['correct'][trials]):
                # skip incorrect trials (the task var 'correct' is the same for all trials with the same condition)
                continue
            cond_avg_activity.append(activity[trials].mean(axis=0)) # average activity across trials with the same condition. shape = num_units. 

            trial_info = {k: v for k, v in zip(standardized_names, cond)}
            trials_info.append(trial_info)

        cond_avg_activity = np.array(cond_avg_activity) # (conditions, recording_timesteps)

        # resampling
        n_samples = int(cond_avg_activity.shape[-1]*dt/self.dt)
        cond_avg_activity = downsample(cond_avg_activity, n_samples) # downsample, take mean across each window where self.dt = 50 ms and recordings 
        # originally dt = 1ms

        # print(cond_avg_activity.shape)
        return init_unit_dt, self.dt, cond_avg_activity, trials_info

    def process_data(self, data_dir):
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)

        # TODO not the same levels of coh for all units!
        # for file_path in data_dir.iterdir():
            # mat = spio.loadmat(str(file_path), simplify_cells=True)
            # unit = mat['unit']
            # task_var = unit['task_variable']
            # print(np.unique(task_var['stim_dir']), np.unique(task_var['stim_col']))

        # use one session as the reference for the trial info (sessions can have different levels of coherence but map them to the same reference)
        ref_file = data_dir / "ar090313_1_a1_Vstim_100_850_ms.mat"
        ref_mat = spio.loadmat(str(ref_file), simplify_cells=True)
        trials_info = self._process_unit(ref_mat['unit'])[-1] # ref_mat['unit']['response] = (2043, 751) = (trials, recording_timesteps)

        dt = None
        activity = []
        init_unit_dts = [] # list containing initial unit dts, before resampling recordings to match dt
        for i, file_path in enumerate(data_dir.iterdir()):
            mat = spio.loadmat(str(file_path), simplify_cells=True)
            unit = mat['unit']
            init_unit_dt, unit_dt, unit_activity, unit_trials_info = self._process_unit(unit)
            init_unit_dts.append(init_unit_dt)

            #TODO
            if unit_activity.shape[0] != 72:
                continue
            activity.append(unit_activity)
            dt = unit_dt if dt is None else dt

            # assume all units have the same dt
            assert dt == unit_dt
            #TODO not the same trials_info for all units...
            # assert trials_info == unit_trials_info

        activity = np.transpose(np.array(activity), (2, 1, 0)) # (727=neurons, 72=trials, 50=timesteps) --> n_steps x n_trials x n_neurons

        # correct_choice = []
        # postprocessing
        # NOTE KZ: updated trial info representation
        for trial_info in trials_info:
            # TODO KZ: MAKE SURE THIS IS CORRECT!!
            trial_info['coh_motion'] = trial_info.pop('coh_mod1')
            trial_info['coh_color'] = trial_info.pop('coh_mod2')
            trial_info['ground_truth'] = trial_info.pop('gt_choice')
            # gt = 1 --> coh is negative, gt = 2 --> coh is positive
            if trial_info['ground_truth'] == -1:
                trial_info['ground_truth'] = 1
            elif trial_info['ground_truth'] == 1:
                trial_info['ground_truth'] = 2

        # mante_timing = {
        #     'fixation': 350,
        #     'stimulus': 330,
        #     'decision': 300
        # }
        # data_dict = {
        #     "dt": dt,
        #     "activity": activity,
        #     "trial_info": trials_info,
        #     "trial_timing": [mante_timing]*len(trials_info),
        #     "region": ["PFC"]*activity.shape[-1],
        #     # "decision": correct_choice # ground truth instead of decision since trial-averaged
        #     "decision": [trial_info['gt_choice'] for trial_info in trials_info] # ground truth instead of decision since trial-averaged
        # }
        # return data_dict
        regions = np.array(["PFC"]*activity.shape[-1])
        return activity, trials_info, init_unit_dts, dt, regions
    
MANTE13_DATASET_STANDARD = Mante13Dataset(dt = MANTE13_DT)

class Mante13DatasetV2(TrialDataset):
    """
    This version of the Mante dataset is slower to load but allows to cross validate across trials
    """
    var_mapping = {"targ_dir": "gt_choice", "stim_dir": "coh_mod1", "stim_col2dir": "coh_mod2", "context": "context"}
    region = "PFC"

    def __init__(self, data_dir: str | Path = None, dt: int = 50, keep_correct_trials: bool = True,
                 neuron_join: str = 'inner'):
        """
        Args:
             dt: ms
             keep_correct_trials: keep only correct trials be default
             neuron_join: how to concat unit datasets, only 'inner' is implemented (discard neurons without all the
                conditions)
             data_dir: directory containing the matlab data files
        """
        self.dt = dt
        self.keep_correct_trials = keep_correct_trials
        self.neuron_join = neuron_join

        # download the data if data_dir doesn't exist
        data_dir = DATA_DIR if data_dir is None else data_dir
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        if not data_dir.exists():
            download(DOWNLOAD_DIR)

        activity, trial_info, regions = self.process_data(Path(data_dir))
        super().__init__(activity, trial_info, regions=regions, dt=dt)

    def process_data(self, data_dir):
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)

        # list of unit datasets (datasets with a single neuron and a variable number of trials)
        datasets = []
        # loop over the data files in data_dir
        for k, file_path in enumerate(data_dir.iterdir()):
            # load matlab data
            mat = spio.loadmat(str(file_path), simplify_cells=True)
            unit = mat['unit']

            unit_dt = (unit['time'][1] - unit['time'][0]) * 1000  # ms
            task_var = unit['task_variable']
            unit_activity = unit["response"]  # trial x time

            # resampling
            n_samples = int(unit_activity.shape[-1] * unit_dt / self.dt)
            unit_activity = downsample(unit_activity, n_samples)

            unit_activity = unit_activity.T[:, :, None]  # time x trial x neuron

            # make a list of trial info
            n_trials = unit_activity.shape[1]
            unit_trial_info = []
            for i in range(n_trials):
                info = {}
                for k, v in self.var_mapping.items():
                    info[v] = task_var[k][i]
                unit_trial_info.append(info)

            if self.keep_correct_trials:
                # keep correct trials only
                correct_trials = task_var["correct"]
                correct_trials = np.squeeze(np.nonzero(
                    correct_trials))  # simple bool indexing [:, correct_trials] not working https://stackoverflow.com/questions/7820809/understanding-weird-boolean-2d-array-indexing-behavior-in-numpy
                unit_activity = unit_activity[:, correct_trials]
                unit_trial_info = np.array(unit_trial_info)[correct_trials]

            unit_dataset = TrialDataset(unit_activity, unit_trial_info, [self.region])
            datasets.append(unit_dataset)

        self.unit_datasets = deepcopy(datasets)

        dataset = self.process_unit_datasets(datasets)
        return dataset.activity, dataset.trial_info, dataset.regions

    def process_unit_datasets(self, datasets):
        # average conditions, sort=True ensures that the trial conditions are aligned before the aggregation!
        cond_avg = ConditionAverage(sort=True)
        cond_avg_datasets = []
        for i, dataset in enumerate(datasets):
            dataset = cond_avg.transform(dataset)
            dataset.dt = self.dt

            if self.neuron_join == 'inner':
                # discard neurons without all the conditions
                if dataset.activity.shape[1] < 72:
                    continue
            else:
                raise NotImplementedError("Only neuron_join = 'inner' is implemented")
            cond_avg_datasets.append(dataset)
        datasets = cond_avg_datasets

        dataset = DatasetAggregation(axis=-1).transform(*datasets)
        dataset.dt = self.dt
        trial_info = dataset.trial_info
        # choice = 1, 2 instead of 1, -1 (to make it the same as the model datasets)
        for info in trial_info:
            if info['gt_choice'] == -1:
                info['gt_choice'] = 1
            else:
                info['gt_choice'] = 2
        dataset.trial_info = trial_info
        return dataset

    def trial_split(self, ratio=0.5):
        # TODO: random seed
        # TODO: return Mante13Dataset instead of TrialDataset?
        assert hasattr(self, 'unit_datasets')
        split = DatasetSplit(ratio=ratio)
        datasets1 = []
        datasets2 = []
        for dataset in self.unit_datasets:
            split.fit(dataset)
            dataset1, dataset2 = split.transform(dataset)
            dataset1.dt = dataset2.dt = self.dt
            datasets1.append(dataset1)
            datasets2.append(dataset2)

        dataset1 = self.process_unit_datasets(datasets1)
        dataset2 = self.process_unit_datasets(datasets2)
        return dataset1, dataset2