from __future__ import annotations

from copy import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d

from multisys.neural_datasets.trial_dataset import TrialDataset

# TODO: transform_ to modify the dataset in place

def downsample(activity, n_samples):
    # activity: ... x n_timesteps x ...
    n_steps = activity.shape[1]
    assert n_samples <= n_steps

    window = int(n_steps / n_samples)
    resampled_act = np.zeros((activity.shape[0], n_samples, *activity.shape[2:]))
    for k in range(n_samples):
        # average within the temporal window
        resampled_act[:, k] = activity[:, k * window:(k + 1) * window].mean(axis=1) # take the mean across all timesteps during this dt!
    return resampled_act


# def bin_spike_times(spike_times, dt, bin_range=None):
#     """
#     Args:
#         spike_times: list (neurons) of lists (spikes times for each neuron)
#         dt: bin width
#     Returns:
#         n_neurons x n_bins np.ndarray of binned spikes
#     """
#     # TODO: why 0 is the min?
#     bin_range = (0, max([times[-1] for times in spike_times])) if bin_range is None else bin_range
#     n_bins = np.ceil(bin_range[1]/dt).astype(int)

#     binned_spikes = []  # list of binned spikes for each neuron
#     for neuron_spike_times in spike_times:
#         binned_spikes.append(
#             np.histogram(neuron_spike_times, n_bins, bin_range)[0]
#         )
#     binned_spikes = np.array(binned_spikes)
#     return binned_spikes


# def moving_average(signal, n=3, padding=None):
#     if padding:
#         # repeat the first element of signal so that the output as the same shape as the input
#         # signal = np.concatenate([np.repeat(signal[0], n-1, axis=0), signal], axis=0)
#         signal = np.concatenate([np.zeros((n-1, *signal.shape[1:])), signal], axis=0)
#     # time is the first dim of signal
#     ret = np.cumsum(signal, axis=0, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


# class TimeRestriction:
#     def __init__(self, start_idx=None, stop_idx=None, t=None, t_min=None, t_max=None):
#         self.start_idx = start_idx
#         self.stop_idx = stop_idx

#         if t is not None:
#             assert t_min is not None
#             assert t_max is not None
#             # (t, t_min, t_max) overwrites start_idx, stop_idx
#             self.start_idx = np.argwhere(t >= t_min)[0][0]
#             self.stop_idx = np.argwhere(t > t_max)[0][0]

#     def transform(self, inp: np.ndarray | TrialDataset, inplace=True):
#         if isinstance(inp, np.ndarray):
#             return inp[self.start_idx:self.stop_idx]
#         else:
#             inp.set_activity(inp.get_activity()[self.start_idx:self.stop_idx])
#             if hasattr(inp, 't'):
#                 inp.t = inp.t[self.start_idx:self.stop_idx]
#             return inp


class DatasetAggregation:
    def __init__(self, axis, axis_label="neuron"):
        self.axis = axis
        self.axis_label = axis_label

    def transform(self, *datasets):
        activity = np.concatenate([dataset.activity for dataset in datasets], axis=self.axis)

        if self.axis_label == "neuron":
            # concatenate along the neuron axis
            regions = np.concatenate([dataset.regions for dataset in datasets])
            # TODO: assert that the trial info is the same and are aligned for all datasets
            trial_info = datasets[0].trial_info
            data = datasets[0].data

        elif self.axis_label == "trial":
            # concatenate trial info
            trial_info = np.concatenate([dataset.get_trial_info() for dataset in datasets])
            # assume that trial axis = 1 for all arrays in dataset.data
            # TODO: make it more general using named axes
            data = {
                k: np.concatenate([dataset.get(k) for dataset in datasets], axis=1)
                for k in datasets[0].data.keys()
            }
            regions = datasets[0].regions
        else:
            raise ValueError

        agg_dataset = TrialDataset(activity, trial_info, regions, **data)

        if hasattr(datasets[0], 'dt'):
            agg_dataset.dt = datasets[0].dt
        return agg_dataset


class DatasetSplit:
    def __init__(self, ratio):
        # self.axis = axis
        self.ratio = ratio

    def fit(self, dataset: TrialDataset):
        pass

    def transform(self, dataset: TrialDataset):
        # compute self.trials1, self.trials2
        self.trials1 = []
        self.trials2 = []
        for cond in dataset.get_conditions():
            # n_trials can be equal to 1
            cond_trials = dataset.get_condition_trials(condition=cond)
            n_trials = len(cond_trials)
            np.random.shuffle(cond_trials)
            cond_trials1 = cond_trials[:round(n_trials*self.ratio)]
            cond_trials2 = cond_trials[round(n_trials*self.ratio):]

            self.trials1.extend(cond_trials1)
            self.trials2.extend(cond_trials2)

    # def transform(self, dataset):
        # print(self.trials1)
        # print(self.trials2)
        # split according to self.trials1, self.trials2
        dataset1 = TrialDataset(dataset.activity[:, self.trials1], dataset.trial_info[self.trials1], dataset.regions)
        dataset2 = TrialDataset(dataset.activity[:, self.trials2], dataset.trial_info[self.trials2], dataset.regions)
        return dataset1, dataset2


# TODO: average within conditions != condition average baseline that we plot
class ConditionAverage:
    def __init__(self, cond_fields: list[str] | str = "all", sort=True):
        self.cond_fields = [cond_fields] if isinstance(cond_fields, str) and cond_fields != 'all' else cond_fields
        self.sort = sort  # sort conditions

    # TODO: move that to utils
    def _dict_list_unique(self, dict_list):
        return [dict(s) for s in set(frozenset(d.items()) for d in dict_list)]

    def transform(self, dataset):
        dataset = copy(dataset)

        if self.cond_fields == 'all':
            conditions = self._dict_list_unique(dataset.get_trial_info())
            self.cond_fields = list(conditions[0].keys())
        else:
            # take only the trial info for the keys in cond_fields
            conditions = [{k: info[k] for k in self.cond_fields} for info in dataset.get_trial_info()]
            conditions = self._dict_list_unique(conditions)
        if self.sort:
            # sort the conditions based on the values ordered by cond_fields
            conditions = sorted(conditions, key=lambda x: tuple(x[k] for k in self.cond_fields))

        cond_activity = np.zeros((dataset.get_activity().shape[0], len(conditions), dataset.get_activity().shape[-1]))
        cond_trial_info = []
        for i, c in enumerate(conditions):
            # get the trials satisfying the condition c
            cond_trials = dataset.get_condition_trials(c)
            act = dataset.get_activity(trial=cond_trials)  # n_steps x n_trials_in_cond x n_neurons
            assert act.shape[1] > 0

            cond_activity[:, i] = act.mean(axis=1)
            # take the first trial in cond_trials for the trial_info of the condition
            cond_trial_info.append(dataset.get_trial_info(trial=cond_trials[0])[0])

        dataset.set_activity(cond_activity)
        dataset.set_trial_info(cond_trial_info)
        assert len(dataset.get_trial_info()) == dataset.activity.shape[1]
        return dataset


# class TemporalSmoother:
#     def __init__(self, std, dt, filter='gaussian'):
#         self.std = std
#         self.dt = dt
#         self.filter = filter

#     def transform(self, dataset):
#         dataset = copy(dataset)
#         activity = dataset.get_activity()
#         if self.filter == 'gaussian':
#             dataset.activity = gaussian_filter1d(activity, self.std / self.dt, axis=0)
#         else:
#             raise NotImplementedError
#         return dataset


# class Downsampler:
#     def __init__(self, dt):
#         self.dt = dt

#     def transform(self, dataset):
#         dataset = copy(dataset)
#         activity = dataset.get_activity()
#         n_steps = activity.shape[0]

#         self.n_samples = int(n_steps*dataset.dt/self.dt)
#         assert self.n_samples <= n_steps

#         window = int(n_steps / self.n_samples)
#         resampled_act = np.zeros((self.n_samples, *activity.shape[1:]))
#         for k in range(self.n_samples):
#             # average within the temporal window
#             resampled_act[k] = activity[k * window:(k + 1) * window].mean(axis=0)

#         dataset.set_activity(resampled_act)
#         dataset.dt = self.dt

#         # update time TODO: store clock_time in trial_info or in another attribute?
#         if 'time' in dataset.get_trial_info()[0]:
#             trial_info = dataset.get_trial_info()
#             for i, info in enumerate(trial_info):
#                 clock_time = dataset.get_trial_info(trial=i, label='time')[0]
#                 info['time'] = np.array([clock_time[k*window] for k in range(self.n_samples)])
#             dataset.set_trial_info(trial_info)

#         return dataset


# def restrict_dataset_region(dataset, region):
#     dataset = copy(dataset)
#     dataset.activity = dataset.get_activity(region=region)
#     dataset.regions = dataset.regions[np.isin(dataset.regions, region)]
#     return dataset


# class RegionRestricter:
#     def __init__(self, region):
#         self.region = region

#     def transform(self, dataset, inplace=False):
#         dataset = copy(dataset) if not inplace else dataset # don't want to change the original dataset TODO: better way to do that
#         dataset.activity = dataset.get_activity(region=self.region)
#         dataset.regions = dataset.regions[np.isin(dataset.regions, self.region)]
#         return dataset