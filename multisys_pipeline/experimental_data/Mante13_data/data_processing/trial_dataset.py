from pathlib import Path
from tempfile import mkdtemp
import numpy as np

class TrialDataset:
    """
    Trial based dataset with the same number of timesteps for each trial
    """
    def __init__(self, activity, trial_info, regions=None, dt=None, **kwargs):
        self.activity = np.array(activity)  # time x trial x neuron
        self.n_trials = self.activity.shape[0]
        self.n_timesteps = self.activity.shape[1]
        self.n_neurons = self.activity.shape[2]
        self.trial_info = np.array(trial_info)
        self.regions = regions if regions is not None else [""]*self.n_neurons
        self.dt = dt
        if "model_path" in kwargs: # TODO KZ: simplify, right now this is needed for analysis
            self.model_path = kwargs.pop("model_path")
        if "config" in kwargs: # TODO KZ: simplify, right now this is needed for analysis
            self.config = kwargs.pop("config")
        self.data = kwargs  # additional data

        assert len(self.activity.shape) == 3, self.activity.shape
        assert len(self.trial_info) == self.n_trials, \
            "Expected {} and {} to be equal".format(len(self.trial_info), self.n_trials)
        # assert len(self.trials_timing) == self.activity.shape[1]
        assert len(self.regions) == self.n_neurons, \
            "Expected the number of regions {} to match the number of neurons {}".format(len(self.regions),
                                                                                         self.n_neurons)

    def get_neuron_indices(self, region=None):
        if region is None:
            return np.arange(self.n_neurons)
        elif isinstance(region, str):
            return np.where(self.regions == region)[0]
        elif isinstance(region, (list, np.ndarray)):
            bool_mask = np.logical_or.reduce([self.regions == r for r in region])
            return np.where(bool_mask)[0]
        else:
            raise TypeError

    def get_trial_indices(self, trial=None):
        if trial is None:
            return np.arange(self.n_trials)
        elif isinstance(trial, (int, np.integer)):
            return [trial]
        elif isinstance(trial, (list, np.ndarray)):
            if isinstance(trial, np.ndarray):
                assert len(trial.shape) == 1
            return trial
        else:
            raise TypeError("Unexpected type {} for trial".format(type(trial)))

    def get_condition_trials(self, condition=None):
        if condition is None:
            return np.arange(self.n_trials)
        elif isinstance(condition, dict):
            mask = np.ones(self.n_trials)
            for cond_key, cond_val in condition.items():
                info_val = [trial_info[cond_key] for trial_info in self.trial_info]
                if isinstance(cond_val, (np.ndarray, list)):
                    # the value of the condition is a list (e.g. one hot task rule)
                    cond_mask = (info_val == np.array(cond_val)).all(axis=1)
                else:
                    if np.array(cond_val).dtype == float:
                        # allow some negligible differences
                        cond_mask = np.abs(info_val - np.array(cond_val)) < 1e-8
                    else:
                        cond_mask = info_val == np.array(cond_val)
                mask = np.logical_and(mask, cond_mask)
            return np.where(mask)[0]
        else:
            raise TypeError("Unexpected type for {}".format(condition))

    def set_activity(self, activity):
        self.activity = activity

    def get_activity(self, trial=None, region=None, condition=None, batch_first=True):
        """
        Args:
            trial: None, int, or list/ndarray of int
            region: None (equivalent to all the regions), str,
                list/ndarray of str: activity within all the given regions
            condition: None, or condition dict. If not None, trial is ignored

        Returns:
            activity: ndarray of size n_timesteps x n_trials x n_neurons
        """
        if condition is not None:
            act = self.get_activity(trial=self.get_condition_trials(condition), region=region)
        else:
            act = self.activity[self.get_trial_indices(trial)]
            act = act[:, :, self.get_neuron_indices(region)]
        if not batch_first:
            act = act.transpose((1, 0, 2))
        return act

    def get_activity_by_condition(self, conditions, region=None, average=False):
        """
        Sort the trials by condition
        Args:
            conditions: list of condition dict
            region: str
            average: if true, average trials of the same condition

        Returns:
            act_by_cond: list of length len(conditions) of arrays of size either
                n_steps x n_trials_in_cond x n_neurons if average, otherwise n_steps x n_neurons
        """
        act_by_cond = []
        for c in conditions:
            # n_steps x n_trials_in_cond x n_neurons
            act = self.get_activity(condition=c, region=region)
            n_trials, n_steps, n_neurons = act.shape
            assert n_steps > 0
            if average:
                act = act.mean(axis=0)
            act_by_cond.append(act)
        return act_by_cond

    def get_trial_info(self, trial=None, label=None):
        # return a copy to prevent modifications from outside this class
        trials_info = self.trial_info[self.get_trial_indices(trial)]
        if isinstance(label, str):
            return np.array([trial_info[label] for trial_info in trials_info])
        else:
            return [info.copy() for info in trials_info]

    def set_trial_info(self, trial_info):
        self.trial_info = np.array(trial_info)

    def set_regions(self, regions):
        self.regions = regions

    def get_regions(self):
        return self.regions

    def get_region_names(self):
        _, idx = np.unique(self.regions, return_index=True)
        return self.regions[np.sort(idx)]

    def get_conditions(self):
        trial_info = self.get_trial_info()
        conditions = [dict(s) for s in set(frozenset(d.items()) for d in trial_info)]
        return conditions

    def get(self, label):
        if label in self.data:
            return self.data[label]
        else:
            raise KeyError

    def get_sensory_input(self, trial=None):
        if 'sensory_input' in self.data:
            return self.data['sensory_input'][self.get_trial_indices(trial)]
        else:
            raise KeyError

    def get_response(self, trial=None):
        if 'response' in self.data:
            return self.data['response'][self.get_trial_indices(trial)]
        else:
            raise KeyError