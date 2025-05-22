"""Mante 2013 simplified task."""
import copy

import neurogym as ngym
import numpy as np

FIXATION = 350
DECISION = 300

# TODO: Check if this is representative of the task from the paper
class Mante13Env(ngym.TrialEnv):
    """Context-based two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    Args:
        mode: str, 'train', 'test' or 'record'. Determines timing of delay
        cohs_motion: arr of floats, possible coherences of motion stimulus
        cohs_color: arr of floats, possible coherences of color stimulus
        dt: int, timestep of integration in ms
        timing: dict, optional, timing of the task
        noise_sigma: float, input noise level
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'two-alternative', 'supervised', 'context-dependent',
                 'motion', 'color', 'integrative stimuli']
    }

    def __init__(self, mode, rand_seed, context_during_fixation, dt, include_color_targets,
                 cohs_motion=None,
                 cohs_color=None,
                 timing=None, stim_noise_sigma=0.25):

        super().__init__(dt=dt)
        self.include_color_targets = include_color_targets

        # Random number generator
        self.rand_seed = rand_seed
        if rand_seed != None:
            self.rng = np.random.RandomState(rand_seed)
        else:
            np.random.RandomState()
        self.dt = dt
        self.mode = mode
        self.context_during_fixation = context_during_fixation # whether to show context cue during fixation period

        # The coherences of each modality
        # Modality 1: motion coherence (dots moving left or right)
        # coh_motion = -1 --> all dots moving left, coh_motion = 1 --> all dots moving right
        # coh_motion = 0 --> half dots moving left, half dots moving right

        # Modality 2: color coherence (dots colored red or green)
        # coh_color = -1 --> all dots colored red, coh_color = 1 --> all dots colored green
        # coh_color = 0 --> half dots colored red, half dots colored green
        if cohs_motion == None:
            cohs_motion = [-0.50, -0.15, -0.05, 0.05, 0.15, 0.50]
        self.cohs_motion = cohs_motion
        if cohs_color == None:
            cohs_color = [-0.50, -0.18, -0.06, 0.06, 0.18, 0.50]
        self.cohs_color = cohs_color
        self.stim_noise_sigma = stim_noise_sigma # / np.sqrt(self.dt)  # Input noise normalized by dt (in ms)

        # Set timing according to mode or input
        if mode in ['train', 'valid']:
            # variable timing for training
            self.timing = {
                "fixation": FIXATION,
                "stimulus": 750,
                "delay": 300, # ["truncated_exponential", [300, 0, 3000]],
                "decision": DECISION
            }
        elif mode in ['record', 'test']:
            # fixed timing for recording
            self.timing = {
                "fixation": FIXATION,
                "stimulus": 750,
                "delay": 300,
                "decision": DECISION
            }
            assert self.stim_noise_sigma == 0.0 #, "stim_noise_sigma must be 0 for recording mode"
        else:
            assert timing != None, "Must specify mode or timing dict!"
        if timing:
            self.timing.update(timing)
        self.init_timing = copy.deepcopy(self.timing)
        self.temp_timing = False

        # Initialize observation space.
        if self.include_color_targets:
            # 9 dimensions: 1 for fixation, 2 for context, 2 for motion stimulus, 2 for color stimulus, 2 for color target order
            num_obs_dims = 9
            obs_dim_names = {'fixation': 0, # fixation cue is 1 until decision period
                            'prop_left': 1, # proportion of dots moving left, between 0 and 1
                            'prop_right': 2, # proportion of dots moving right, between 0 and 1
                            'prop_red': 3, # proportion of dots colored red, between 0 and 1
                            'prop_green': 4, # proportion of dots colored green, between 0 and 1
                            'context': [5, 6], # context is one-hot encoded (motion vs color)
                            'color_target_order': [7, 8]} # color targets are one-hot encoded (indicate if left target is red or green while the right is the remaining color)
            self.observation_space = ngym.spaces.Box(
                -np.inf, np.inf, shape=(num_obs_dims,), dtype=np.float32, name=obs_dim_names)
        else:
            # 7 dimensions: 1 for fixation, 2 for context, 2 for motion stimulus, 2 for color stimulus
            num_obs_dims = 7
            obs_dim_names = {'fixation': 0, # fixation cue is 1 until decision period
                            'prop_left': 1, # proportion of dots moving left, between 0 and 1
                            'prop_right': 2, # proportion of dots moving right, between 0 and 1
                            'prop_red': 3, # proportion of dots colored red, between 0 and 1
                            'prop_green': 4, # proportion of dots colored green, between 0 and 1
                            'context': [5, 6]} # context is one-hot encoded
            self.observation_space = ngym.spaces.Box(
                -np.inf, np.inf, shape=(num_obs_dims,), dtype=np.float32, name=obs_dim_names)

        # Initialize action space.
        # 3 possible actions: 0 for fixation, 1 for choice 1, 2 for choice 2
        num_actions = 3 # 0: no-go, choice 1, choice 2
        act_dim_names = {'fixation': 0, 'choice': [1, 2]}
        self.action_space = ngym.spaces.Discrete(n=num_actions, name=act_dim_names)

        # NOTE: Rewards only apply to reinforcement learning (not implemented yet)
        # # Rewards
        # self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        # if rewards:
        #     self.rewards.update(rewards)
        # self.abort = False

    def _new_trial(self, given_trial_info=None):

        # Trial info
        trial = {
            'context': self.rng.choice([-1, 1]), # 1: motion context, -1: color context
            'coh_motion': self.rng.choice(self.cohs_motion),
            'coh_color': self.rng.choice(self.cohs_color),
        }
        if self.include_color_targets:
            i = self.rng.choice([0,1])
            trial['color_target_order'] = [[1, 0], [0, 1]][i] # 1: green, 0: red
        if given_trial_info != None:
            trial = given_trial_info
            if self.include_color_targets and 'color_target_order' not in trial.keys():
                trial['color_target_order'] = [0, 1] # default to left target is red, right target is green
        trial['prop_right'] = (trial['coh_motion'] + 1) / 2
        trial['prop_left'] = 1 - trial['prop_right']
        trial['prop_green'] = (trial['coh_color'] + 1) / 2
        trial['prop_red'] = 1 - trial['prop_green']
        assert np.isclose(trial['coh_motion'], trial['prop_right'] - trial['prop_left'])
        assert np.isclose(trial['coh_color'], trial['prop_green'] - trial['prop_red'])

        # Set trial timing
        assert self.temp_timing == False
        if given_trial_info != None and 'timing' in given_trial_info.keys(): # TODO: make sure this is working correctly
            # if timing is specified in trial_info, temporarily replace self.timing from original
            self.timing = trial['timing']
            self.temp_timing = True
            trial['timing'] = trial['timing']

        # Set periods using self.timing
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        trial = self.save_period_durations(trial)

        # Set ground truth based on context and coherences
        if trial['context'] == 1: # motion context
            context_onehot = [1, 0]
            # right if more rightward motion, left if more leftward motion
            trial['ground_truth'] = 2 if trial['coh_motion'] > 0 else 1
        elif trial['context'] == -1: # color context
            context_onehot = [0, 1]
            if self.include_color_targets:
                if trial['color_target_order'] == [1,0]: # left target is green, right target is red
                    trial['ground_truth'] = 2 if trial['prop_red'] > trial['prop_green'] else 1
                elif trial['color_target_order'] == [0,1]: # left target is red, right target is green
                    trial['ground_truth'] = 2 if trial['prop_green'] > trial['prop_red'] else 1
            else:
                # right if more green dots, left if more red dots
                trial['ground_truth'] = 2 if trial['prop_red'] > trial['prop_green'] else 1

        # Add values to observations
        context_periods = ['stimulus', 'delay']
        if self.context_during_fixation:
            context_periods = ['fixation'] + context_periods
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(context_onehot, period=context_periods, where='context')
        self.add_ob(trial['prop_left'], period='stimulus', where='prop_left')
        self.add_ob(trial['prop_right'], period='stimulus', where='prop_right')
        self.add_ob(trial['prop_green'], period='stimulus', where='prop_green')
        self.add_ob(trial['prop_red'], period='stimulus', where='prop_red')
        if self.include_color_targets:
            self.add_ob(trial['color_target_order'], period=['fixation', 'stimulus', 'delay', 'decision'], where='color_target_order')
        if self.mode == 'train':
            # add noise to training stimuli only
            for w in ['prop_left', 'prop_right', 'prop_green', 'prop_red']:
                self.add_randn(mu=0, sigma=self.stim_noise_sigma, period='stimulus', where=w)

        # Ground truth
        gt_index = trial['ground_truth'] - 1
        self.set_groundtruth(gt_index, period='decision', where='choice')

        if self.temp_timing:
            # reset timing to original to make sure it is reset for the next trial
            self.timing = self.init_timing
            self.temp_timing = False

        return trial

    def save_period_durations(self, trial):
        # compute durations of each period, trial length in time, and trial length in indices.
        # save in trial dictionary
        if 'timing' not in trial.keys():
            assert len(self.end_t) > 0
            durations = {}
            for i in range(0, len(self.end_t)):
                curr_period = list(self.end_t.keys())[i]
                # first period duration stays as-is
                if i == 0:
                    durations[curr_period] = self.end_t[curr_period]
                else:
                    prev_period = list(self.end_t.keys())[i-1]
                    durations[curr_period] = self.end_t[curr_period] - self.end_t[prev_period]
            trial['timing'] = durations
        assert sum(trial['timing'].values()) == list(self.end_t.values())[-1]
        trial['trial_len_t'] = list(self.end_t.values())[-1]
        trial['trial_len_i'] = list(self.end_ind.values())[-1]
        trial['dt'] = self.dt
        trial['end_i'] = self.end_ind
        trial['start_i'] = self.start_ind
        return trial

    # NOTE: _step function only applies to reinforcement learning (not implemented yet)
    # def _step(self, action):
    #     new_trial = False
    #     # rewards
    #     reward = 0
    #     gt = self.gt_now
    #     # observations
    #     if self.in_period('fixation'):
    #         if action != 0:  # action = 0 means fixating
    #             new_trial = self.abort
    #             reward += self.rewards['abort']
    #     elif self.in_period('decision'):
    #         if action != 0:
    #             new_trial = True
    #             if action == gt:
    #                 reward += self.rewards['correct']
    #                 self.performance = 1
    #             else:
    #                 reward += self.rewards['fail']
    #     return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}



class OriginalManteEnv(ngym.TrialEnv):
    """Context-based two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average. Modified from Mante13Env
    to be closer to original 2013 task.

    Note: Could be merged with Mante13Env

    Args:
        mode: str, 'train', 'test' or 'record'. Determines timing of delay
        cohs_motion: dictionary with arrs of floats, possible coherences of motion stimulus, one for "train" and one for "test"
        cohs_color: dictionary with arrs of floats, possible coherences of color stimulus,  one for "train" and one for "test"
        dt: int, timestep of integration in ms
        timing: dict, optional, timing of the task
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'two-alternative', 'supervised', 'context-dependent',
                 'motion', 'color', 'integrative stimuli']
    }

    def __init__(self, mode, rand_seed, stim_noise_sigma, context_during_fixation, dt,
                 cohs_motion=None,
                 cohs_color=None,
                 timing=None): # IMPORTANT NOTE: original sigma = 31.623 * sqrt(dt=1/1000) = 1 --> if dt=0.050, should be 7.071

        super().__init__(dt=dt)

        # Random number generator
        self.rand_seed = rand_seed
        if rand_seed != None:
            self.rng = np.random.RandomState(rand_seed)
        else:
            np.random.RandomState()
        self.dt = dt
        self.stim_noise_sigma = stim_noise_sigma
        self.context_during_fixation = context_during_fixation # whether to show context cue during fixation period

        # The coherences of each modality
        # Modality 1: motion coherence (dots moving left or right)
        # coh_motion = -1 --> all dots moving left, coh_motion = 1 --> all dots moving right
        # coh_motion = 0 --> half dots moving left, half dots moving right

        # Modality 2: color coherence (dots colored red or green)
        # coh_color = -1 --> all dots colored red, coh_color = 1 --> all dots colored green
        # coh_color = 0 --> half dots colored red, half dots colored green
        if cohs_motion == None:
            cohs_motion = {'train': [-0.1875, 0.1875], 'test': [-0.15, -0.036, -0.009, 0.009, 0.036, 0.15]}
        self.cohs_motion = cohs_motion
        if cohs_color == None:
            cohs_color = {'train': [-0.1875, 0.1875], 'test': [-0.15, -0.036, -0.009, 0.009, 0.036, 0.15]}
        self.cohs_color = cohs_color
        #self.stim_noise_sigma = stim_noise_sigma / np.sqrt(self.dt)  # Input noise

        self.mode=mode #required for tasks later


        # Set timing according to mode or input
        if mode in ['train', 'valid']:
            # variable timing for training
            self.timing = {
                "fixation": FIXATION,
                "stimulus": 750,
                "delay": 300, # ["truncated_exponential", [300, 0, 3000]],
                "decision": DECISION
            }
        elif mode in ['record', 'test']:
            # fixed timing for recording
            self.timing = {
                "fixation": FIXATION,
                "stimulus": 750,
                "delay": 300,
                "decision": DECISION
            }
            self.stim_noise_sigma = 0
            #assert stim_noise_sigma == 0, "stim_noise_sigma must be 0 for recording mode"
        else:
            assert timing != None, "Must specify mode or timing dict!"
        if timing:
            self.timing.update(timing)
        self.init_timing = copy.deepcopy(self.timing)
        self.temp_timing = False

        # Initialize observation space.
        # 4 dimensions: 2 for context, 1 for motion stimulus, 1 for color stimulus
        num_obs_dims = 4
        obs_dim_names = {'coh_motion': 0, # motion coherence
                        'coh_color': 1, # color 'coherence'
                        'context': [2, 3], # context is one-hot encoded
                        }
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(num_obs_dims,), dtype=np.float32, name=obs_dim_names)

        num_actions = 1 # only outputs one number, -1, 0, 1
        act_dim_names = {'choice':0}
        self.action_space = ngym.spaces.Discrete(n=num_actions, name=act_dim_names)


    def _new_trial(self, given_trial_info=None):

        # Trial info
        if self.mode in ['train', 'valid']:
            trial = {
                'context': self.rng.choice([-1, 1]), # 1: motion context, -1: color context
                'coh_motion': np.random.uniform(low=self.cohs_motion['train'][0], high=self.cohs_motion['train'][1]),
                'coh_color': np.random.uniform(low=self.cohs_color['train'][0], high=self.cohs_color['train'][1]),
            }
        elif self.mode in ['test', 'record']:
            trial = {
                'context': self.rng.choice([-1, 1]), # -1: motion context, 1: color context
                'coh_motion': self.rng.choice(self.cohs_motion['test']),
                'coh_color': self.rng.choice(self.cohs_color['test']),
            }

        if given_trial_info != None:
            trial = given_trial_info

        # Set trial timing
        assert self.temp_timing == False
        if given_trial_info != None and 'timing' in given_trial_info.keys(): # TODO: make sure this is working correctly
            # if timing is specified in trial_info, temporarily replace self.timing from original
            self.timing = trial['timing']
            self.temp_timing = True
            trial['timing'] = trial['timing']

        # Set periods using self.timing
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        trial = self.save_period_durations(trial)

        # Set ground truth based on context and coherences
        if trial['context'] == 1: # motion context
            context_onehot = [1, 0]
            # right if more rightward motion, left if more leftward motion
            trial['ground_truth'] = 1 if trial['coh_motion'] > 0 else -1
        elif trial['context'] == -1: # color context
            context_onehot = [0, 1]
            # right if more green dots, left if more red dots
            trial['ground_truth'] = 1 if trial['coh_color'] > 0 else -1

        # Add values to observations
        context_periods = ['stimulus', 'delay']
        if self.context_during_fixation:
            context_periods = ['fixation'] + context_periods
        self.add_ob(context_onehot, period=context_periods, where='context')
        self.add_ob(trial['coh_motion'], period='stimulus', where='coh_motion')
        self.add_ob(trial['coh_color'], period='stimulus', where='coh_color')
        if self.mode == 'train':
            # add noise from gaussian distribution to stimulus for training stimuli only
            for w in ['coh_motion', 'coh_color']:
                self.add_randn(mu=0, sigma=self.stim_noise_sigma, period='stimulus', where=w)

        # Set ground truth
        self.gt = np.zeros(trial['trial_len_i'])
        self.gt[trial['start_i']['decision']:trial['end_i']['decision']] = trial['ground_truth']

        if self.temp_timing:
            # reset timing to original to make sure it is reset for the next trial
            self.timing = self.init_timing
            self.temp_timing = False

        return trial

    def save_period_durations(self, trial):
        # compute durations of each period, trial length in time, and trial length in indices.
        # save in trial dictionary
        if 'timing' not in trial.keys():
            assert len(self.end_t) > 0
            durations = {}
            for i in range(0, len(self.end_t)):
                curr_period = list(self.end_t.keys())[i]
                # first period duration stays as-is
                if i == 0:
                    durations[curr_period] = self.end_t[curr_period]
                else:
                    prev_period = list(self.end_t.keys())[i-1]
                    durations[curr_period] = self.end_t[curr_period] - self.end_t[prev_period]
            trial['timing'] = durations
        assert sum(trial['timing'].values()) == list(self.end_t.values())[-1]
        trial['trial_len_t'] = list(self.end_t.values())[-1]
        trial['trial_len_i'] = list(self.end_ind.values())[-1]
        trial['dt'] = self.dt
        trial['end_i'] = self.end_ind
        trial['start_i'] = self.start_ind
        return trial



if __name__ == '__main__':
    MODE = 'train'
    # COHS_MOTION = #[0.05, 0.15, 0.50]
    # COHS_COLOR = #[0.06, 0.18, 0.50]
    RAND_SEED = 1
    STIM_NOISE_SIGMA = 0.25 # 0
    NUM_TRIALS = 5
    CONTEXT_DURING_FIXATION = True
    INCLUDE_COLOR_TARGETS = True
    dt = 50

    env = Mante13Env(mode=MODE,
                #   cohs_motion=COHS_MOTION,
                #   cohs_color=COHS_COLOR,
                  rand_seed=RAND_SEED,
                  stim_noise_sigma=STIM_NOISE_SIGMA,
                  context_during_fixation=CONTEXT_DURING_FIXATION,
                  dt=dt,
                  include_color_targets=INCLUDE_COLOR_TARGETS,)

    # Test new_trial
    for t in range(NUM_TRIALS):
        trial = env.new_trial()
        print(trial)
        # print(trial['len_ind'])
        # print(env.observation_space)
        # print(env.action_space)
        # print(env.timing)
        # print(env.ob.shape)
        # print(env.ob)
        # print(env.gt.shape)
        print(env.ob)
        print(env.gt)
        # print('__________________')
        # print(t)

    example_trial_info = {'context': -1, 'coh_motion': 0.15, 'coh_color': 0.18, 'color_target_order': [0,1]}
    # trial = env.new_trial(example_trial_info)
    trial = env.new_trial(given_trial_info=example_trial_info)
    assert trial['context'] == example_trial_info['context']
    assert trial['coh_motion'] == example_trial_info['coh_motion']
    assert trial['coh_color'] == example_trial_info['coh_color']
    print(trial)
