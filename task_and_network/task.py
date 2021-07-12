"""Collections of tasks."""

from __future__ import division
import six
import numpy as np

rules_dict = \
    {

        'ozg': ['overlap', 'zero_gap', 'gap'],

        'odr_delay_check': ['odr500', 'odr1000', 'odr', 'odr2000', 'odr2500', ],

        # 'odr_mix_delay_time' : ['odr_mix',],
        'odr_mix_delay_time': ['odr_mix', 'overlap', 'zero_gap', 'gap', 'odrd', 'gap500', ],
        'odr_mix_delay_time_uniform': ['odr_mix_uniform', 'odr_mix_uniform_05_25', 'odr_mix_uniform_01_29'],
        'odr_mix_delay_time_uniform_05_25': ['odr_mix_uniform_05_25', ],
        'odr_mix_delay_time_uniform_01_29': ['odr_mix_uniform_01_29', ],
        # 'odr_mix_delay_time_uniform_05_25_05step' : ['odr_mix_uniform_05_25_05step',],
        # 'odr_mix_delay_time_uniform_06_24_06step' : ['odr_mix_uniform_06_24_06step',],
        # 'odr_mix_delay_time_uniform_01_29_07step' : ['odr_mix_uniform_01_29_07step',],
        'odr_mix_delay_time_uniform_05_25_05step': ['odr_mix_uniform_05_25_05step', 'overlap', 'zero_gap', 'gap',
                                                    'odrd', 'gap500', ],
        'odr_mix_delay_time_uniform_06_24_06step': ['odr_mix_uniform_06_24_06step', 'overlap', 'zero_gap', 'gap',
                                                    'odrd', 'gap500', ],
        'odr_mix_delay_time_uniform_01_29_07step': ['odr_mix_uniform_01_29_07step', 'overlap', 'zero_gap', 'gap',
                                                    'odrd', 'gap500', ],

        'odr_mix_delay_time_uniform_05_25_01step': ['odr_mix_uniform_05_25_01step', 'overlap', 'zero_gap', 'gap',
                                                    'odrd', 'gap500', ],
        'odr_mix_delay_time_uniform_00_30_01step': ['odr_mix_uniform_00_30_01step', 'overlap', 'zero_gap', 'gap',
                                                    'odrd', 'gap500', ],

        'all_new_odr': ['overlap', 'zero_gap', 'gap', 'odr', 'odrd', 'gap500', ],
        'all_new_odr500': ['overlap', 'zero_gap', 'gap', 'odr500', 'odrd', 'gap500', ],
        'all_new_odr1000': ['overlap', 'zero_gap', 'gap', 'odr1000', 'odrd', 'gap500', ],
        'all_new_odr2000': ['overlap', 'zero_gap', 'gap', 'odr2000', 'odrd', 'gap500', ],
        'all_new_odr2500': ['overlap', 'zero_gap', 'gap', 'odr2500', 'odrd', 'gap500', ],

        'all_new_odr3000': ['overlap', 'zero_gap', 'gap', 'odr3000', 'odrd', 'gap500', ],

        'all_new_odr6000': ['overlap', 'zero_gap', 'gap', 'odr6000', 'odrd', 'gap500', ],

        'all_new_odr15000': ['overlap', 'zero_gap', 'gap', 'odr15000', 'odrd', 'gap500', ],

        'all_new': ['overlap', 'zero_gap', 'gap', 'odr', 'odrd', 'gap500', ],

        'mix_MoN_6tasks': ['match_or_non', 'match_or_non_easy', 'overlap', 'zero_gap', 'gap', 'odr', 'odrd',
                           'gap500', ],

        'mix_p_MoN_6tasks': ['match_or_non', 'match_or_non_easy', 'match_or_non_passive', 'overlap', 'zero_gap', 'gap',
                             'odr', 'odrd', 'gap500', ],

        'seq_train_test': ['odr500', 'odrd', 'overlap', 'zero_gap', 'gap', ],

        'MNM_color': ['MNM_color', ],

        'MNM_color_6tasks': ['MNM_color', 'overlap', 'zero_gap', 'gap', 'odr', 'odrd', 'gap500', ],

        'MNM_color_6tasks_1240': ['MNM_color_1240', 'overlap', 'zero_gap', 'gap', 'odr', 'odrd', 'gap500', ],

        'MNM_strength': ['MNM_strength', ],

        'MNM_sequential': ['MNM_sequential_phase1_m', 'MNM_sequential_phase1_nm', 'MNM_sequential_phase2',
                           'MNM_sequential_phase3', 'MNM_sequential_phase4'],

        # MNM_set stores all the match or non-match tasks.
        # It is used to determine whether a task has match or non-match property
        # Do not use it as a training rule set
        'MNM_set': ['match_or_non', 'match_or_non_easy', 'match_or_non_passive', 'MNM_color', 'MNM_strength',
                    'MNM_sequential_phase1_m', 'MNM_sequential_phase1_nm', 'MNM_sequential_phase2',
                    'MNM_sequential_phase3', 'MNM_sequential_phase4'],

        'Capacity_6tasks': ['Capacity_color', 'overlap', 'zero_gap', 'gap', 'odr', 'odrd', 'gap500', ],
    }

# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind


def get_num_ring(ruleset):
    '''get number of stimulus rings'''
    return 3 if ruleset == 'oicdmc' else 2


def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])


def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule] + config['rule_start']


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_eachring)  # preferences

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.input_loc = list()  # add by yichen
        self.output_loc = list()  # add by yichen
        self.distract_loc = list()  # add by yichen
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:, :, :] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)

        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var, additonal_expand_type=tuple):
        """Expand an int/float to list."""
        # if not hasattr(var, '__iter__'):
        # if not isinstance(var, list): #modified by yichen
        if not hasattr(var, '__iter__') or isinstance(var, additonal_expand_type):  # modified by yichen
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
            mods can also be tuple for three modality/channel (RGB) encoding. In this way, strengths will be ignored in type stim, for the
            elements in the tuple will control the strength in corresponding channel (element0:RED, element1:GREEN, element2:BLUE)
        """

        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
            elif loc_type == 'stim':  # modified by yichen
                # Assuming that mods[i] starts from 1
                if not isinstance(mods[i], tuple):
                    self.x[ons[i]: offs[i], i, 1 + (mods[i] - 1) * self.n_eachring:1 + mods[i] * self.n_eachring] \
                        += self.add_x_loc(locs[i]) * strengths[i]
                else:
                    for color in range(len(mods[i])):  # 0:Red 1:Green 2:Blue
                        self.x[ons[i]: offs[i], i, 1 + color * self.n_eachring:1 + (color + 1) * self.n_eachring] \
                            += self.add_x_loc(locs[i]) * mods[i][color]

                self.input_loc.append(self.add_x_loc(locs[i]))  # add by yichen
            #########################add by yichen###############################################
            elif loc_type == 'distract':
                # Assuming that mods[i] starts from 1
                if not isinstance(mods[i], tuple):
                    self.x[ons[i]: offs[i], i, 1 + (mods[i] - 1) * self.n_eachring:1 + mods[i] * self.n_eachring] \
                        += self.add_x_loc(locs[i]) * strengths[i]
                else:
                    for color in range(len(mods[i])):  # 0:Red 1:Green 2:Blue
                        self.x[ons[i]: offs[i], i, 1 + color * self.n_eachring:1 + (color + 1) * self.n_eachring] \
                            += self.add_x_loc(locs[i]) * mods[i][color]
                self.distract_loc.append(self.add_x_loc(locs[i]))
            elif loc_type == 'choice':
                # Assuming that mods[i] starts from 1
                if not isinstance(mods[i], tuple):
                    choices = self.add_choice_loc(locs[i]) * strengths[i]
                    self.x[ons[i]: offs[i], i, 1 + (mods[i] - 1) * self.n_eachring:1 + mods[i] * self.n_eachring] \
                        += choices
                else:
                    for color in range(len(mods[i])):  # 0:Red 1:Green 2:Blue
                        choices = self.add_choice_loc(locs[i]) * mods[i][color]
                        self.x[ons[i]: offs[i], i, 1 + color * self.n_eachring:1 + (color + 1) * self.n_eachring] \
                            += choices

            #########################add by yichen###############################################
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i]) * strengths[i]
                    self.output_loc.append(self.add_y_loc(locs[i]))  # add by yichen
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    self.output_loc.append(y_tmp)  # add by yichen
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons, post_offs=None, passive=False):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on = int(100 / self.dt)  # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)
        post_offs = self.expand(post_offs)  # when to stop checking. default is the end of the trial

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                if passive:
                    c_mask[post_ons[i]:post_offs[i], i, :] = 0.
                else:
                    c_mask[post_ons[i]:post_offs[i], i, :] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 1.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important

            self.c_mask = c_mask.reshape((self.tdim * self.batch_size, self.n_output))
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Having it larger than 1 encourages the network to achieve higher performance
                if passive:
                    c_mask[post_ons[i]:, i] = 0.
                else:
                    c_mask[post_ons[i]:, i] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i] = 1.

            self.c_mask = c_mask.reshape((self.tdim * self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input."""
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start'] + rule] = strength
        elif rule in rules_dict[self.config['ruleset']]:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength
        else:
            import difflib
            similar_score = list()
            for origin_rule in rules_dict[self.config['ruleset']]:
                similar_score.append(difflib.SequenceMatcher(None, rule, origin_rule).quick_ratio())
            max_similar_index = similar_score.index(max(similar_score))
            similar_rule = rules_dict[self.config['ruleset']][max_similar_index]
            ind_rule = get_rule_index(similar_rule, self.config)
            self.x[on:off, :, ind_rule] = strength

            # def add_rule(self, rule, on=None, off=None, strength=1.):

    #    """Add rule input."""
    #    if isinstance(rule, int):
    #        self.x[on:off, :, self.config['rule_start']+rule] = strength
    #    else:
    #        ind_rule = get_rule_index(rule, self.config)
    #        self.x[on:off, :, ind_rule] = strength

    # def add_x_loc(self, x_loc):
    # """Input activity given location."""
    # dist = get_dist(x_loc-self.pref)  # periodic boundary
    # dist /= np.pi/8
    # return 0.8*np.exp(-dist**2/2)

    # add by yichen
    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc - self.pref)  # periodic boundary
        if 'in_loc_type' in self.config and self.config['in_loc_type'] == 'one_hot':
            # One-hot input
            x = np.zeros_like(dist)
            ind = np.argmin(dist)
            x[ind] = 1.
        else:
            dist /= np.pi / 8
            x = 0.8 * np.exp(-dist ** 2 / 2)
        return x

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc - self.pref)  # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi / 8
            y = 0.8 * np.exp(-dist ** 2 / 2)
        else:
            # One-hot output
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y

    # add by yichen
    def add_choice_loc(self, choice_loc, one_hot=False):
        """Input activity given choice location."""
        """choice_loc can be tuple to input multiple choices at one time"""
        choices = np.zeros_like(self.pref)

        if isinstance(choice_loc, tuple):
            for loc in choice_loc:
                dist = get_dist(loc - self.pref)  # periodic boundary
                if one_hot:
                    # One-hot
                    ind = np.argmin(dist)
                    choices[ind] = 1.
                else:
                    dist /= np.pi / 8
                    choices += 0.8 * np.exp(-dist ** 2 / 2)
        else:
            dist = get_dist(choice_loc - self.pref)  # periodic boundary
            if one_hot:
                # One-hot
                ind = np.argmin(dist)
                choices[ind] = 1.
            else:
                dist /= np.pi / 8
                choices += 0.8 * np.exp(-dist ** 2 / 2)

        return choices


def test_init(config, mode, **kwargs):
    '''
    Test initialization of model. mode is not actually used
    Fixation is on then off.
    '''
    dt = config['dt']
    tdim = int(10000 / dt)
    fix_offs = [int(800 / dt)]
    batch_size = 1

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)

    return trial


# add by yichen
def odr_(config, mode, anti_response, delay1_time, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s
        fix_offs = stim_offs + int(delay1_time / dt)  # last for 1.5s
        tdim = fix_offs + int(500 / dt)
        stim_mod = 1
    elif mode == 'test':
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = 1

        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s
        fix_offs = stim_offs + int(delay1_time / dt)  # last for 1.5s
        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'delay1': (stim_offs, fix_offs),
                    'go1': (fix_offs, None)}

    return trial


# add by yichen
def odr(config, mode, **kwargs):
    return odr_(config, mode, False, 1500, **kwargs)


# add by yichen
def odr500(config, mode, **kwargs):
    return odr_(config, mode, False, 500, **kwargs)


# add by yichen
def odr750(config, mode, **kwargs):
    return odr_(config, mode, False, 750, **kwargs)


# add by yichen
def odr1000(config, mode, **kwargs):
    return odr_(config, mode, False, 1000, **kwargs)


# add by yichen
def odr1100(config, mode, **kwargs):
    return odr_(config, mode, False, 1100, **kwargs)


# add by yichen
def odr1200(config, mode, **kwargs):
    return odr_(config, mode, False, 1200, **kwargs)


# add by yichen
def odr1300(config, mode, **kwargs):
    return odr_(config, mode, False, 1300, **kwargs) \
        # add by yichen


def odr1400(config, mode, **kwargs):
    return odr_(config, mode, False, 1400, **kwargs)


# add by yichen
def odr1600(config, mode, **kwargs):
    return odr_(config, mode, False, 1600, **kwargs)


# add by yichen
def odr1700(config, mode, **kwargs):
    return odr_(config, mode, False, 1700, **kwargs)


# add by yichen
def odr1800(config, mode, **kwargs):
    return odr_(config, mode, False, 1800, **kwargs)


# add by yichen
def odr2000(config, mode, **kwargs):
    return odr_(config, mode, False, 2000, **kwargs)


# add by yichen
def odr2500(config, mode, **kwargs):
    return odr_(config, mode, False, 2500, **kwargs)


# add by yichen
def odr3000(config, mode, **kwargs):
    return odr_(config, mode, False, 3000, **kwargs)


# add by yichen
def odr3500(config, mode, **kwargs):
    return odr_(config, mode, False, 3500, **kwargs)


# add by yichen
def odr4000(config, mode, **kwargs):
    return odr_(config, mode, False, 4000, **kwargs)


# add by yichen
def odr4500(config, mode, **kwargs):
    return odr_(config, mode, False, 4500, **kwargs)


# add by yichen
def odr5000(config, mode, **kwargs):
    return odr_(config, mode, False, 5000, **kwargs)


# add by yichen
def odr6000(config, mode, **kwargs):
    return odr_(config, mode, False, 6000, **kwargs)


# add by yichen
def odr15000(config, mode, **kwargs):
    return odr_(config, mode, False, 15000, **kwargs)


# add by yichen

# add by yichen
def odr_mix_(config, mode, delay_time_bound=(500, 2500), **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s

        # delay1_time = delay_time_bound[0]+np.absolute(((delay_time_bound[1]-delay_time_bound[0])/3)*np.random.randn(batch_size))
        # delay1_time = np.clip(delay1_time,delay_time_bound[0],delay_time_bound[1])
        mean_delay_time = (delay_time_bound[1] + delay_time_bound[0]) / 2
        standard_div = (delay_time_bound[1] - delay_time_bound[0]) / 6
        delay1_time = mean_delay_time + standard_div * np.random.randn(batch_size)
        delay1_time = np.clip(delay1_time, delay_time_bound[0], delay_time_bound[1])

        max_fix_off = stim_offs + int(delay_time_bound[1] / dt)
        fix_offs = stim_offs + (delay1_time / dt).astype(np.int32)
        tdim = max_fix_off + int(500 / dt)
        stim_mod = 1
    elif 'test' in mode:
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = 1

        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s
        delay1_time = int(mode.split('-')[1])
        fix_offs = stim_offs + int(delay1_time / dt)
        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)
    check_offs = fix_offs + int(500 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)

    response_locs = stim_locs

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, post_offs=check_offs)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'delay1': (stim_offs, fix_offs),
                    'go1': (fix_offs, check_offs)}

    return trial


# add by yichen
def odr_mix(config, mode, **kwargs):
    return odr_mix_(config, mode, delay_time_bound=(500, 2500), **kwargs)


# add by yichen
def odr_mix_uniform_(config, mode, delay_time_bound=(1000, 2000), delay_step=None, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s

        if delay_step is None:
            delay1_time = np.random.uniform(delay_time_bound[0], delay_time_bound[1], batch_size)
        else:
            time_range = list(range(delay_time_bound[0], delay_time_bound[1] + 1, delay_step))
            delay1_time = np.random.choice(time_range, size=(batch_size))

        max_fix_off = stim_offs + int(delay_time_bound[1] / dt)
        fix_offs = stim_offs + (delay1_time / dt).astype(np.int32)
        tdim = max_fix_off + int(500 / dt)
        stim_mod = 1
    elif 'test' in mode:
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = 1

        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s
        delay1_time = int(mode.split('-')[1])
        fix_offs = stim_offs + int(delay1_time / dt)
        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)
    check_offs = fix_offs + int(500 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)

    response_locs = stim_locs

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, post_offs=check_offs)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'delay1': (stim_offs, fix_offs),
                    'go1': (fix_offs, check_offs)}

    return trial


def odr_mix_uniform(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(1000, 2000), **kwargs)


def odr_mix_uniform_05_25(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(500, 2500), **kwargs)


def odr_mix_uniform_01_29(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(100, 2900), **kwargs)


def odr_mix_uniform_05_25_05step(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(500, 2500), delay_step=500, **kwargs)


def odr_mix_uniform_05_25_01step(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(500, 2500), delay_step=100, **kwargs)


def odr_mix_uniform_00_30_01step(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(0, 3000), delay_step=100, **kwargs)


def odr_mix_uniform_06_24_06step(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(600, 2400), delay_step=600, **kwargs)


def odr_mix_uniform_01_29_07step(config, mode, **kwargs):
    return odr_mix_uniform_(config, mode, delay_time_bound=(100, 2900), delay_step=700, **kwargs)


def odrd_(config, mode, **kwargs):
    # use test-x-y to control the cue/stim(x), and distractor(y) if you want to specifiy their location

    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s
        distract_ons = stim_offs + int(500 / dt)  # delay for 0.5s
        distract_offs = distract_ons + int(500 / dt)  # last for 0.5s
        fix_offs = distract_offs + int(500 / dt)  # last for 0.5s
        tdim = fix_offs + int(500 / dt)
        stim_mod = 1

    elif mode[0:4] == 'test':
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc

        if '-' in mode:
            stim_locs = 2 * np.pi * (0 * stim_locs + int(mode.split('-')[1])) / config['n_eachring']

        stim_mod = 1

        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(500 / dt)  # last for 0.5s
        distract_ons = stim_offs + int(500 / dt)  # delay for 0.5s
        distract_offs = distract_ons + int(500 / dt)  # last for 0.5s
        fix_offs = distract_offs + int(500 / dt)  # last for 0.5s
        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    response_locs = stim_locs
    if mode[0:4] == 'test':
        if '-' in mode:
            distract_locs = 2 * np.pi * (0 * stim_locs + int(mode.split('-')[2])) / config['n_eachring']
        elif mode == 'test':
            distract_locs = (stim_locs + np.pi) % (2 * np.pi)

    elif mode == 'random':
        distract_locs = (stim_locs + rng.choice(np.arange(1, config['n_eachring'])) * (
                    2 * np.pi / config['n_eachring'])) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('distract', distract_locs, ons=distract_ons, offs=distract_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'delay1': (stim_offs, distract_ons),
                    'distract1': (distract_ons, distract_offs),
                    'delay2': (distract_offs, fix_offs),
                    'go1': (fix_offs, None)}

    return trial


# add by yichen
def odrd(config, mode, **kwargs):
    return odrd_(config, mode, **kwargs)


# add by yichen
def overlap_(config, mode, anti_response, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(100 / dt)  # last for 100ms
        fix_offs = stim_offs  # turn off at the same time
        tdim = fix_offs + int(500 / dt)
        stim_mod = 1

    elif mode == 'test':
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = 1

        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(100 / dt)  # last for 100ms
        fix_offs = stim_offs  # turn off at the same time
        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'go1': (fix_offs, None)}

    return trial


# add by yichen
def overlap(config, mode, **kwargs):
    return overlap_(config, mode, True, **kwargs)


# add by yichen
def zero_gap_(config, mode, anti_response, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(100 / dt)  # last for 100ms
        fix_offs = stim_ons  # turn off when stim appears
        tdim = stim_offs + int(500 / dt)
        stim_mod = 1

    elif mode == 'test':
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = 1

        stim_ons = int(1000 / dt)
        stim_offs = stim_ons + int(100 / dt)  # last for 100ms
        fix_offs = stim_ons  # turn off when stim appears
        tdim = stim_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # check_ons= stim_offs + int(100/dt)
    check_ons = stim_ons + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    # trial.add('fix_out', offs=stim_offs)
    trial.add('fix_out', offs=stim_ons)
    # trial.add('out', response_locs, ons=stim_offs)
    trial.add('out', response_locs, ons=stim_ons)
    # trial.add_c_mask(pre_offs=stim_offs, post_ons=check_ons)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'go1': (stim_ons, None)}

    return trial


# add by yichen
def zero_gap(config, mode, **kwargs):
    return zero_gap_(config, mode, True, **kwargs)


# add by yichen
def gap_(config, mode, anti_response, gap_time, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimuluss and on/off time
        fix_offs = int(1000 / dt)
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = fix_offs + int(gap_time / dt)  # gap for 100ms
        stim_offs = stim_ons + int(100 / dt)  # last for 100ms

        tdim = stim_offs + int(500 / dt)
        stim_mod = 1

    elif mode == 'test':
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = 1

        fix_offs = int(1000 / dt)
        stim_ons = fix_offs + int(gap_time / dt)  # gap for 100ms
        stim_offs = stim_ons + int(100 / dt)  # last for 100ms
        tdim = stim_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # check_ons= stim_offs + int(100/dt)
    check_ons = stim_ons + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    # trial.add('fix_out', offs=stim_offs)
    trial.add('fix_out', offs=stim_ons)
    # trial.add('out', response_locs, ons=stim_offs
    trial.add('out', response_locs, ons=stim_ons)
    # trial.add_c_mask(pre_offs=stim_offs, post_ons=check_ons)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

    trial.epochs = {'fix1': (None, fix_offs),
                    'delay1': (fix_offs, stim_ons),
                    'stim1': (stim_ons, stim_offs),
                    'go1': (stim_ons, None)}

    return trial


# add by yichen
def gap(config, mode, **kwargs):
    return gap_(config, mode, True, 100, **kwargs)


# add by yichen
def gap500(config, mode, **kwargs):
    return gap_(config, mode, True, 500, **kwargs)


def match_or_non_(config, mode, easy_task, passive, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        # Stimuli
        batch_size = kwargs['batch_size']
        stim1_locs = rng.rand(batch_size) * 2 * np.pi
        match_or_not = rng.randint(0, 2,
                                   batch_size)  # an array consists of 0&1 with the size of stim1_locs. 0 is match 1 is non-match
        devi_dist = rng.randint(1, config['n_eachring']) * (2 * np.pi / config['n_eachring'])
        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        match_choice = np.random.randint(0, 2, batch_size)
        if easy_task:
            upper_choice = np.zeros_like(match_choice)
            lower_choice = np.ones_like(match_choice) * np.pi
            # upper is match,lower is nonmatch
        else:
            upper_choice = match_choice * 3 / 2 * np.pi + 1 / 4 * np.pi
            lower_choice = (upper_choice + np.pi) % (2 * np.pi)
            """
            match_choice: 0: the upper is "match", the lower is "non-match"
                          1: the upper is "non-match", the lower is "match"
            upper choice: match: 1/4pi non-match: 7/4pi
            lower choice: match: 3/4pi non-match: 5/4pi
            """
        choices = list(zip(upper_choice, lower_choice))

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(500 / dt)  # delay for 0.5s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(500 / dt)  # delay for 0.5s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

        # stimuli mode
        stim_mod = 1

    elif mode == 'test':
        # Stimuli
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)

        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc

        # match_or_not = rng.randint(0,2,len(stim1_locs))
        # devi_dist = rng.randint(1,config['n_eachring'])*(2*np.pi/config['n_eachring'])

        match_or_not = np.unravel_index(range(batch_size), (config['n_eachring'] * 2, 8))[0] % 2
        devi_dist = np.pi

        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        match_choice = np.random.randint(0, 2, batch_size)  ###########
        if easy_task:
            upper_choice = np.zeros_like(match_choice)
            lower_choice = np.ones_like(match_choice) * np.pi
            # upper is match,lower is nonmatch#
        else:
            upper_choice = match_choice * 3 / 2 * np.pi + 1 / 4 * np.pi
            lower_choice = (upper_choice + np.pi) % (2 * np.pi)
            """
            match_choice: 0: the upper is "match", the lower is "non-match"
                          1: the upper is "non-match", the lower is "match"
            upper choice: match: 1/4pi non-match: 7/4pi
            lower choice: match: 3/4pi non-match: 5/4pi
            """
        choices = list(zip(upper_choice, lower_choice))

        # stimuli mode
        stim_mod = 1

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(500 / dt)  # delay for 0.5s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(500 / dt)  # delay for 0.5s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Response locations
    if easy_task:
        response_locs = match_or_not * np.pi
    else:
        response_locs = np.abs(match_or_not - match_choice) * np.pi

    trial = Trial(config, tdim, batch_size)
    if passive:
        trial.add('fix_in')
    else:
        trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=stim_mod)
    # you can change the choice encode mode in add_choice_loc func,default one_hot = False
    trial.add('choice', choices, ons=choice_ons, mods=stim_mod)
    if passive:
        trial.add('fix_out', offs=fix_offs)
        trial.add('out', response_locs * 0, ons=fix_offs)
    else:
        trial.add('fix_out', offs=fix_offs)
        trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, passive=passive)

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def match_or_non(config, mode, **kwargs):
    return match_or_non_(config, mode, easy_task=False, passive=False, **kwargs)


def match_or_non_easy(config, mode, **kwargs):
    return match_or_non_(config, mode, easy_task=True, passive=False, **kwargs)


def match_or_non_passive(config, mode, **kwargs):
    return match_or_non_(config, mode, easy_task=False, passive=True, **kwargs)


def MNM_color_(config, mode, delaytime, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        # Stimuli
        batch_size = kwargs['batch_size']
        stim1_locs = rng.rand(batch_size) * 2 * np.pi
        match_or_not = rng.randint(0, 2,
                                   batch_size)  # an array consists of 0&1 with the size of stim1_locs. 0 is match 1 is non-match
        devi_dist = rng.randint(1, config['n_eachring']) * (2 * np.pi / config['n_eachring'])
        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(delaytime / dt)  # delay
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(delaytime / dt)  # delay

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        # Stimuli
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)

        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc

        # match_or_not = rng.randint(0,2,len(stim1_locs))
        # devi_dist = rng.randint(1,config['n_eachring'])*(2*np.pi/config['n_eachring'])

        match_or_not = np.unravel_index(range(batch_size), (config['n_eachring'] * 2, 8))[0] % 2
        devi_dist = np.pi

        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(delaytime / dt)  # delay
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(delaytime / dt)  # delay

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Match and Non-Match choice location
    M_choice_loc = np.random.randint(config['n_eachring'], size=batch_size) * (2 * np.pi / config['n_eachring'])
    NM_choice_loc = (M_choice_loc + np.pi) % (2 * np.pi)

    # Response locations
    response_locs = (M_choice_loc + match_or_not * np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('choice', M_choice_loc, ons=choice_ons, mods=(0, 1, 0))  # Green
    trial.add('choice', NM_choice_loc, ons=choice_ons, mods=(0, 0, 1))  # Blue

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, )

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def MNM_color(config, mode, **kwargs):  # 0.5s delay
    return MNM_color_(config, mode, delaytime=500, **kwargs)


def MNM_color_1240(config, mode, **kwargs):  # 1.24s delay
    return MNM_color_(config, mode, delaytime=1240, **kwargs)


def MNM_strength_(config, mode, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        # Stimuli
        batch_size = kwargs['batch_size']
        stim1_locs = rng.rand(batch_size) * 2 * np.pi
        match_or_not = rng.randint(0, 2,
                                   batch_size)  # an array consists of 0&1 with the size of stim1_locs. 0 is match 1 is non-match
        devi_dist = rng.randint(1, config['n_eachring']) * (2 * np.pi / config['n_eachring'])
        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(500 / dt)  # delay for 0.5s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(500 / dt)  # delay for 0.5s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        # Stimuli
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)

        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc

        # match_or_not = rng.randint(0,2,len(stim1_locs))
        # devi_dist = rng.randint(1,config['n_eachring'])*(2*np.pi/config['n_eachring'])

        match_or_not = np.unravel_index(range(batch_size), (config['n_eachring'] * 2, 8))[0] % 2
        devi_dist = np.pi

        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(500 / dt)  # delay for 0.5s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(500 / dt)  # delay for 0.5s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Match and Non-Match choice location
    M_choice_loc = np.random.randint(config['n_eachring'], size=batch_size) * (2 * np.pi / config['n_eachring'])
    NM_choice_loc = (M_choice_loc + np.pi) % (2 * np.pi)

    # Response locations
    response_locs = (M_choice_loc + match_or_not * np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=1)
    trial.add('choice', M_choice_loc, ons=choice_ons, mods=1, strengths=1)
    trial.add('choice', NM_choice_loc, ons=choice_ons, mods=1, strengths=0.5)

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, )

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def MNM_strength(config, mode, **kwargs):
    return MNM_strength_(config, mode, **kwargs)


def MNM_sequential_phase_1_(config, mode, MorNM, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        # Stimuli
        batch_size = kwargs['batch_size']
        stim1_locs = np.ones(batch_size) * 0.5 * np.pi
        if MorNM == 'match':
            match_or_not = np.zeros((batch_size,), dtype=int)
        elif MorNM == 'non-match':
            match_or_not = np.ones((batch_size,), dtype=int)

        devi_dist = np.pi
        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(250 / dt)  # delay for 0.25s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(250 / dt)  # delay for 0.25s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        # Stimuli
        batch_size = 32

        stim1_locs = np.ones(batch_size) * 0.5 * np.pi

        match_or_not = np.unravel_index(range(batch_size), (2, 16))[0]

        devi_dist = np.pi

        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(250 / dt)  # delay for 0.25s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(250 / dt)  # delay for 0.25s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Match and Non-Match choice location
    M_choice_loc = np.zeros(batch_size)
    NM_choice_loc = np.ones(batch_size) * np.pi

    # Response locations
    response_locs = (M_choice_loc + match_or_not * np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('choice', M_choice_loc, ons=choice_ons, mods=(0, 1, 0))  # Green
    trial.add('choice', NM_choice_loc, ons=choice_ons, mods=(0, 0, 1))  # Blue

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, )

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def MNM_sequential_phase_2_(config, mode, **kwargs):  # def MNM_sequential_phase_1_2_(config, mode, phase, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        # Stimuli
        batch_size = kwargs['batch_size']
        stim1_locs = np.ones(batch_size) * 0.5 * np.pi
        # if phase == 1:
        #    match_or_not = np.ones((batch_size,), dtype=int)
        #    match_or_not[:batch_size//2] = 0
        # elif phase == 2:
        match_or_not = rng.randint(0, 2,
                                   batch_size)  # an array consists of 0&1 with the size of stim1_locs. 0 is match 1 is non-match
        devi_dist = np.pi
        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(250 / dt)  # delay for 0.25s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(250 / dt)  # delay for 0.25s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        # Stimuli
        batch_size = 32

        stim1_locs = np.ones(batch_size) * 0.5 * np.pi

        match_or_not = np.unravel_index(range(batch_size), (2, 16))[0]

        devi_dist = np.pi

        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(250 / dt)  # delay for 0.25s
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(250 / dt)  # delay for 0.25s

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Match and Non-Match choice location
    M_choice_loc = np.zeros(batch_size)
    NM_choice_loc = np.ones(batch_size) * np.pi

    # Response locations
    response_locs = (M_choice_loc + match_or_not * np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('choice', M_choice_loc, ons=choice_ons, mods=(0, 1, 0))  # Green
    trial.add('choice', NM_choice_loc, ons=choice_ons, mods=(0, 0, 1))  # Blue

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, )

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def MNM_sequential_phase_3_4_(config, mode, delaytime, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        # Stimuli
        batch_size = kwargs['batch_size']
        stim1_locs = rng.rand(batch_size) * 2 * np.pi  # random location
        match_or_not = rng.randint(0, 2, batch_size)  # mix trials. 0 is match 1 is non-match
        devi_dist = np.pi
        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(delaytime / dt)  # delay
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(delaytime / dt)  # delay

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        # Stimuli
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)

        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc

        match_or_not = np.unravel_index(range(batch_size), (config['n_eachring'] * 2, 8))[0] % 2
        devi_dist = np.pi

        stim2_locs = (stim1_locs + devi_dist * match_or_not) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(delaytime / dt)  # delay
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(delaytime / dt)  # delay

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Match and Non-Match choice location
    M_choice_loc = np.zeros(batch_size)
    NM_choice_loc = np.ones(batch_size) * np.pi

    # Response locations
    response_locs = (M_choice_loc + match_or_not * np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('choice', M_choice_loc, ons=choice_ons, mods=(0, 1, 0))  # Green
    trial.add('choice', NM_choice_loc, ons=choice_ons, mods=(0, 0, 1))  # Blue

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, )

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def MNM_sequential_phase1_m(config, mode, **kwargs):
    return MNM_sequential_phase_1_(config, mode, 'match', **kwargs)


def MNM_sequential_phase1_nm(config, mode, **kwargs):
    return MNM_sequential_phase_1_(config, mode, 'non-match', **kwargs)


def MNM_sequential_phase2(config, mode, **kwargs):
    return MNM_sequential_phase_2_(config, mode, **kwargs)


def MNM_sequential_phase3(config, mode, **kwargs):
    return MNM_sequential_phase_3_4_(config, mode, 250, **kwargs)


def MNM_sequential_phase4(config, mode, **kwargs):
    return MNM_sequential_phase_3_4_(config, mode, 1500, **kwargs)


def Capacity_color_(config, mode, delaytime, **kwargs):
    #                delay1    delay2
    # ----------^^^^^-----^^^^^-----^^^^^
    #           stim1     stim2     choice
    # ------------------------------>>>>>
    #            fixation            go
    #
    # >72 n_eachring suggested
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':  # Randomly generate parameters
        # Stimuli
        batch_size = kwargs['batch_size']
        # stim1s = rng.rand(5,batch_size)*2*np.pi
        stim1s = rng.rand(2, batch_size) * 2 * np.pi  # reduce stims per epoch to 2

        devi_dist = np.zeros_like(stim1s)
        match_or_not = rng.randint(0, 2,
                                   batch_size)  # an array consists of 0&1 with the size of stim1_locs. 0 is match 1 is non-match
        devi_dist[0, :] = rng.randint(1, config['n_eachring']) * (2 * np.pi / config['n_eachring']) * match_or_not
        stim2s = (stim1s + devi_dist) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(delaytime / dt)  # delay
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(delaytime / dt)  # delay

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        # Stimuli
        n_stim_loc, _ = batch_shape = config['n_eachring'], 16
        batch_size = np.prod(batch_shape)
        ind_stim_loc, _ = np.unravel_index(range(batch_size), batch_shape)

        # stim1s = rng.rand(5,batch_size)*2*np.pi
        stim1s = rng.rand(2, batch_size) * 2 * np.pi  # reduce stims per epoch to 2

        stim1s[0, :] = 2 * np.pi * ind_stim_loc / n_stim_loc

        match_or_not = np.unravel_index(range(batch_size), (config['n_eachring'] * 2, 8))[0] % 2
        devi_dist = np.zeros_like(stim1s)
        devi_dist[0, :] = match_or_not * np.pi

        stim2s = (stim1s + devi_dist) % (2 * np.pi)

        # Timeline
        stim1_ons = int(1000 / dt)
        stim1_offs = stim1_ons + int(500 / dt)  # last for 0.5s

        stim2_ons = stim1_offs + int(delaytime / dt)  # delay
        stim2_offs = stim2_ons + int(500 / dt)  # last for 0.5s

        choice_ons = stim2_offs + int(delaytime / dt)  # delay

        fix_offs = choice_ons

        tdim = fix_offs + int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    # Match and Non-Match choice location
    M_choice_loc = np.random.randint(config['n_eachring'], size=batch_size) * (2 * np.pi / config['n_eachring'])
    NM_choice_loc = (M_choice_loc + np.pi) % (2 * np.pi)

    # Response locations
    response_locs = (M_choice_loc + match_or_not * np.pi) % (2 * np.pi)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1s[0], ons=stim1_ons, offs=stim1_offs, mods=(1, 1, 1))  # RGB(white)
    trial.add('stim', stim2s[0], ons=stim2_ons, offs=stim2_offs, mods=(1, 1, 1))  # RGB(white)
    # for i in range(1,5):
    for i in range(1, 2):
        trial.add('distract', stim1s[i], ons=stim1_ons, offs=stim1_offs, mods=(1, 1, 1))  # RGB(white)
    # for i in range(1,5):
    for i in range(1, 2):
        trial.add('distract', stim2s[i], ons=stim2_ons, offs=stim2_offs, mods=(1, 1, 1))  # RGB(white)

    trial.add('choice', M_choice_loc, ons=choice_ons, mods=(0, 1, 0))  # Green
    trial.add('choice', NM_choice_loc, ons=choice_ons, mods=(0, 0, 1))  # Blue

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons, )

    trial.epochs = {'fix1': (None, stim1_ons),
                    'stim1': (stim1_ons, stim1_offs),
                    'delay1': (stim1_offs, stim2_ons),
                    'stim2': (stim2_ons, stim2_offs),
                    'delay2': (stim2_offs, choice_ons),
                    'go1': (fix_offs, None)}  # go period is also the choice display period

    return trial


def Capacity_color(config, mode, **kwargs):
    return Capacity_color_(config, mode, 500, **kwargs)


rule_mapping = {
    'odr': odr,
    # odr_delay_check
    'odr500': odr500,
    'odr750': odr750,
    'odr1000': odr1000,
    'odr1100': odr1100,
    'odr1200': odr1200,
    'odr1300': odr1300,
    'odr1400': odr1400,
    'odr1600': odr1600,
    'odr1700': odr1700,
    'odr1800': odr1800,
    'odr2000': odr2000,
    'odr2500': odr2500,
    'odr3000': odr3000,
    'odr3500': odr3500,
    'odr4000': odr4000,
    'odr4500': odr4500,
    'odr5000': odr5000,
    'odr6000': odr6000,
    'odr15000': odr15000,
    ###################
    'odr_mix': odr_mix,
    'odr_mix_uniform': odr_mix_uniform,
    'odr_mix_uniform_05_25': odr_mix_uniform_05_25,
    'odr_mix_uniform_01_29': odr_mix_uniform_01_29,
    'odr_mix_uniform_05_25_05step': odr_mix_uniform_05_25_05step,
    'odr_mix_uniform_06_24_06step': odr_mix_uniform_06_24_06step,
    'odr_mix_uniform_01_29_07step': odr_mix_uniform_01_29_07step,
    'odr_mix_uniform_05_25_01step': odr_mix_uniform_05_25_01step,
    'odr_mix_uniform_00_30_01step': odr_mix_uniform_00_30_01step,
    'odrd': odrd,
    'overlap': overlap,
    'zero_gap': zero_gap,
    'gap': gap,
    'gap500': gap500,
    'match_or_non': match_or_non,
    'match_or_non_easy': match_or_non_easy,
    'match_or_non_passive': match_or_non_passive,
    'MNM_color': MNM_color,
    'MNM_color_1240': MNM_color_1240,
    'MNM_strength': MNM_strength,
    # 'MNM_sequential_phase1':MNM_sequential_phase1,
    'MNM_sequential_phase1_m': MNM_sequential_phase1_m,
    'MNM_sequential_phase1_nm': MNM_sequential_phase1_nm,
    'MNM_sequential_phase2': MNM_sequential_phase2,
    'MNM_sequential_phase3': MNM_sequential_phase3,
    'MNM_sequential_phase4': MNM_sequential_phase4,
    'Capacity_color': Capacity_color,
}

rule_name = {
    'odr': 'ODR',
    # odr_delay_check
    'odr500': 'ODR500',
    'odr750': 'ODR750',
    'odr1000': 'ODR1000',
    'odr1100': 'ODR1100',
    'odr1200': 'ODR1200',
    'odr1300': 'ODR1300',
    'odr1400': 'ODR1400',
    'odr1600': 'ODR1600',
    'odr1700': 'ODR1700',
    'odr1800': 'ODR1800',
    'odr2000': 'ODR2000',
    'odr2500': 'ODR2500',
    'odr3000': 'ODR3000',
    'odr3500': 'ODR3500',
    'odr4000': 'ODR4000',
    'odr4500': 'ODR4500',
    'odr5000': 'ODR5000',
    'odr6000': 'ODR6000',
    'odr15000': 'ODR15000',
    ####################
    'odr_mix': 'ODR mix delay length',
    'odr_mix_uniform': 'ODR mix delay (uniform) length',
    'odr_mix_uniform_05_25': 'ODR mix delay (uniform 0.5-2.5s)',
    'odr_mix_uniform_01_29': 'ODR mix delay (uniform 0.1-2.9s)',
    'odr_mix_uniform_05_25_05step': 'ODR mix delay (uniform 0.5-2.5s,0.5s step)',
    'odr_mix_uniform_06_24_06step': 'ODR mix delay (uniform 0.6-2.4s,0.6s step)',
    'odr_mix_uniform_01_29_07step': 'ODR mix delay (uniform 0.1-2.9s,0.7s step)',
    'odr_mix_uniform_05_25_01step': 'ODR mix delay (uniform 0.5-2.5s,0.1s step)',
    'odr_mix_uniform_00_30_01step': 'ODR mix delay (uniform 0.0-3.0s,0.1s step)',
    'odrd': 'ODR+d',
    'overlap': 'Overlap',
    'zero_gap': 'Zero_gap',
    'gap': 'Gap',
    'gap500': 'Gap500',
    'match_or_non': 'MorN',
    'match_or_non_easy': 'MorNe',
    'match_or_non_passive': 'MorNp',
    'MNM_color': 'MNM color encode',
    'MNM_color_1240': 'MNM 1.24s color encode',
    'MNM_strength': 'MNM strength encode',
    # 'MNM_sequential_phase1':'MNM seq phase1',
    'MNM_sequential_phase1_m': 'MNM seq phase1 M',
    'MNM_sequential_phase1_nm': 'MNM seq phase1 NM',
    'MNM_sequential_phase2': 'MNM seq phase2',
    'MNM_sequential_phase3': 'MNM seq phase3',
    'MNM_sequential_phase4': 'MNM seq phase4',
    'Capacity_color': 'Capacity task'
}


def generate_trials(rule, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    trial = rule_mapping[rule](config, mode, **kwargs)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else:  # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else:  # default behavior
        rule_off = None

    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']

    if rule is 'testinit':
        # Add no rule
        return trial

    if isinstance(rule, six.string_types):
        # rule is not iterable
        # Expand to list
        if 'rule_strength' in kwargs:
            rule_strength = [kwargs['rule_strength']]
        else:
            rule_strength = [1.]
        rule = [rule]

    else:
        if 'rule_strength' in kwargs:
            rule_strength = kwargs['rule_strength']
        else:
            rule_strength = [1.] * len(rule)

    for r, s in zip(rule, rule_strength):
        trial.add_rule(r, on=rule_on, off=rule_off, strength=s)

    if noise_on:
        trial.add_x_noise()

    return trial
