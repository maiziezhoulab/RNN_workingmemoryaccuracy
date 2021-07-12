import pickle
# trial generation and network building #

import sys


import os
import tensorflow as tf
import pickle
import copy
from numpy import *
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
sys.path.append('.')
from utils import tools
from scipy.stats.stats import pearsonr
from utils.tools import mkdir_p,smooth

from task_and_network.task import generate_trials
from task_and_network.network import Model,popvec
from analysis.PSTH_print_basic_info import print_basic_info
from analysis.PSTH_compute_H import compute_H, gen_task_info
from analysis.PSTH_compute_H import Get_H
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.optimize import curve_fit
import math
import cmath
import statistics


def select_neurons_excitatory(
        hp,
        rule,
        task_info,
        epoch,
        correct_H,
        correct_loc_info,
):
    """Select significant neurons with firing rates at least greater than fixation in one location

    """
    task_info_rule = task_info[rule]
    significant_neuron = []
    for neuron in range(hp['n_rnn']):
        fix_loc = []
        epoch_loc = []
        for loc in task_info[rule]['in_loc_set']:
            fix_loc.append([])
            epoch_loc.append([])

        for loc in task_info[rule]['in_loc_set']:
            fix_level = correct_H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                    correct_loc_info == loc, neuron].mean(axis=0)*50

            fire_rate = correct_H[task_info_rule['epoch_info'][epoch][0]:task_info_rule['epoch_info'][epoch][1], \
                        correct_loc_info == loc, neuron].mean(axis=0)*50
            fix_loc[loc].append(fix_level.mean())
            epoch_loc[loc].append(fire_rate.mean())
        fix_epoch_flag = []
        for each_fix, each_epoch in zip(fix_loc,epoch_loc):
            if str(fix) != 'nan' and str(epoch) != 'nan':
                if each_epoch > each_fix:
                    fix_epoch_flag.append(True)
                else: fix_epoch_flag.append(False)
            else:
                print('empty location')

        # epoch fr > fixation fr in one location
        if True in fix_epoch_flag:
            significant_neuron.append(neuron)
    return significant_neuron

def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))*180/np.pi
    corr_loc = dist < 36

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)

    return perf, list(dist),y_loc,y_hat_loc

def gaussian(x, a, u, sig):
    """Compute gaussian curve.
    """
    return a * np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))

def compute_H_(hp, model_dir, rule, trial_num,trial=None, task_mode='test'):
    """Compute gaussian curve.

    Returns:
    h: Firing rate matrix (Time, Batch, Unit)
    perf: Performance (boolean: correct or error trials)
    y_loc: Target output location (-1 for fixation) at last time point
    y_hat_loc: Predicted output location at last time point.
    """
    if trial is None:
        trial = generate_trials(rule, hp, task_mode, noise_on=False)

    sub_dir = model_dir + '/' + str(trial_num) + '/'
    model = Model(sub_dir, hp=hp)
    with tf.Session() as sess:
        model.restore() # load model by trial number
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h = sess.run(model.h, feed_dict=feed_dict)
        y_hat = sess.run(model.y_hat, feed_dict=feed_dict)
        perf, dist, y_loc, y_hat_loc = get_perf(y_hat, trial.y_loc)
    return h, perf,y_loc,y_hat_loc

def split_trials(hp,log, model_dir, rule, models_select,task_info, trial=None, task_mode='test'):
    """Split trials.
    Returns:
     all_H: Firing rate matrix (Time, Batch, Unit)
     correct_idx: Correct trials index
     correct_cw_idx: Correct clockwise trials index
     correct_ccw_idx: Correct counterclockwise trials index
     all_stage_keys: model stage
     all_in_loc_info: model trials location information
     dev_theta: Absolute distance between actual output location and median location
    """

    is_dict = False
    is_list = False
    if isinstance(models_select, dict):
        temp_list = list()
        is_dict = True
        for key,value in models_select.items():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(models_select, list):
        temp_list = models_select
        is_list = True

    all_stage_keys = []

    for i in range(len(temp_list)):
        # decide stage
        growth = log['perf_'+rule][temp_list[i]//log['trials'][1]]
        if (is_list and growth > hp['mid_target_perf']) or (is_dict and temp_list[i] in models_select['mature']):
            mature_key = "mature"
        elif (is_list and growth > hp['early_target_perf']) or (is_dict and temp_list[i] in models_select['mid']):
            mature_key = "mid"
        elif is_list or (is_dict and temp_list[i] in models_select['early']):
            mature_key = "early"

        model_num = temp_list[i]
        H, perf,y_loc,y_hat_loc = compute_H_(hp, model_dir, rule, model_num,trial=None, task_mode='test')
        correct_idx, correct_cw_idx, correct_ccw_idx,dev_theta = median_split_cw_ccw(perf, y_hat_loc,task_info,rule)

        temp_task_info = deepcopy(task_info)
        all_in_loc_info = temp_task_info[rule]['in_loc']

        if i == 0:
            all_H = H
            all_stage_keys = mature_key
        else:
            all_H = np.concatenate([all_H,H],-2)
            all_stage_keys += [mature_key]*len(perf)
    return all_H,correct_idx,correct_cw_idx, correct_ccw_idx, all_stage_keys,all_in_loc_info,dev_theta

def median_split_cw_ccw(perf,y_hat_loc,task_info,rule):
    """Split correct trials into clockwise and counterclockwise by the median of output location.
    Returns:
     correct_idx: Correct trials index
     correct_cw_idx: Correct clockwise trials index
     correct_ccw_idx: Correct counterclockwise trials index
     dev_theta: Absolute distance between actual output location and median location
    """
    num_loc = len(task_info[rule]['in_loc_set'])
    median_loc = []
    correct_y_hat_loc = []
    correct_y_hat_idx = []
    for i in range(num_loc):
        correct_y_hat_loc.append([])
        correct_y_hat_idx.append([])
        median_loc.append([])
    correct_idx = []
    for i in range(len(perf)):
        if perf[i]:
            correct_idx.append(i) # find correct trials
            if i//16 == 0 and y_hat_loc[i]<np.pi*0.25: # at location 0
                correct_y_hat_loc[i//16].append(y_hat_loc[i]+2*np.pi)
                correct_y_hat_idx[i//16].append(i)
            else: # all other locations
                correct_y_hat_loc[i//16].append(y_hat_loc[i])
                correct_y_hat_idx[i//16].append(i)
    for i in range(num_loc):
        if len(correct_y_hat_loc[i]):
            # compute median of correct trials locations at each location
            median_loc[i] = statistics.median(correct_y_hat_loc[i])

    correct_cw_idx = []
    correct_ccw_idx = []
    dev_theta = []
    for i in range(len(median_loc)):
        if median_loc[i]:
            for each_y_hat_loc, each_y_hat_idx in zip(correct_y_hat_loc[i],correct_y_hat_idx[i]):
                # compute absolute distance between actual output location and median location
                dist = abs(each_y_hat_loc - median_loc[i])*180/np.pi
                dev_theta.append(dist)
                if each_y_hat_loc <= median_loc[i]:  # clockwise
                    correct_cw_idx.append(each_y_hat_idx)
                else: correct_ccw_idx.append(each_y_hat_idx) # counterclockwise
    return correct_idx,correct_cw_idx, correct_ccw_idx,dev_theta

def select_H_by_trialidx(H,idx):
    """slice 3D matrix by second dimension
    """
    return np.take(H, idx, axis=1)

def max_central(max_dir,tuning):
    """Compute spatical tuning central location
    """
    temp_len = len(tuning)
    if temp_len%2 == 0:
        mc_len = temp_len + 1
    else:
        mc_len = temp_len

    firerate_max_central = np.zeros(mc_len)
    idx_max_central = []
    for i in range(temp_len):
        new_index = (i-max_dir+temp_len//2)%temp_len
        firerate_max_central[new_index] = tuning[i]
        idx_max_central.append(new_index)
    if temp_len%2 == 0:
        firerate_max_central[-1] = firerate_max_central[0]
        idx_max_central.append(idx_max_central[0])

    return firerate_max_central

def gen_single_neuron_tuning(neuron,epoch,rule,H,loc_info,task_info):
    """Compute single neuron raw firing rate at all locations
    """
    single_neuron_firerate = []
    for loc in task_info[rule]['in_loc_set']:
        if loc in loc_info:
            firerate = H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1], \
                                   loc_info == loc, neuron].mean(axis=0)
        else:print('loc not in')
        firerate = firerate*50
        single_neuron_firerate.append(firerate.mean())
    return single_neuron_firerate

def fit_gaussion_curve(single_neuron_firerate,task_info,rule):
    """Fit single neuron firing rates at spatial tuning locations into gaussian curve
    """
    single_neuron_firerate = [x for x in single_neuron_firerate if str(x) != 'nan']
    location = np.arange(len(task_info[rule]['in_loc_set']))
    max_dir = single_neuron_firerate.index(max(single_neuron_firerate))
    firerate = np.array(single_neuron_firerate)
    tuning = max_central(max_dir,firerate)
    single_x = np.arange(len(tuning))
    tuning_idx = max_central(max_dir,location)
    gaussian_x = np.arange(-0.1, len(tuning) - 0.9, 0.1)
    paras, _ = curve_fit(gaussian, single_x, tuning + (-1) * np.min(tuning), \
                            p0=[np.max(tuning) + 1, len(tuning) // 2, 1])
    gaussian_y = gaussian(gaussian_x, paras[0], paras[1], paras[2]) - np.min(tuning) * (-1)
    # fit_y = gaussian(tuning_idx, paras[0], paras[1], paras[2]) - np.min(tuning) * (-1)

    # sorted_y = [x for _, x in sorted(zip(tuning_idx[:-1], fit_y[:-1]))]
    return gaussian_x, gaussian_y

def compute_fano_factor(values):
    """Compute fano factor = variance / mean
    """
    return np.var(values)/np.mean(values)

def gen_task_info(hp,model_dir,rules):
    """generate task infomation for each rule
    """
    task_info_file = model_dir + '/task_info.pkl'
    if os.path.isfile(task_info_file):
        with open(task_info_file, 'rb') as tinfr:
            task_info = pickle.load(tinfr)
    else:
        task_info = dict()

    trial_store = dict()

    print("Epoch information:")
    for rule in rules:
        task_info[rule] = dict()
        trial_store[rule] = generate_trials(rule, hp, 'test', noise_on=False)

        n_stims = len([ep for ep in trial_store[rule].epochs.keys() if 'stim' in ep])
        stim_loc_log_len = int(len(trial_store[rule].input_loc) / n_stims)
        task_info[rule]['in_loc'] = np.array([np.argmax(i) for i in trial_store[rule].input_loc[:stim_loc_log_len]])
        if n_stims != 1:
            for nst in range(2, n_stims + 1):
                task_info[rule]['in_loc_' + str(nst)] = \
                    np.array([np.argmax(i) for i in
                              trial_store[rule].input_loc[(nst - 1) * stim_loc_log_len:nst * stim_loc_log_len]])

        task_info[rule]['in_loc_set'] = sorted(set(task_info[rule]['in_loc']))
        task_info[rule]['epoch_info'] = trial_store[rule].epochs
        print('\t' + rule + ':')
        for e_name, e_time in task_info[rule]['epoch_info'].items():
            print('\t\t' + e_name + ':', e_time)

    with open(task_info_file, 'wb') as tinf:
        pickle.dump(task_info, tinf)
    return task_info

def find_epoch_best_location(rule,epoch,task_info,correct_H,correct_loc_info,significant_neuron):
    """Find best cue/delay location for each neuron
    """
    neuron_info = []
    for neuron in significant_neuron:
        firerate_list = []
        for loc in task_info[rule]['in_loc_set']:
            firerate = correct_H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1], \
                                   correct_loc_info == loc, neuron].mean(axis=0)*50
            firerate_list.append(firerate.mean())
        max_dir = firerate_list.index(max(firerate_list))
        neuron_info.append((neuron,max_dir))
    return neuron_info

def gen_single_neuron_tuning_median_last(neuron,epoch,rule,H,loc_info,task_info,time):
    """ Generate single neuron average raw firing rate and median for all locations (use last 1s delay)
    """
    real_par = 50
    single_neuron_firerate = []
    single_neuron_median = []
    for loc in task_info[rule]['in_loc_set']:
        single_neuron_median.append([])

    for loc in task_info[rule]['in_loc_set']:
        firerate = H[task_info[rule]['epoch_info'][epoch][1]-time:task_info[rule]['epoch_info'][epoch][1], \
                               loc_info == loc, neuron].mean(axis=0)
        firerate = firerate*real_par
        fix_level = H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                    loc_info == loc, neuron].mean(axis=1).mean(axis=0) * 50
        firerate_norm = firerate / fix_level - 1

        single_neuron_firerate.append(firerate.mean())
        if loc in loc_info:
            single_neuron_median[loc] = statistics.median(firerate)
    return single_neuron_firerate,single_neuron_median

def gen_single_neuron_tuning_median(neuron,epoch,rule,H,loc_info,task_info):
    """ Generate single neuron average raw firing rate and median for all locations
    """
    real_par = 50
    single_neuron_firerate = []
    single_neuron_median = []
    for loc in task_info[rule]['in_loc_set']:
        single_neuron_median.append([])

    for loc in task_info[rule]['in_loc_set']:
        firerate = H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1], \
                               loc_info == loc, neuron].mean(axis=0)

        firerate = firerate*real_par
        single_neuron_firerate.append(firerate.mean())
        if loc in loc_info:
            single_neuron_median[loc] = statistics.median(firerate)
    return single_neuron_firerate, single_neuron_median

def compute_gaussian_mean(single_neuron_firerate):
    """ Compute gaussian mean of single neuron
    :param single_neuron_firerate: firng rate of all locations
    :return: gaussian mean
    """
    max_dir = single_neuron_firerate.index(max(single_neuron_firerate))
    firerate = np.array(single_neuron_firerate)

    tuning = max_central(max_dir, firerate)
    single_x = np.arange(len(tuning))

    gaussian_x = np.arange(-0.1, len(tuning) - 0.9, 0.1)
    paras, _ = curve_fit(gaussian, single_x, tuning + (-1) * np.min(tuning), \
                         p0=[np.max(tuning) + 1, len(tuning) // 2, 1])
    gaussian_mean = paras[1]
    return gaussian_mean


