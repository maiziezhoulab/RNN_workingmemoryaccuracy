import sys
import os
import numpy as np
import tensorflow as tf
import numpy as np
import pickle
import copy
from numpy import *
from copy import deepcopy
from matplotlib import pyplot as plt
sys.path.append('.')
from utils import tools
from scipy.stats.stats import pearsonr
from utils.tools import mkdir_p,smooth
import matplotlib as mpl
from utils.functions import split_trials,max_central,select_H_by_trialidx,select_neurons_excitatory,gen_task_info,gen_single_neuron_tuning_median_last

def R_distribute_last(correct_idx,correct_cw_idx,correct_ccw_idx,epoch, rule, correct_H, correct_loc_info,task_info,significant_neuron,dev_theta,model_perf,neuron_name,time,stage):

    signed_dev_theta_list = []
    dev_fr_list = []
    location = np.arange(len(task_info[rule]['in_loc_set']))
    r_list = []
    for neuron in significant_neuron:
        neuron_dev_fr = []
        neuron_dev_theta = []
        for i, trial in enumerate(correct_idx):
            # compute single neuron average firing rate of 8 locations and their median
            single_neuron_firerate,single_neuron_median = gen_single_neuron_tuning_median_last(neuron, epoch,rule, correct_H,correct_loc_info, task_info,time)

            # find neuron best location
            max_dir = single_neuron_firerate.index(np.nanmax(np.array(single_neuron_firerate)))

            # reording locations based on spatial tuning locations
            tuning_loc = max_central(max_dir, location)
            current_loc = correct_loc_info[i]

            # compoute actural raw firing rate for current trial location (use last 1s delay)
            actual_fr_current_loc = correct_H[task_info[rule]['epoch_info'][epoch][1]-time:task_info[rule]['epoch_info'][epoch][1], \
                               i, neuron].mean(axis=0)*50

            max_dir_idx = tuning_loc.tolist().index(max_dir)
            current_loc_idx = tuning_loc.tolist().index(current_loc)

            # decide tuning drift (sign of theta_dev)
            if current_loc_idx == 0 or current_loc_idx == len(task_info[rule]['in_loc_set']) or current_loc_idx == max_dir_idx:
                signed_dev_theta_list.append('nan')
                dev_fr_list.append(actual_fr_current_loc - single_neuron_median[trial // 16])
            if current_loc_idx < max_dir_idx and trial in correct_ccw_idx or current_loc_idx > max_dir_idx and trial in correct_cw_idx:
                signed_dev_theta_list.append(dev_theta[i])
                neuron_dev_theta.append(dev_theta[i])

                dev_fr_list.append(actual_fr_current_loc - single_neuron_median[trial // 16])
                neuron_dev_fr.append(actual_fr_current_loc - single_neuron_median[trial // 16])
            if current_loc_idx < max_dir_idx and trial in correct_cw_idx or \
                    current_loc_idx > max_dir_idx and trial in correct_ccw_idx:
                signed_dev_theta_list.append(-dev_theta[i])
                neuron_dev_theta.append(-dev_theta[i])

                dev_fr_list.append(actual_fr_current_loc - single_neuron_median[trial // 16])
                neuron_dev_fr.append(actual_fr_current_loc - single_neuron_median[trial // 16])

        # compute pearson correlation for neuron
        r,_ = pearsonr(neuron_dev_theta, neuron_dev_fr)
        r_list.append(r)

    r_mean = np.mean(np.array(r_list))

    # plot histogram of pearson correlation distribution
    plt.hist(r_list,bins=30)
    plt.axvline(r_mean, color='k', linestyle='--')
    plt.xlabel('R')
    plt.title(str(stage)+' stage '+neuron_name+' pearson correlation coefficient_mean = '+ str(r_mean))
    mpl.rcParams['pdf.fonttype'] = 42
    # save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/' + epoch + '/' +\
    #             'saccade_deviation_new'+'/'+'R_distribution(1s)_last_1s'+'/'+ str(stage) +'/'
    # mkdir_p(save_path)
    # plt.savefig(save_path + rule + '_' + epoch + '_' + str(model_perf) + '_'+str(stage)+'_'+neuron_name+'_'+str(len(significant_neuron))+'_saccade_dev_R_distribution_last_1s_norm.pdf', bbox_inches='tight')
    # plt.close()
    plt.show()

if __name__ == '__main__':
        import argparse
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        model_dir = 'data/6tasks_8loc_256neuron_odr3000_seed0'

        hp = tools.load_hp(model_dir)
        log = tools.load_log(model_dir)

        # Model parameters
        stage = 'mature'
        rule = 'odr3000'
        epoch = 'delay1'

        # Prepare task info
        task_info = gen_task_info(hp, model_dir, [rule])

        # Select one model
        model_select_odr3000 = {'mature':748800,'mid':710400,'early':604160}

        models_select = model_select_odr3000

        model_idx = log['trials'].index(models_select[stage])
        model_perf = log['perf_' + rule][model_idx]

        # Split trials to correct trials, clockwise and counterclockwise trials
        all_H, correct_idx, correct_cw_idx, correct_ccw_idx, all_stage_keys, all_in_loc_info, dev_theta = split_trials(hp,log, model_dir, rule, [models_select[stage]], task_info, trial=None, task_mode='test')

        # Correct trials firing rate matrix (Time, Batch, Unit) and location information
        correct_H = select_H_by_trialidx(all_H,correct_idx)
        correct_loc_info = np.array(all_in_loc_info)[correct_idx]

        # Select delay neurons
        significant_neuron_delay = select_neurons_excitatory(hp,rule,task_info,epoch,correct_H,correct_loc_info)
        print('# delay_significant_neuron = ', len(significant_neuron_delay))

        # Plot pearson correlation coefficients distribution of saccade dev vs theta dev for last 1s delay
        time_50 = 50
        R_distribute_last(correct_idx,correct_cw_idx, correct_ccw_idx, epoch, rule, correct_H, correct_loc_info,
                            task_info, significant_neuron_delay,dev_theta,model_perf,'delay_neuron',time_50,stage)

        #

