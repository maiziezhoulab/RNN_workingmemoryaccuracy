import os

from analysis.PSTH_print_basic_info import print_basic_info

# basic packages #
import numpy as np

import sys

sys.path.append('.')
from utils import tools

# plot #
import matplotlib.pyplot as plt
import matplotlib as mpl

# DL & task #
import tensorflow as tf
from task_and_network.task import generate_trials
from task_and_network.network import Model
from utils.functions import get_perf
from utils.tools import mkdir_p

def saccade_distribut_analysis(hp, log, rule, model_dir, trial_list, ):
    early_saccade_dir_error = list()
    mid_saccade_dir_error = list()
    mature_saccade_dir_error = list()
    early_saccade_dir_correct = list()
    mid_saccade_dir_correct = list()
    mature_saccade_dir_correct = list()

    is_dict = False
    is_list = False
    if isinstance(trial_list, dict):
        temp_list = list()
        is_dict = True
        for value in trial_list[rule].values():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(trial_list, list):
        temp_list = trial_list
        is_list = True

    for trial_num in temp_list:
        saccade_dir_temp_error = list()
        saccade_dir_temp_correct = list()

        # reload model
        model = Model(model_dir + '/' + str(trial_num) + '/', hp=hp)
        with tf.Session() as sess:
            model.restore()
            trial = generate_trials(rule, hp, 'test')
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            y_hat = sess.run(model.y_hat, feed_dict=feed_dict)
            perf, dist,_ ,_ = get_perf(y_hat, trial.y_loc)
            for i in range(len(dist)):
                if perf[i] == 0:  # error trials
                    saccade_dir_temp_error.append(dist[i])
                else:  # correct trials
                    saccade_dir_temp_correct.append(dist[i])

        matur = log['perf_' + rule][trial_num // log['trials'][1]]

        if (is_list and matur > hp['mid_target_perf']) or (is_dict and trial_num in trial_list[rule]['mature']):
            mature_saccade_dir_error += saccade_dir_temp_error
            mature_saccade_dir_correct += saccade_dir_temp_correct
        elif (is_list and matur > hp['early_target_perf']) or (is_dict and trial_num in trial_list[rule]['mid']):
            mid_saccade_dir_error += saccade_dir_temp_error
            mid_saccade_dir_correct += saccade_dir_temp_correct
        elif is_list or (is_dict and trial_num in trial_list[rule]['early']):
            early_saccade_dir_error += saccade_dir_temp_error
            early_saccade_dir_correct += saccade_dir_temp_correct

    # plot correct and error trials saccade distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mpl.rcParams['pdf.fonttype'] = 42
    axes[0, 0].hist(early_saccade_dir_error, bins=30, range=(0, 180), histtype="stepfilled", alpha=0.6, color="green")
    axes[0, 0].set_title("early")
    axes[0, 1].hist(mid_saccade_dir_error, bins=30, range=(0, 180), histtype="stepfilled", alpha=0.6, color="blue")
    axes[0, 1].set_title("mid")
    axes[0, 2].hist(mature_saccade_dir_error, bins=30, range=(0, 180), histtype="stepfilled", alpha=0.6, color="red")
    axes[0, 2].set_title("mature")

    axes[1, 0].hist(early_saccade_dir_correct, bins=30, range=(0, 180), histtype="stepfilled", alpha=0.6, color="green")
    axes[1, 0].set_title("early")
    axes[1, 1].hist(mid_saccade_dir_correct, bins=30, range=(0, 180), histtype="stepfilled", alpha=0.6, color="blue")
    axes[1, 1].set_title("mid")
    axes[1, 2].hist(mature_saccade_dir_correct, bins=30, range=(0, 180), histtype="stepfilled", alpha=0.6, color="red")
    axes[1, 2].set_title("mature")

    y_error_max = max([axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1], axes[0, 2].get_ylim()[1]])

    y_correct_max = max([axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1], axes[1, 2].get_ylim()[1]])

    axes[0, 0].set_ylim([0, y_error_max])
    axes[0, 1].set_ylim([0, y_error_max])
    axes[0, 2].set_ylim([0, y_error_max])
    axes[1, 0].set_ylim([0, y_correct_max])
    axes[1, 1].set_ylim([0, y_correct_max])
    axes[1, 2].set_ylim([0, y_correct_max])

    for j in range(2):
        for i in range(3):
            axes[j, i].set_xlabel("distance to stim1($\degree$)")
    fig.suptitle("saccade distribut analysis for error (upper) and correct (lower) trials")

    save_folder = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/'
    mkdir_p(save_folder)
    save_pic = save_folder + 'saccade_distribut_analysis_by_growth_new'
    # mkdir_p(save_folder)
    plt.savefig(save_pic + '.png', transparent=False)
    plt.savefig(save_pic + '.eps', transparent=False)
    plt.savefig(save_pic + '.pdf', transparent=False)
    plt.show()
    plt.close(fig)



def combined_trial_range(trial_range, task):
    combined_range = []
    stage = ['early', 'mid', 'mature']
    for i in stage:
        combined_range += trial_range[task][i]
    return combined_range


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = 'data/6tasks_8loc_256neuron_odr3000_seed0'

    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    # model parameters
    rule = 'odr3000'

    # select multiple models
    trial_range = print_basic_info(hp, log, model_dir, smooth_growth=True, smooth_window=5, auto_range_select=True)


    # plot saccade distribution for correct and error trials
    saccade_distribut_analysis(hp, log, rule, model_dir, trial_list=combined_trial_range(trial_range, rule), )
