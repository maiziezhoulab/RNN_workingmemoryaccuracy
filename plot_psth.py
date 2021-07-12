# basic packages #
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pickle
import copy
from numpy import *
from copy import deepcopy

from utils.tools import mkdir_p, range_auto_select, auto_model_select
from utils import tools

from matplotlib import pyplot as plt
import matplotlib as mpl
from utils.functions import select_H_by_trialidx, compute_H_, select_neurons_excitatory, gen_task_info


# -------------find neuron best cue/delay location-------------
def gen_neuron_info(rule, epoch, significant_neuron, correct_H, loc_info, task_info, norm=True):
    neuron_info = []
    flag = False
    for neuron in significant_neuron:
        neuron_firerate_list = []
        for loc in task_info[rule]['in_loc_set']:
            fix_level = correct_H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                        loc_info == loc, neuron].mean(axis=0) * 50
            raw_firerate = correct_H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1], \
                           loc_info == loc, neuron].mean(axis=0) * 50
            norm_firerate = raw_firerate / fix_level.mean(axis=0) - 1
            if norm:
                firerate = norm_firerate
            else:
                firerate = raw_firerate
            neuron_firerate_list.append(firerate.mean())
        # print(neuron_firerate_list)
        try:
            neuron_max_index = neuron_firerate_list.index(np.nanmax(np.array(neuron_firerate_list)))
            neuron_info.append((neuron, neuron_max_index))
        except:
            flag = True
    return neuron_info, flag


def gen_psth_log(rule, epoch, neuron_info, H, loc_info, task_info, oppo_sel_dir=False, norm=True):
    # print('\tGenerating PSTH ' + rule + ' ' + epoch)

    psth_log = []
    for neuron in neuron_info:
        if oppo_sel_dir:
            loc = (neuron[1] + len(task_info[rule]['in_loc_set']) // 2) % len(
                task_info[rule]['in_loc_set'])  # oppo location
            if len(task_info[rule]['in_loc_set']) % 2:  # odd number of locations
                psth_temp = (H[:, loc_info == loc, neuron[0]].mean(axis=1) + \
                             H[:, loc_info == (loc + 1) % len(task_info[rule]['in_loc_set']),
                             neuron[0]].mean(axis=1)) * 50 / 2.0
            else:  # even number of locations
                psth_temp = H[:, loc_info == loc, neuron[0]].mean(axis=1) * 50
        else:
            loc = neuron[1]
            psth_temp = H[:, loc_info == loc, neuron[0]].mean(axis=1) * 50

        fix_level = H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                    loc_info == loc, neuron[0]].mean(axis=1).mean(axis=0) * 50
        psth_norm = psth_temp / fix_level - 1
        if norm:
            psth_log.append(psth_norm)
        else:
            psth_log.append(psth_temp)
    psth_log = np.nanmean(np.array(psth_log), axis=0)
    return psth_log


def correct_error_trials_split(all_H, perf, rule, task_info):
    """ Split correct and error trials
    """
    correct_idx = []
    error_idx = []
    for i in range(len(perf)):
        if perf[i]:
            correct_idx.append(i)
        else:
            error_idx.append(i)
    temp_task_info = deepcopy(task_info)
    all_in_loc_info = temp_task_info[rule]['in_loc']

    correct_H = select_H_by_trialidx(all_H, correct_idx)
    correct_loc_info = np.array(all_in_loc_info)[correct_idx]

    error_H = select_H_by_trialidx(all_H, error_idx)
    error_loc_info = np.array(all_in_loc_info)[error_idx]
    return correct_H, error_H, correct_loc_info, error_loc_info


def plot_multiple_model_psth(hp, log, model_dir, rule, epoch, model_select,task_info, neuron_name, plot_oppo_dir=False,norm=True, ):
    # print("Start ploting PSTH")
    # print("\trule: " + rule + " selective epoch: " + epoch)
    data_to_plot_correct = {}
    data_to_plot_error = {}
    data_types = ["psth", "neuron_num", "model_perf"]
    if plot_oppo_dir:
        data_types.append('psth_oppo')
    for key in model_select.keys():
        data_to_plot_correct[key] = {}
        data_to_plot_error[key] = {}
        for data_type in data_types:
            data_to_plot_correct[key][data_type] = []
            data_to_plot_error[key][data_type] = []

    for stage, values in model_select.items():
        for model in values:
            # print('stage: ', stage, 'model: ', model)
            all_H, perf, y_loc, y_hat_loc = compute_H_(hp, model_dir, rule, model, trial=None,task_mode='test')
            # split correct and error trials
            correct_H, error_H, correct_loc_info, error_loc_info = correct_error_trials_split(all_H, perf, rule,
                                                                                              task_info)
            # select significant neurons from correct trials
            if neuron_name == 'all_neuron':
                significant_neuron_all = np.arange(hp['n_rnn'])
                # print('# all_significant_neuron = ', len(significant_neuron_all))
                significant_neuron = significant_neuron_all
            if neuron_name == 'delay_neuron':
                significant_neuron_delay = select_neurons_excitatory(hp, rule, task_info, 'delay1', correct_H,
                                                                     correct_loc_info)
                # print('# delay_significant_neuron = ', len(significant_neuron_delay))
                significant_neuron = significant_neuron_delay
            if neuron_name == 'cue_neuron':
                significant_neuron_cue = select_neurons_excitatory(hp, rule, task_info, 'stim1', correct_H,
                                                                   correct_loc_info)
                print('# cue_significant_neuron = ', len(significant_neuron_cue))
                significant_neuron = significant_neuron_cue

            # compute model performance
            model_idx = log['trials'].index(model)
            model_perf = log['perf_' + rule][model_idx]
            if model_perf < 0.1:
                continue

            # find best cue/delay location in significant neuron
            neuron_epoch_info, flag = gen_neuron_info(rule, epoch, significant_neuron, correct_H, correct_loc_info,
                                                      task_info, norm=norm)
            # find correct/error trials psth
            psth_log_correct = gen_psth_log(rule, epoch, neuron_epoch_info, correct_H, correct_loc_info, task_info,
                                            oppo_sel_dir=False, norm=norm)
            psth_log_error = gen_psth_log(rule, epoch, neuron_epoch_info, error_H, error_loc_info, task_info,
                                          oppo_sel_dir=False, norm=norm)

            # if odrd task with opposite direction
            if plot_oppo_dir:
                psth_oppo_correct = gen_psth_log(rule, epoch, neuron_epoch_info, correct_H, correct_loc_info, task_info,
                                                 oppo_sel_dir=True, norm=norm)
                psth_oppo_error = gen_psth_log(rule, epoch, neuron_epoch_info, error_H, error_loc_info, task_info,
                                               oppo_sel_dir=True, norm=norm)

            # store data to plot
            data_to_plot_correct[stage]['psth'].append(psth_log_correct.tolist())
            data_to_plot_correct[stage]['neuron_num'].append(len(significant_neuron))
            data_to_plot_correct[stage]['model_perf'].append(model_perf)
            data_to_plot_error[stage]['psth'].append(psth_log_error.tolist())
            data_to_plot_error[stage]['neuron_num'].append(len(significant_neuron))
            data_to_plot_error[stage]['model_perf'].append(model_perf)
            if plot_oppo_dir:
                data_to_plot_correct[stage]['psth_oppo'].append(psth_oppo_correct)
                data_to_plot_error[stage]['psth_oppo'].append(psth_oppo_error)
    for m_key in data_to_plot_correct.keys():
        for data_type in data_types:
            data_to_plot_correct[m_key][data_type] = np.nanmean(np.array(data_to_plot_correct[m_key][data_type]),
                                                                axis=0)

    for m_key in data_to_plot_error.keys():
        for data_type in data_types:
            data_to_plot_error[m_key][data_type] = np.nanmean(np.array(data_to_plot_error[m_key][data_type]), axis=0)

    colors = {'mature': 'red', 'mid': 'blue', 'early': 'green'}
    title = 'Rule:' + rule + ' Epoch:' + epoch + ' Neuron_type:' + '_' + neuron_name
    if norm:
        save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/' + \
                    'PSTH_correct_error_multiple' + '/' + epoch + '/' + 'norm' + '/'
    else:
        save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/' + \
                    'PSTH_correct_error_multiple' + '/' + epoch + '/' + 'raw' + '/'

    # plot
    if plot_oppo_dir == False:
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.suptitle(title)
        for stage in data_to_plot_correct.keys():
            ax.plot(np.arange(len(data_to_plot_correct[stage]['psth'])) * hp['dt'] / 1000,
                    data_to_plot_correct[stage]['psth'], \
                    label='correct_' + str(stage) + '_' + str(data_to_plot_correct[stage]['model_perf']) + \
                          '_neuron_' + str(data_to_plot_correct[stage]['neuron_num']),
                    color=colors[stage])
            ax.plot(np.arange(len(data_to_plot_error[stage]['psth'])) * hp['dt'] / 1000,
                    data_to_plot_error[stage]['psth'], \
                    label='error_' + str(stage) + '_' + str(data_to_plot_error[stage]['model_perf']) + \
                          '_neuron_' + str(data_to_plot_error[stage]['neuron_num']),
                    color=colors[stage], linestyle='--')
            ax.set_xlabel("time/s")
            ax.set_ylabel("activity")
            ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        mpl.rcParams['pdf.fonttype'] = 42
        mkdir_p(save_path)
        plt.savefig(save_path + rule + '_' + epoch + '_PSTHnew_' + neuron_name + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    if plot_oppo_dir:
        fig, ax = plt.subplots(1, 2, figsize=(35, 10))
        fig.suptitle(title)

        y_correct_max_list = []
        y_correct_min_list = []
        y_error_max_list = []
        y_error_min_list = []

        for stage in data_to_plot_correct.keys():
            # plot correct trial on left figure
            ax[0].plot(np.arange(len(data_to_plot_correct[stage]['psth'])) * hp['dt'] / 1000,
                       data_to_plot_correct[stage]['psth'], \
                       label='correct_' + str(stage) + '_' + str(data_to_plot_correct[stage]['model_perf']),
                       color=colors[stage])
            if plot_oppo_dir:
                ax[0].plot(np.arange(len(data_to_plot_correct[stage]['psth_oppo'])) * hp['dt'] / 1000,
                           data_to_plot_correct[stage]['psth_oppo'], \
                           label='correct_' + str(stage) + '_' + str(data_to_plot_correct[stage]['model_perf']),
                           color=colors[stage], linestyle='--')
            y_correct_max_list.append(np.nanmax(np.array(data_to_plot_correct[stage]["psth"])))
            y_correct_min_list.append(np.nanmin(np.array(data_to_plot_correct[stage]["psth"])))
            y_correct_max_list.append(np.nanmax(np.array(data_to_plot_correct[stage]["psth_oppo"])))
            y_correct_min_list.append(np.nanmin(np.array(data_to_plot_correct[stage]["psth_oppo"])))
            # plot error trials on right figure
            ax[1].plot(np.arange(len(data_to_plot_error[stage]['psth'])) * hp['dt'] / 1000,
                       data_to_plot_error[stage]['psth'], \
                       label='error_' + str(stage) + '_' + str(data_to_plot_error[stage]['model_perf']),
                       color=colors[stage])
            if plot_oppo_dir:
                ax[1].plot(np.arange(len(data_to_plot_error[stage]['psth_oppo'])) * hp['dt'] / 1000,
                           data_to_plot_error[stage]['psth_oppo'], \
                           label='error_' + str(stage) + '_' + str(data_to_plot_error[stage]['model_perf']),
                           color=colors[stage], linestyle='--')
            y_error_max_list.append(np.nanmax(np.array(data_to_plot_error[stage]["psth"])))
            y_error_min_list.append(np.nanmin(np.array(data_to_plot_error[stage]["psth"])))
            y_error_max_list.append(np.nanmax(np.array(data_to_plot_error[stage]["psth_oppo"])))
            y_error_min_list.append(np.nanmin(np.array(data_to_plot_error[stage]["psth_oppo"])))

        y_max = np.nanmax(np.array([np.nanmax(np.array(y_correct_max_list)), np.nanmax(np.array(y_error_max_list))]))
        y_min = np.nanmin(np.array([np.nanmin(np.array(y_correct_min_list)), np.nanmin(np.array(y_error_min_list))]))
        ax[0].set_title('correct trials')
        ax[0].set_xlabel("time/s")
        ax[0].set_ylabel("activity")
        ax[0].legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        ax[0].set_ylim([y_min, y_max + 1])

        ax[1].set_title('error trials')
        ax[1].set_xlabel("time/s")
        ax[1].set_ylabel("activity")
        ax[1].legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        ax[1].set_ylim([y_min, y_max + 1])
        mpl.rcParams['pdf.fonttype'] = 42
        mkdir_p(save_path)
        plt.savefig(save_path + rule + '_' + epoch + '_PSTHnew_' + neuron_name + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    import argparse
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = 'data/6tasks_8loc_256neuron_odr3000_seed0'

    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    # model parameters
    rule = 'odr3000'
    epoch = 'delay1'

    # prepare task info
    task_info = gen_task_info(hp, model_dir, [rule])

    # select multiple models for mature, mid and early stage models
    model_select = auto_model_select(hp, log, smooth_window=9, perf_margin=0.05, max_model_num_limit=30)


    # ODR task with normalized firing rate for delay neurons
    plot_multiple_model_psth(hp, log, model_dir, rule, epoch, model_select[rule], task_info, 'delay_neuron', plot_oppo_dir=False,norm=True, )

    # ODRD task with normalized firing rate for delay neurons
    plot_multiple_model_psth(hp, log, model_dir, rule, epoch, model_select[rule], task_info, 'delay_neuron', plot_oppo_dir=True,
                             norm=True, )



