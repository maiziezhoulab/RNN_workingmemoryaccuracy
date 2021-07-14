
import os
import numpy as np
import tensorflow as tf
import pickle
import sys
from numpy import *
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib as mpl
sys.path.append('.')
from utils import tools
from utils.tools import mkdir_p,smooth
from utils.functions import split_trials, gen_task_info, select_H_by_trialidx

def find_best_epoch_location(rule,epoch,task_info,correct_H,correct_loc_info,significant_neuron,norm=False):
    neuron_info = []
    for neuron in significant_neuron:
        firerate_list = []
        for loc in task_info[rule]['in_loc_set']:
            firerate = correct_H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1], \
                                   correct_loc_info == loc, neuron].mean(axis=0)*50

            fix_level = correct_H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                        correct_loc_info == loc, neuron].mean(axis=1).mean(axis=0) * 50
            firerate_norm = firerate / fix_level - 1
            if norm:
                firerate_list.append(firerate_norm.mean())
            else:
                firerate_list.append(firerate.mean())
        # compute best epoch location
        max_dir = firerate_list.index(np.nanmax(np.array(firerate_list)))

        # compute best epoch location normalized average firing rate
        neuron_trials_fr = correct_H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1], \
                           correct_loc_info == max_dir, neuron].mean(axis=1) * 50
        neuron_fix_level = correct_H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                           correct_loc_info == max_dir, neuron].mean(axis=1).mean(axis=0) * 50
        neuron_trials_fr_norm = neuron_trials_fr/neuron_fix_level -1
        if norm:
            best_epoch_fr = np.nanmax(np.array(neuron_trials_fr_norm))
        else:
            best_epoch_fr = np.nanmax(np.array(neuron_trials_fr))
        # store neuron information
        # (neuron, best epoch location, best epoch location normalized firing rate)
        neuron_info.append((neuron,max_dir,best_epoch_fr))
    return neuron_info


def gen_multiple_trials_heatmap(rule, epoch, task_info, all_H, correct_cw_idx,correct_ccw_idx,stage,neuron_info,model_dir,loc_num,model_perf,norm=False):
    # sort neuron by best epoch location
    neuron_info.sort(key=lambda x: x[1])

    # choose one location = loc_num
    correct_cw_idx_one_loc = [x for x in correct_cw_idx if 16*loc_num <= x <= (loc_num+1)*16-1]
    correct_ccw_idx_one_loc = [x for x in correct_ccw_idx if 16*loc_num <= x <= (loc_num+1)*16-1]
    correct_idx_one_loc = correct_cw_idx_one_loc + correct_ccw_idx_one_loc

    correct_H_one_loc = select_H_by_trialidx(all_H, correct_idx_one_loc)

    heatmap = []
    ytick = []
    for neuron in neuron_info:
        neuron_psth = correct_H_one_loc[:, :, neuron[0]].mean(axis=1) * 50
        fix_level = correct_H_one_loc[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                    :, neuron[0]].mean(axis=1).mean(axis=0)*50
        neuron_psth_norm = neuron_psth / fix_level - 1
        # ------- neuron normalized fr divide by neuron best epoch fr
        if norm:
            heatmap.append(neuron_psth_norm/neuron[2])
        else:
            heatmap.append(neuron_psth/neuron[2])
        ytick.append(str(neuron[0])+'_'+str(neuron[1]))
    # print('fff')
    # plot heatmap
    import seaborn as sns
    fig, ax = plt.subplots(1, 1, figsize=(45, 40))
    sns.heatmap(heatmap, cmap=sns.diverging_palette(240, 10, n=30),vmin=0, vmax=1)
    ax.set_title(str(stage)+' all trials heatmap at location 90')
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.arange(175))
    ax.set_xticklabels(np.arange(0, 3.5, 0.02))
    ax.set_ylabel('Sorted Neurons')
    ax.set_yticks(np.arange(256))
    ax.set_yticklabels(ytick)

    n = 25  # Keeps every 25th label
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    mpl.rcParams['pdf.fonttype'] = 42

    # save
    save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[
        -1] + '/' + rule + '/' + 'heatmap_' + '/'+ 'multiple_trials_loc'+str(loc_num) +'/'+str(stage)+'/'
    mkdir_p(save_path)
    plt.savefig(save_path + rule + '_' + str(model_perf) + '_'+str(epoch)+ '_'  + 'All_neurons_best_epoch_norm' + '_all_trials_heatmap_at_location_'+str(loc_num)+'.pdf',
                bbox_inches='tight')
    plt.show()
    plt.close()





if __name__ == '__main__':
        import argparse
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # ============ 6tasks_360loc_256neuron==============
        model_dir = 'data/6tasks_360loc_256neuron'
        hp = tools.load_hp(model_dir)
        log = tools.load_log(model_dir)

        # model parameters
        stage = 'mid'
        rule = 'odr'
        norm = True

        # prepare task info
        task_info = gen_task_info(hp,model_dir,[rule])

        # select one model
        model_select_odr = {'mature': 153600, 'mid': 70400, 'early':49920 }
        models_select = model_select_odr
        model_idx = log['trials'].index(models_select[stage])
        model_perf = log['perf_'+rule][model_idx]

        # find correct trials and split them into clockwise and counterclockwise trials
        all_H, correct_idx, correct_cw_idx, correct_ccw_idx, all_stage_keys, all_in_loc_info, dev_theta = split_trials(hp,log, model_dir, rule, [models_select[stage]], task_info, trial=None, task_mode='test')

        # correct trials firing rate matrix (Time, Batch, Unit) and location information
        correct_H = select_H_by_trialidx(all_H,correct_idx)
        correct_loc_info = np.array(all_in_loc_info)[correct_idx]

        # select all neurons
        significant_neuron_all = np.arange(hp['n_rnn'])
        print('# all_significant_neuron = ', len(significant_neuron_all))
        significant_neuron = significant_neuron_all

        # find best cue location for each neuron for cue period
        best_cue_neuron_info = find_best_epoch_location(rule,'stim1',task_info,correct_H,correct_loc_info,significant_neuron_all,norm=norm)

        # plot heatmap with neurons sorted by best cue location
        loc_num = 90
        gen_multiple_trials_heatmap(rule, 'stim1', task_info, all_H, correct_cw_idx,correct_ccw_idx, stage, best_cue_neuron_info, model_dir,loc_num,model_perf,norm=norm)



