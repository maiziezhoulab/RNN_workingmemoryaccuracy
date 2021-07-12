import sys
from numpy import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
sys.path.append('..')
from utils import tools
from utils.tools import mkdir_p
from utils.functions import select_neurons_excitatory,gen_task_info,select_H_by_trialidx,compute_fano_factor
from utils.functions import fit_gaussion_curve, split_trials,max_central

def gen_single_neuron_tuning_fano_last(neuron,epoch,rule,H,loc_info,task_info,time):
    real_par = 50
    single_neuron_firerate = []
    fano_factors = []
    for loc in task_info[rule]['in_loc_set']:
        firerate = H[task_info[rule]['epoch_info'][epoch][1]-time:task_info[rule]['epoch_info'][epoch][1], \
                               loc_info == loc, neuron].mean(axis=0)*real_par
        fix_level = H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                    loc_info == loc, neuron].mean(axis=1).mean(axis=0) * 50
        firerate_norm = firerate / fix_level - 1
        fano = compute_fano_factor(np.array(firerate))
        single_neuron_firerate.append(firerate_norm.mean())
        fano_factors.append(fano)
    return single_neuron_firerate,fano_factors

def plot_fano_tuning_average_last(correct_H,correct_loc_info,significant_neuron,rule,epoch,task_info,model_perf,stage,neuron_name,time):
    location = np.arange(len(task_info[rule]['in_loc_set']))
    all_neuron_firerate = []
    all_neuron_fano_factors = []

    for loc in range(len(task_info[rule]['in_loc_set'])+1):
        all_neuron_firerate.append([])
        all_neuron_fano_factors.append([])

    for neuron in significant_neuron:
        plot_fano_factors = []

        # compute normalized firing rate and fano factors
        single_neuron_firerate,fano_factors = gen_single_neuron_tuning_fano_last(neuron, epoch, rule, correct_H, correct_loc_info, task_info,time)

        # find best location
        max_dir = single_neuron_firerate.index(np.nanmax(np.array(single_neuron_firerate)))

        # find spatial tuning location center
        tuning = max_central(max_dir, single_neuron_firerate)
        tuning_idx = max_central(max_dir, location)

        # reordering based on spatial tuning locations
        for idx in tuning_idx:
            plot_fano_factors.append(fano_factors[int(idx)])
        for i in range(len(tuning)):
            if str(tuning[i]) != 'nan':
                all_neuron_firerate[i].append(tuning[i])
                all_neuron_fano_factors[i].append(plot_fano_factors[i])
    all_neuron_firerate = [np.mean(x) for x in all_neuron_firerate]
    all_neuron_fano_factors = [np.mean(x) for x in all_neuron_fano_factors]

    if stage == 'mature':
        color = 'red'
        mark = 'or'
    elif stage == 'mid':
        color = 'blue'
        mark = 'ob'
    elif stage == 'early':
        color = 'green'
        mark = 'og'

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))

    # fit firing rate to gaussian curve
    try:
        gaussian_x, gaussian_y = fit_gaussion_curve(all_neuron_firerate)
        ax[0].plot(gaussian_x, gaussian_y, color=color, linestyle='--')
        ax[0].plot(np.arange(len(task_info[rule]['in_loc_set'])+1), all_neuron_firerate, mark)
    except Exception:
        ax[0].plot(np.arange(len(task_info[rule]['in_loc_set'])+1), all_neuron_firerate, color=color, linestyle='--', marker='o')
    ax[0].set_xticks(np.arange(len(task_info[rule]['in_loc_set'])+1))
    ax[0].set_title('Tuning curve for ' + neuron_name + ' (' + str(len(significant_neuron))+')')
    ax[0].set_xlabel('Location')
    ax[0].set_ylabel('Firing Rate')
    ax[1].plot(np.arange(len(task_info[rule]['in_loc_set'])+1),np.array(all_neuron_fano_factors),color=color,linestyle='--',marker='o')
    ax[1].set_title('Fano factor for each location of '+ neuron_name +' ('+str(len(significant_neuron))+')')
    ax[1].set_xticks(np.arange(len(task_info[rule]['in_loc_set'])+1))
    ax[1].set_xlabel('Location')
    ax[1].set_ylabel('Fano Factor')
    mpl.rcParams['pdf.fonttype'] = 42
    # save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/' + epoch + '/' +\
    #             'fano_factor_new_model_last_'+str(time)+'/'+ 'averaged'+'/'
    # mkdir_p(save_path)
    #
    # plt.savefig(save_path + rule + '_' + epoch + '_' + str(model_perf) + '_'+str(stage) +'_'+neuron_name+'_'+str(len(significant_neuron))+'_fano_factor_last_'+str(time)+'.pdf', bbox_inches='tight')
    # # plt.savefig(save_path + rule + '_' + epoch + '_' + str(model_perf) + '_mature_all_neurons_' + str(neuron) + '_fano_factor (20ms).png', bbox_inches='tight')
    # plt.close()
    plt.show()


if __name__ == '__main__':
        import argparse
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        model_dir = 'data/6tasks_8loc_256neuron_odr3000_seed0'

        hp = tools.load_hp(model_dir)
        log = tools.load_log(model_dir)

        # model parameters
        stage = 'mature'
        rule = 'odr3000'
        epoch = 'delay1'

        # prepare task info
        task_info = gen_task_info(hp,model_dir,[rule])

        # select one model
        model_select_odr3000 = {'mature':748800,'mid':710400,'early':604160}

        models_select = model_select_odr3000

        model_idx = log['trials'].index(models_select[stage])
        model_perf = log['perf_' + rule][model_idx]

        # split trials to correct trials, clockwise and counterclockwise trials
        all_H, correct_idx, correct_cw_idx, correct_ccw_idx, all_stage_keys, all_in_loc_info, dev_theta = split_trials(hp,log, model_dir, rule, [models_select[stage]], task_info, trial=None, task_mode='test-3000')

        # correct trials firing rate matrix (Time, Batch, Unit) and location information
        correct_H = select_H_by_trialidx(all_H,correct_idx)
        correct_loc_info = np.array(all_in_loc_info)[correct_idx]

        # select delay neurons
        significant_neuron_delay = select_neurons_excitatory(hp,rule,task_info,epoch,correct_H,correct_loc_info)
        print('# delay_significant_neuron = ', len(significant_neuron_delay))

        # plot tuning curve and fano factor plot (use last 1s delay to plot) for delay neurons
        time_50 = 50 # last 1s
        plot_fano_tuning_average_last(correct_H, correct_loc_info, significant_neuron_delay, rule, epoch, task_info,
                                      model_perf, stage, 'delay_neuron', time_50)

