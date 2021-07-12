
from numpy import *
from matplotlib import pyplot as plt
import matplotlib as mpl

# sys.path.append('.')
from utils import tools
from utils.tools import mkdir_p
from utils.functions import split_trials, select_H_by_trialidx, gen_task_info
import numpy as np


def gen_single_neuron_PSTH_(hp, rule, correct_H, correct_loc_info, stage, model_perf,task_info,model_dir):
    # for neuron in range(hp['n_rnn']):
    for neuron in [4]:
        fig, ax = plt.subplots(3, 3, figsize=(14, 10))
        mpl.rcParams['pdf.fonttype'] = 42
        import numpy as np
        xtick = np.arange(11) * 0.5 # xlabels with every 0.5s
        for loc in task_info[rule]['in_loc_set']:
            psth_temp = correct_H[:, correct_loc_info == loc, neuron].mean(axis=1) * 50
            fix_level = correct_H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                        correct_loc_info == loc, neuron].mean(axis=1).mean(axis=0) * 50
            psth_temp = psth_temp / fix_level - 1
            psth_trial = correct_H[:, correct_loc_info == loc, neuron] * 50
            # compute normalized psth
            psth_trial = psth_trial / fix_level - 1
            num_trial = psth_trial.shape[1]
            # plot psth of every correct trial for each location
            if loc == 0:
                for i in range(num_trial):
                    ax[1, 2].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[1, 2].set_title(f'Location {loc+1}')
            if loc == 1:
                for i in range(num_trial):
                    ax[0, 2].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[0, 2].set_title(f'Location {loc+1}')
            if loc == 2:
                for i in range(num_trial):
                    ax[0, 1].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[0, 1].set_title(f'Location {loc+1}')
            if loc == 3:
                for i in range(num_trial):
                    ax[0, 0].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[0, 0].set_title(f'Location {loc+1}')
            if loc == 4:
                for i in range(num_trial):
                    ax[1, 0].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[1, 0].set_title(f'Location {loc+1}')
            if loc == 5:
                for i in range(num_trial):
                    ax[2, 0].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[2, 0].set_title(f'Location {loc+1}')
            if loc == 6:
                for i in range(num_trial):
                    ax[2, 1].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[2, 1].set_title(f'Location {loc+1}')
            if loc == 7:
                for i in range(num_trial):
                    ax[2, 2].plot(np.arange(len(psth_trial[:, i])) * hp['dt'] / 1000, psth_trial[:, i])
                    ax[2, 2].set_title(f'Location {loc+1}')
        y_max = []
        y_min = []
        for i in range(3):
            for j in range(3):
                y_min.append(ax[i, j].get_ylim()[0])
                y_max.append(ax[i, j].get_ylim()[1])
        y_max = max(y_max)
        y_min = min(y_min)
        for i in range(3):
            for j in range(3):
                ax[i, j].set_ylim([y_min, y_max])
        ax[1, 1].set_visible(False)
        plt.setp(ax, ylabel='Firing Rate')
        plt.setp(ax, xticks=xtick, xticklabels=xtick)
        plt.suptitle('single neuron_' + str(neuron) + ' all psth plot for 8 locations')
        mpl.rcParams['pdf.fonttype'] = 42
        save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/' + \
                    'single_neuron_psth' + '/' + 'all_psth_norm' + '/' + str(stage) + '___' + str(model_perf) + '/'
        mkdir_p(save_path)

        plt.savefig(save_path + rule + '_' + str(stage) + '_' + str(model_perf) + '_' + 'neuron_' + str(
            neuron) + '_all_psth_plot_8_locs (1s).pdf', bbox_inches='tight')
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
    stage = 'mature'
    rule = 'odr3000'

    # prepare task info
    task_info = gen_task_info(hp,model_dir,[rule])

    # select single model
    model_select_odr3000 = {'mature':748800,'mid':710400,'early':604160}

    models_select = model_select_odr3000
    model_idx = log['trials'].index(models_select[stage])
    model_perf = log['perf_' + rule][model_idx]
    print(stage + '_model_perf = ', model_perf)

    # split trials to correct trials, clockwise and counterclockwise trials
    all_H, correct_idx, correct_cw_idx, correct_ccw_idx, all_stage_keys, all_in_loc_info,theta_dev = split_trials(hp, log,\
    model_dir, rule,[models_select[stage]],task_info,trial=None,task_mode='test')

    # correct trials firing rate matrix (Time, Batch, Unit) and location information
    correct_H = select_H_by_trialidx(all_H, correct_idx)
    correct_loc_info = np.array(all_in_loc_info)[correct_idx]

    # plot PSTH for each neuron
    gen_single_neuron_PSTH_(hp, rule, correct_H, correct_loc_info, stage, model_perf,task_info,model_dir)





