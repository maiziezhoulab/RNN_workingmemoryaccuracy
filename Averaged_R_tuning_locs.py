from numpy import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
# sys.path.append('.')
from utils import tools
from scipy.stats.stats import pearsonr
from utils.tools import mkdir_p
import statistics
from utils.functions import split_trials,select_H_by_trialidx,max_central,select_neurons_excitatory,gen_task_info,gen_single_neuron_tuning_median_last

def gen_single_neuron_tuning_median_last(neuron,epoch,rule,H,loc_info,task_info,time):
    ''' Generate single neuron raw firing rate and median firing rate at all location
    '''
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

def plot_R_last(correct_idx,correct_cw_idx,correct_ccw_idx, epoch, rule, correct_H, correct_loc_info,task_info,significant_neuron,dev_theta,model_perf,stage,neuron_name,time):
    real_par = 50
    import numpy as np
    location = np.arange(len(task_info[rule]['in_loc_set']))
    r_list = []
    tuning_dev_fr = []
    tuning_dev_theta = []
    for i in range(len(task_info[rule]['in_loc_set'])+1):
        tuning_dev_theta.append([])
        tuning_dev_fr.append([])
    for neuron in significant_neuron:
        neuron_dev_fr = []
        neuron_dev_theta = []
        neuron_tuning_dev_fr = []
        neuron_tuning_dev_theta = []
        for loc in task_info[rule]['in_loc_set']:
            neuron_dev_fr.append([])
            neuron_dev_theta.append([])
            neuron_tuning_dev_fr.append([])
            neuron_tuning_dev_theta.append([])
        # compute single neuron average firing rate of 8 locations and their median
        single_neuron_firerate, single_neuron_median = gen_single_neuron_tuning_median_last(neuron, epoch, rule, correct_H,\
                                                                                       correct_loc_info, task_info,time)
        # compute best epoch location
        max_dir = single_neuron_firerate.index(np.nanmax(np.array(single_neuron_firerate)))
        tuning_loc = max_central(max_dir, location)
        max_dir_idx = tuning_loc.tolist().index(max_dir)
        for i, trial in enumerate(correct_idx):
            current_loc = correct_loc_info[i]
            actual_fr_current_loc = correct_H[
                                    task_info[rule]['epoch_info'][epoch][1]-time:task_info[rule]['epoch_info'][epoch][1], \
                                    i, neuron].mean(axis=0)
            actual_fr_current_loc = actual_fr_current_loc*real_par
            neuron_dev_fr[trial // 16].append(actual_fr_current_loc - single_neuron_median[trial // 16])
            current_loc_idx = tuning_loc.tolist().index(current_loc) # find current loc index in the tuning locations

            # decide tuning drift (sign of theta_dev)
            if current_loc_idx == 0 or current_loc_idx == len(task_info[rule]['in_loc_set']):
                neuron_dev_theta[trial//16].append(dev_theta[i])
            elif current_loc_idx == max_dir_idx:
                neuron_dev_theta[trial//16].append(-dev_theta[i])
            elif current_loc_idx < max_dir_idx and trial in correct_ccw_idx or current_loc_idx > max_dir_idx and trial in correct_cw_idx:
                neuron_dev_theta[trial//16].append(dev_theta[i])
            elif current_loc_idx < max_dir_idx and trial in correct_cw_idx or \
                    current_loc_idx > max_dir_idx and trial in correct_ccw_idx:
                neuron_dev_theta[trial//16].append(-dev_theta[i])

        # reordering locations based on spatial tuning locations
        for idx,loc in enumerate(tuning_loc):
            tuning_dev_fr[idx].append(np.array(neuron_dev_fr[int(loc)]).mean())
            tuning_dev_theta[idx].append(np.array(neuron_dev_theta[int(loc)]).mean())
    # compute pearson correlation for each location
    for theta,fr in zip(tuning_dev_theta,tuning_dev_fr):
        theta = [x for x in theta if str(x) != 'nan']
        fr = [x for x in fr if str(x) != 'nan']
        r,_ = pearsonr(theta,fr)
        r_list.append(r)

    if stage == 'mature':
        color = 'red'
    elif stage == 'mid':
        color = 'blue'
    elif stage == 'early':
        color = 'green'

    plt.plot(r_list, color=color, linestyle='--',marker='o')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.xlabel('Tuning Locations')
    plt.title(str(stage)+' stage pearson correlation coeffiencts at spatial tuning locations (last 1s)')
    mpl.rcParams['pdf.fonttype'] = 42
    # save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/' + rule + '/' + epoch + '/' +\
    #             'Averaged_R_8_locs_new_last_1s'+'/'+ str(stage) +'_'+ str(model_perf) +'/'
    # mkdir_p(save_path)
    #
    # plt.savefig(save_path + rule + '_' + epoch + '_' + str(model_perf)+ '_'+str(stage)+'_'+neuron_name+'_'+str(len(significant_neuron))+'_Averaged_R_tuning_locs_last_1s_ai.pdf', bbox_inches='tight')
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
        epoch = 'delay1' # delay period

        # prepare task info
        task_info = gen_task_info(hp,model_dir,[rule])

        # select one model and compute model performance
        model_select_odr3000 = {'mature':748800,'mid':710400,'early':604160}
        models_select = model_select_odr3000
        model_idx = log['trials'].index(models_select[stage])
        model_perf = log['perf_' + rule][model_idx]
        # find correct trials and split them into clockwise and counterclockwise trials
        all_H, correct_idx, correct_cw_idx, correct_ccw_idx, all_stage_keys, all_in_loc_info, dev_theta = split_trials(hp,log, model_dir, rule, [models_select[stage]], task_info, trial=None, task_mode='test')

        # correct trials firing rate matrix (Time, Batch, Unit) and location information
        correct_H = select_H_by_trialidx(all_H,correct_idx)
        correct_loc_info = np.array(all_in_loc_info)[correct_idx]
        # Neurons selection
        significant_neuron_all = np.arange(hp['n_rnn'])
        print('# all_significant_neuron = ', len(significant_neuron_all))
        significant_neuron_delay = select_neurons_excitatory(hp, rule, task_info, epoch, correct_H, correct_loc_info)
        print('# delay_significant_neuron = ', len(significant_neuron_delay))

        # plot averaged R of tuning locations
        time_50 = 50 # use last 1s delay to plot
        plot_R_last(correct_idx, correct_cw_idx, correct_ccw_idx, epoch, rule, correct_H, correct_loc_info,
                            task_info, significant_neuron_delay, dev_theta,model_perf,stage,'delay_neurons',time_50)



        #

