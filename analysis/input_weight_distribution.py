import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
def find_w_rec(w_rec):
    w_rec_list = w_rec.tolist()
    rec_list = []
    pos = 0
    neg = 0
    for each in w_rec_list:
        for i in each:
            rec_list.append(i)
            if i >= 0:
                pos += 1
            else: neg += 1
    return rec_list,pos,neg

def plot_distribution():
    # 3s_fix ------------
    mature_filename = 'figure/figure_6tasks_8loc_256neuron_odr3000_seed0/synaptic_analysis_trial_num-748800w_in748800_3s.pkl'
    mid_filename = 'figure/figure_6tasks_8loc_256neuron_odr3000_seed0/synaptic_analysis_trial_num-719360w_in719360_3s.pkl'
    early_filename = 'figure/figure_6tasks_8loc_256neuron_odr3000_seed0/synaptic_analysis_trial_num-686080w_in686080_3s.pkl'

    mature_filename_360 = 'figure/figure_6tasks_360loc_256neuron/synaptic_analysis_trial_num-907520w_in907520_3s.pkl'
    mid_filename_360 = 'figure/figure_6tasks_360loc_256neuron/synaptic_analysis_trial_num-696320w_in696320_3s.pkl'
    early_filename_360 = 'figure/figure_6tasks_360loc_256neuron/synaptic_analysis_trial_num-407040w_in407040_3s.pkl'

    with open(mature_filename, 'rb') as handle:
        mature_rec_list_ = pickle.load(handle)
        mature_rec_list,mature_pos,mature_neg = find_w_rec(mature_rec_list_)
        mature_mean = np.mean(np.array(mature_rec_list))
    with open(mid_filename, 'rb') as handle:
        mid_rec_list_ = pickle.load(handle)
        mid_rec_list,mid_pos,mid_neg = find_w_rec(mid_rec_list_)
        mid_mean = np.mean(np.array(mid_rec_list))
    with open(early_filename, 'rb') as handle:
        early_rec_list_ = pickle.load(handle)
        early_rec_list,early_pos,early_neg = find_w_rec(early_rec_list_)
        early_mean = np.mean(np.array(early_rec_list))

    with open(mature_filename_360, 'rb') as handle:
        mature_rec_list_360_ = pickle.load(handle)
        mature_rec_list_360,mature_pos_360,mature_neg_360 = find_w_rec(mature_rec_list_360_)
        mature_mean_360 = np.mean(np.array(mature_rec_list_360))
    with open(mid_filename_360, 'rb') as handle:
        mid_rec_list_360_ = pickle.load(handle)
        mid_rec_list_360,mid_pos_360,mid_neg_360 = find_w_rec(mid_rec_list_360_)
        mid_mean_360 = np.mean(np.array(mid_rec_list_360))
    with open(early_filename_360, 'rb') as handle:
        early_rec_list_360_ = pickle.load(handle)
        early_rec_list_360,early_pos_360,early_neg_360 = find_w_rec(early_rec_list_360_)
        early_mean_360 = np.mean(np.array(early_rec_list_360))

    fig, ax = plt.subplots(figsize=(22.0, 16.0), nrows=2, ncols=3)
    x_min_all = -3
    x_max_all = 3
    ax[0,2].hist(mature_rec_list,bins=50,range=[x_min_all, x_max_all],color='red')
    ax[0,2].axvline(0, color='k', linestyle='--')
    ax[0,2].set_title(" mean_w = %.4f" % mature_mean +' num = '+str(mature_neg)+'|'+str(mature_pos),fontsize=18)
    ax[0,1].hist(mid_rec_list,bins=50,range=[x_min_all, x_max_all],color='blue')
    ax[0,1].axvline(0, color='k', linestyle='--')
    ax[0,1].set_title(" mean_w = %.4f" % mid_mean+' num = '+str(mid_neg)+'|'+str(mid_pos),fontsize=18)
    ax[0,0].hist(early_rec_list,bins=50,range=[x_min_all, x_max_all],color='green')
    ax[0,0].axvline(0, color='k', linestyle='--')
    ax[0,0].set_title(" mean_w =  %.4f" % early_mean+' num = '+str(early_neg)+'|'+str(early_pos),fontsize=18)
    ax[0,0].set_ylabel('3s fix delay + 8 locations',fontsize=18)



    ax[1,2].hist(mature_rec_list_360,bins=50,range=[x_min_all, x_max_all],color='red')
    ax[1,2].axvline(0, color='k', linestyle='--')
    ax[1,2].set_title(" mean_w = %.4f" % mature_mean_360+' num = '+str(mature_neg_360)+'|'+str(mature_pos_360),fontsize=18)
    ax[1,1].hist(mid_rec_list_360,bins=50,range=[x_min_all, x_max_all],color='blue')
    ax[1,1].axvline(0, color='k', linestyle='--')
    ax[1,1].set_title(" mean_w = %.4f" % mid_mean_360+' num = '+str(mid_neg_360)+'|'+str(mid_pos_360),fontsize=18)
    ax[1,0].hist(early_rec_list_360,bins=50,range=[x_min_all, x_max_all],color='green')
    ax[1,0].axvline(0, color='k', linestyle='--')
    ax[1,0].set_title( " mean_w =  %.4f" % early_mean_360+' num = '+str(early_neg_360)+'|'+str(early_pos_360),fontsize=18)
    ax[1,0].set_ylabel('3s fix delay + 360 locations',fontsize=18)

    import matplotlib as mpl

    mpl.rcParams['pdf.fonttype'] = 42

    save_name = 'weight_matrix_distribution_cluster_all_3s_fix_in_own_scale'
    #
    for save_format in ['pdf']:
        plt.savefig(save_name + '.' + save_format)
    plt.show()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"








