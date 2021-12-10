# Largely based on clustering.py and variance.py
import os
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from matplotlib import pyplot as plt
from collections import OrderedDict

import sys

sys.path.append('.')
from task_and_network.network import Model
from task_and_network.task import generate_trials
from utils import tools
import csv
import pickle
from utils.functions import gen_task_info,select_H_by_trialidx,split_trials

kelly_colors = \
    [np.array([0.94901961, 0.95294118, 0.95686275]),
     np.array([0.13333333, 0.13333333, 0.13333333]),
     np.array([0.95294118, 0.76470588, 0.]),
     np.array([0.52941176, 0.3372549, 0.57254902]),
     np.array([0.95294118, 0.51764706, 0.]),
     np.array([0.63137255, 0.79215686, 0.94509804]),
     np.array([0.74509804, 0., 0.19607843]),
     np.array([0.76078431, 0.69803922, 0.50196078]),
     np.array([0.51764706, 0.51764706, 0.50980392]),
     np.array([0., 0.53333333, 0.3372549]),
     np.array([0.90196078, 0.56078431, 0.6745098]),
     np.array([0., 0.40392157, 0.64705882]),
     np.array([0.97647059, 0.57647059, 0.4745098]),
     np.array([0.37647059, 0.30588235, 0.59215686]),
     np.array([0.96470588, 0.65098039, 0.]),
     np.array([0.70196078, 0.26666667, 0.42352941]),
     np.array([0.8627451, 0.82745098, 0.]),
     np.array([0.53333333, 0.17647059, 0.09019608]),
     np.array([0.55294118, 0.71372549, 0.]),
     np.array([0.39607843, 0.27058824, 0.13333333]),
     np.array([0.88627451, 0.34509804, 0.13333333]),
     np.array([0.16862745, 0.23921569, 0.14901961]), ]

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

def compute_task_variance(model, sess, rules=None, data_type="rule"):
    # Largely based on variance.py _compute_variance_bymodel func
    """Compute variance for all tasks.

        Args:
            model: network.Model instance
            sess: tensorflow session
            rules: list of rules to compute variance, list of strings
        """
    h_all_byrule = OrderedDict()
    h_all_byepoch = OrderedDict()
    hp = model.hp

    if rules is None:
        rules = hp['rules']

    n_hidden = hp['n_rnn']

    for rule in rules:
        if rule == 'odr_mix_uniform_00_30_01step':
            trial = generate_trials(rule, hp, 'test-3000', noise_on=False)
        else:
            trial = generate_trials(rule, hp, 'test', noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h = sess.run(model.h, feed_dict=feed_dict)

        for e_name, e_time in trial.epochs.items():
            if 'fix' not in e_name:  # Ignore fixation period
                h_all_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1], :, :]

        # Ignore fixation period
        h_all_byrule[rule] = h[trial.epochs['fix1'][1]:, :, :]

    # Reorder h_all_byepoch by epoch-first
    keys = list(h_all_byepoch.keys())
    # ind_key_sort = np.lexsort(zip(*keys))
    # Using mergesort because it is stable
    ind_key_sort = np.argsort(list(zip(*keys))[1], kind='mergesort')
    h_all_byepoch = OrderedDict(
        [(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

    # for data_type in ['rule', 'epoch']:
    if data_type == 'rule':
        h_all = h_all_byrule
    elif data_type == 'epoch':
        h_all = h_all_byepoch
    else:
        raise ValueError

    h_var_all = np.zeros((n_hidden, len(h_all.keys())))
    for i, val in enumerate(h_all.values()):
        # val is Time, Batch, Units
        # Variance across time and stimulus
        # h_var_all[:, i] = val[t_start:].reshape((-1, n_hidden)).var(axis=0)
        # Variance acros stimulus, then averaged across time
        h_var_all[:, i] = val.var(axis=1).mean(axis=0)

    result = {'h_var_all': h_var_all, 'keys': list(h_all.keys()), 'rules': rules}

    return result


def clustering_by_taskvariance(hp, data_type, res, normalization_method='max'):
    # Largely based on clustering.py Class Analysis

    # If not computed, use variance.py
    # fname = os.path.join(model_dir, 'variance_' + data_type + '.pkl')
    # res = tools.load_pickle(fname)
    h_var_all_ = res['h_var_all']
    keys = res['keys']

    # First only get active units. Total variance across tasks larger than 1e-3
    # ind_active = np.where(h_var_all_.sum(axis=1) > 1e-2)[0]
    ind_active = np.where(h_var_all_.sum(axis=1) > 1e-3)[0]
    h_var_all = h_var_all_[ind_active, :]

    # Normalize by the total variance across tasks
    if normalization_method == 'sum':
        h_normvar_all = (h_var_all.T / np.sum(h_var_all, axis=1)).T
    elif normalization_method == 'max':
        h_normvar_all = (h_var_all.T / np.max(h_var_all, axis=1)).T
    elif normalization_method == 'none':
        h_normvar_all = h_var_all
    else:
        raise NotImplementedError()

    ################################## Clustering ################################
    from sklearn import metrics
    X = h_normvar_all

    # Clustering
    from sklearn.cluster import AgglomerativeClustering, KMeans

    # Choose number of clusters that maximize silhouette score
    n_clusters = range(2, 30)
    scores = list()
    labels_list = list()
    for n_cluster in n_clusters:
        # clustering = AgglomerativeClustering(n_cluster, affinity='cosine', linkage='average')
        clustering = KMeans(n_cluster, algorithm='full', n_init=20, random_state=0)
        clustering.fit(X)  # n_samples, n_features = n_units, n_rules/n_epochs
        labels = clustering.labels_  # cluster labels

        score = metrics.silhouette_score(X, labels)

        scores.append(score)
        labels_list.append(labels)

    scores = np.array(scores)

    # Heuristic elbow method
    # Choose the number of cluster when Silhouette score first falls
    # Choose the number of cluster when Silhouette score is maximum
    if data_type == 'rule':
        # i = np.where((scores[1:]-scores[:-1])<0)[0][0]
        i = np.argmax(scores)
    else:
        # The more rigorous method doesn't work well in this case
        # i = n_clusters.index(10)
        i = np.argmax(scores)  #########add by yichen
        # i = n_clusters.index(4)

    labels = labels_list[i]
    n_cluster = n_clusters[i]
    print('Choosing {:d} clusters'.format(n_cluster))

    # Sort clusters by its task preference (important for consistency across nets)
    if data_type == 'rule':
        label_prefs = [np.argmax(h_normvar_all[labels == l].sum(axis=0)) for l in set(labels)]
    elif data_type == 'epoch':
        ## TODO: this may no longer work!
        # label_prefs = [keys[np.argmax(h_normvar_all[labels==l].sum(axis=0))][0] for l in set(labels)]
        label_prefs = [np.argmax(h_normvar_all[labels == l].sum(axis=0)) for l in set(labels)]  #########add by yichen

    ind_label_sort = np.argsort(label_prefs)
    label_prefs = np.array(label_prefs)[ind_label_sort]
    # Relabel
    labels2 = np.zeros_like(labels)
    for i, ind in enumerate(ind_label_sort):
        labels2[labels == ind] = i
    labels = labels2

    ###############################add by yichen############################################
    # adjust the cluster order#
    freq = dict()
    for i in labels:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1
    tmp_seq = sorted(freq.items(), key=lambda item: item[1])  # sort by frequency ,low to high
    relabel_dict = dict()
    for i in range(len(tmp_seq)):
        relabel_dict[tmp_seq[i][0]] = i
    relabel = []
    for i in labels:
        relabel.append(relabel_dict[i])
    labels = relabel
    labels = np.array(labels)
    ########################################################################################

    ind_sort = np.argsort(labels)

    labels = labels[ind_sort]

    cluster_result = dict()
    cluster_result["h_normvar_all"] = h_normvar_all[ind_sort, :]
    cluster_result["ind_active"] = ind_active[ind_sort]

    cluster_result["n_clusters"] = n_clusters
    cluster_result["scores"] = scores
    cluster_result["n_cluster"] = n_cluster

    cluster_result["h_var_all"] = h_var_all
    cluster_result["normalization_method"] = normalization_method
    cluster_result["labels"] = labels
    cluster_result["unique_labels"] = np.unique(labels)

    # cluster_result["model_dir"] = model_dir
    cluster_result["hp"] = hp
    cluster_result["data_type"] = data_type
    cluster_result["rules"] = res['rules']

    return cluster_result


def plot_connectivity(hp, weight_info, model_dir, trial_num, cluster_result=None):
    # Largely based on clustering.py plot

    nx = hp['n_input']
    ny = hp['n_output']
    nh = hp['n_rnn']
    nr = hp['n_eachring']
    rnum = hp['num_ring']
    nrule = len(hp['rules'])

    b_rec = weight_info["b_rec"][:, np.newaxis]
    b_out = weight_info["b_out"][:, np.newaxis]
    w_rec = weight_info["w_rec"]
    w_in = weight_info["w_in"]
    w_out = weight_info["w_out"]

    if cluster_result is not None:
        #ind_active = cluster_result["ind_active"]
        ind_active = np.arange(256)  # plot without cluster
        _w_rec = w_rec[ind_active, :][:, ind_active]
        _w_in = w_in[ind_active, :]
        _w_out = w_out[:, ind_active]
        _b_rec = b_rec[ind_active, :]

        ############################add by Yuanqi#####################################################################
        sort_by = 'best_epoch_loc' # sort by best epoch location
        if sort_by == 'w_in':
            w_in_mod_all = _w_in[:, 1:nr + 1]
            for ir in range(1, rnum):
                w_in_mod_all += _w_in[:, 1 + ir * nr:(ir + 1) * nr + 1]
            w_prefs = np.argmax(w_in_mod_all, axis=1)
        elif sort_by == 'w_out':
            w_prefs = np.argmax(_w_out[1:], axis=0)
        elif sort_by == 'best_epoch_loc':
            w_prefs = np.array(best_loc)[ind_active]

        # sort by labels then by prefs
        # ind_sort = np.lexsort((w_prefs, cluster_result["labels"]))
        ind_sort = np.lexsort((w_prefs, np.zeros_like(w_prefs))) # plot without cluster

        w_rec = _w_rec[ind_sort, :][:, ind_sort]
        w_in = _w_in[ind_sort, :]
        w_out = _w_out[:, ind_sort]
        b_rec = _b_rec[ind_sort, :]
        # labels = cluster_result["labels"][ind_sort]
        labels = np.zeros_like(w_prefs)[ind_sort] # plot without cluster

    l = 0.465
    l0 = (1 - 1.5 * l) / nh

    plot_infos = [(w_rec              , [l               ,l          ,nh*l0    ,nh*l0]),
                  (w_in[:,[0]]        , [l-(nx+5+rnum*3+4)*l0    ,l          ,1*l0     ,nh*l0]),] # Fixation input
                  #5: between win&wrec 4:1(fixation)+3(space)

    for r_ in range(rnum):
        plot_infos.append((w_in[:,r_*nr+1:(r_+1)*nr+1], [l-(nx+5+(rnum-r_)*3-nr*r_)*l0 ,l  ,nr*l0  ,nh*l0])) # Ring input

    plot_infos += [(w_in[:,rnum*nr+1:]    , [l-(nx-rnum*nr+5)*l0,l       ,nrule*l0 ,nh*l0]), # Rule inputs
                   (w_out[[0],:]          , [l               ,l-(4)*l0   ,nh*l0    ,1*l0]),
                   (w_out[1:,:]           , [l               ,l-(ny+6)*l0,nh*l0    ,(ny-1)*l0]),
                   (b_rec                 , [l+(nh+6)*l0     ,l          ,l0       ,nh*l0]),
                   (b_out                 , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0       ,ny*l0])]

    fig = plt.figure(figsize=(18,16))
    print([l, l, nh * l0, nh * l0])
    cax = fig.add_axes([l+nh * l0+0.03, l, 0.01,nh * l0 ])
    cmap = 'coolwarm'
    im_list = []
    for plot_info in plot_infos:
        ax = fig.add_axes(plot_info[1])
        vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5, 50, 95])
        bar_min_ = vmid - (vmax - vmin) / 2
        bar_max_ = vmid + (vmax - vmin) / 2

        im = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='equal',
                           vmin=bar_min_, vmax=bar_max_)

        ax.axis('off')
        im_list.append(im)

    plt.colorbar(im_list[0], cax=cax)
    plt.show()
    # save_pic
    # save_path = 'figure/figure_' + model_dir.rstrip('/').split('/')[-1] + '/synaptic_analysis/'
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # save_name = save_path + "synaptic_analysis_trial_num-" + str(trial_num)

    # save input weights for further distribution plot
    # with open(save_name + "w_in" + str(trial_num) + "_3s.pkl", "wb") as my_pickle:
    #     pickle.dump(plot_infos[2][0], my_pickle)



def synaptic_analysis(hp, model_dir, trial_num, cluster_analysis=False, cluster_type="rule", rules=None):
    weight_info = dict()
    model = Model(model_dir + '/' + str(trial_num), hp=hp)
    cluster_result = None

    with tf.Session() as sess:
        model.restore()
        weight_info["w_in"] = sess.run(model.w_in).T
        weight_info["w_rec"] = sess.run(model.w_rec).T
        weight_info["w_out"] = sess.run(model.w_out).T
        weight_info["b_rec"] = sess.run(model.b_rec)
        weight_info["b_out"] = sess.run(model.b_out)

        if cluster_analysis:
            result = compute_task_variance(model, sess, rules=rules, data_type=cluster_type)
            cluster_result = clustering_by_taskvariance(hp, cluster_type, result, normalization_method='max')

    plot_connectivity(hp, weight_info, model_dir, trial_num, cluster_result=cluster_result)



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # odr 3s delay with 8 locations
    model_dir = '../data/6tasks_8loc_256neuron_odr3000_seed0'
    # odr 3s delay with 360 locations
    # model_dir = '../data/6tasks_360loc_256neuron_odr3000_seed0'

    # load parameters and log
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    # select single model
    model_select_odr3000 = {'mature':748800,'mid':719360,'early':686080} # odr 3s delay with 8 locations

    # model_select_odr3000 = {'mature':907520,'mid':696320,'early':407040} # ord 3s delay with 360 locations

    stage = 'mature'
    rule = 'odr3000'
    epoch = 'stim1'
    models_select = model_select_odr3000
    task_info = gen_task_info(hp, model_dir, [rule])
    # find correct trials and split them into clockwise and counterclockwise trials
    all_H, correct_idx, correct_cw_idx, correct_ccw_idx, all_stage_keys, all_in_loc_info, dev_theta = split_trials(hp,
                                                                                                                   log,
                                                                                                                   model_dir,
                                                                                                                   rule,
                                                                                                                   [
                                                                                                                       models_select[
                                                                                                                           stage]],
                                                                                                                   task_info,
                                                                                                                   trial=None,
                                                                                                                   task_mode='test')
    # correct trials firing rate matrix (Time, Batch, Unit) and location information
    correct_H = select_H_by_trialidx(all_H, correct_idx)
    correct_loc_info = np.array(all_in_loc_info)[correct_idx]

    #  select all neurons
    significant_neuron_all = np.arange(hp['n_rnn'])
    #
    norm = True
    # find best cue location for each neuron for cue period
    best_cue_neuron_info = find_best_epoch_location(rule, epoch, task_info, correct_H, correct_loc_info,
                                                    significant_neuron_all, norm=norm)
    best_loc = []
    for each_info in best_cue_neuron_info:
        best_loc.append(each_info[1])

    # plot weight matrix sorted by best epoch location
    synaptic_analysis(hp, model_dir, models_select[stage], cluster_analysis=True, cluster_type="rule", rules=None)

