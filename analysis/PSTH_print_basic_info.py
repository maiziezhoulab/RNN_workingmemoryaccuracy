import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils import tools

def print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5,auto_range_select=False,
                    avr_window=9,perf_margin=0.05,max_trial_num_limit=30):

    # print('rule trained: ', hp['rule_trains'])
    # print('minimum trial number: 0')
    # print('maximum trial number: ', log['trials'][-1])
    # print('minimum trial step  : ', log['trials'][1])
    # print('total number        : ', len(log['trials']))

    # fig_pref = plt.figure(figsize=(12,9))
    #############################
    if auto_range_select:
        trial_selected = dict()
    #############################
    for rule in hp['rule_trains']:
        if smooth_growth:
            growth = tools.smooth(log['perf_'+rule],smooth_window)
        else:
            growth = log['perf_'+rule]

        # plt.plot(log['trials'], growth, label = rule)
        
        ################################################################
        if auto_range_select:
            trial_selected[rule] = tools.range_auto_select(hp,log,log['perf_'+rule],\
                avr_window=avr_window,perf_margin=perf_margin,max_trial_num_limit=max_trial_num_limit)
            # for m_c in [('early','green'),('mid','blue'),('mature','red')]:
            #     plt.fill_between(log['trials'], growth, where=[i in trial_selected[rule][m_c[0]] for i in log['trials']],\
            #         facecolor=m_c[1],alpha=0.3)
        ################################################################

    # tools.mkdir_p('figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/')
    #
    # plt.xlabel("trial trained")
    # plt.ylabel("perf")
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.title('Growth of Performance')
    # save_name = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/growth_of_performance_range'
    # plt.tight_layout()
    # plt.savefig(save_name+'.png', transparent=False, bbox_inches='tight')
    # plt.savefig(save_name+'.pdf', transparent=False, bbox_inches='tight')
    # plt.savefig(save_name+'.eps', transparent=False, bbox_inches='tight')
    # plt.show()
    #########################
    if auto_range_select:
        return trial_selected
    #########################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', type=str, default='data/6tasks')
    args = parser.parse_args()

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    print_basic_info(hp,log,model_dir)