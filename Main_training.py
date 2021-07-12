from training.train import train

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument('--modeldir', type=str, default='data/6tasks')
    parser.add_argument('--modeldir', type=str, default='data/6tasks')
    parser.add_argument('--randseed', type=int, default=0)
    parser.add_argument('--continue_after_target', default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    hp = {# number of units each ring
          'n_eachring': 8,
          # number of rings/modalities
          'num_ring': 1,
          'activation': 'softplus',
          'n_rnn': 256,
          'learning_rate': 0.001,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': False,
          'target_perf': 0.995,
          'mature_target_perf': 0.95,
          'mid_target_perf': 0.65,
          'early_target_perf': 0.35,}

    train(args.modeldir,
        seed=args.randseed,
        hp=hp,
        ruleset='all_new',
        rule_trains=['overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20,
        continue_after_target_reached=args.continue_after_target)