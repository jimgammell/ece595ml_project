import os
import pickle
import argparse
import json
from copy import copy
import torch

from utils import create_log_file, log_print as print

# Parse command line arguments.
#  These are redundant with the config file arguments but can be convenient for debugging.
def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files',
                        dest='config_files',
                        nargs='+',
                        default=None,
                        help='Specify which json files to use. If unspecified, the program will run all json files in the \'config_files\' directory.')
    parser.add_argument('--no_initial_measure',
                        dest='init_meas',
                        action='store_false',
                        default=True,
                        help='Do not measure initial performance prior to training. Overrides json file.')
    parser.add_argument('--num_epochs',
                        dest='num_epochs',
                        type=int,
                        default=None,
                        help='Specify number of epochs for which to train during each trial. Overrides json file.')
    args = parser.parse_args()
    return args

# Parse json file to get trial settings.
def get_json_file_arguments(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as F:
        config_params = json.load(F)
    return config_params

def expand_args(args_list):
    if type(args_list) != list:
        args_list = [args_list]
    for (idx, args) in enumerate(args_list):
        if any([type(item) == list for _, item in args.items()]):
            for key, item in args.items():
                if type(item) == list:
                    del args_list[idx]
                    for x in item:
                        args_copy = copy(args)
                        args_copy[key] = x
                        args_list.extend(expand_args(args_copy))
                    break
        else:
            break
    return args_list

def main():
    cl_args = get_command_line_arguments()
    
    config_dir = os.path.join('.', 'config')
    if cl_args.config_files != None:
        config_files = cl_args.config_files
    else:
        config_files = [f for f in os.listdir(config_dir) if f.split('.')[-1] == 'json']
    
    for config_file in config_files:
        config_args = get_json_file_arguments(config_dir, config_file)
        
        # If no results directory is specified, use the name of the json file for the folder name.
        if not 'results_dir' in config_args:
            results_dir = os.path.join('.', 'results', config_file.split('.')[0])
        else:
            results_dir = os.path.join('.', 'results', config_args['results_dir'])
        if not os.path.exists(os.path.join('.', 'results')):
            os.mkdir(os.path.join('.', 'results'))
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        create_log_file(os.path.join(results_dir, 'log.txt'))
        
        # Multiple trials may be specified in a single config file.
        #  If so, break up into one set of arguments per trial.
        trial_args_list = expand_args(config_args)
        
        # Run and save the trials specified in the config file.
        for (idx, trial_args) in enumerate(trial_args_list):
            # Override number of epochs and batch size with command line arguments, if present.
            if cl_args.num_epochs != None:
                config_args['num_epochs'] = cl_args.num_epochs
            config_args['eval_initial_performance'] = (cl_args.init_meas and config_args['eval_initial_performance'])

            # Run the trial.
            print('Running {} trial'.format(config_args['trial_type']))
            print('\tConfig file: {}'.format(os.path.join(config_dir, config_file)))
            print('\tResults directory: {}'.format(results_dir))
            if config_args['trial_type'] == 'false_positive':
                from false_positive_experiment import run_trial
            elif config_args['trial_type'] == 'noisy':
                from noisy_dataset_experiment import run_trial
            elif config_args['trial_type'] == 'imbalanced':
                from imbalanced_dataset_experiment import run_trial
            else:
                assert False
            results, model = run_trial(trial_args)
            
            # Save results from the trial. Append number to name corresponding to order in which this trial
            #  was defined in the config file.
            with open(os.path.join(results_dir, 'results_{}.pickle'.format(idx)), 'wb') as F:
                pickle.dump(results, F)
            torch.save(model.state_dict(), os.path.join(results_dir, 'model_{}.pickle'.format(idx)))

if __name__ == '__main__':
    main()