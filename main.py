import os
import pickle
import argparse
import json

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

def main():
    cl_args = get_command_line_arguments()
    
    config_dir = os.path.join('.', 'config')
    if cl_args.config_files != None:
        config_files = cl_args['config_files']
    else:
        config_files = [f for f in os.listdir(config_path) if f.split('.')[-1] == 'json']
    
    for config_file in config_files:
        config_args = get_json_file_arguments(config_dir, config_file)
        
        # If no results directory is specified, use the name of the json file for the folder name.
        if results_dir == None:
            results_dir = os.path.join('.', config_file.split('.')[0]+'_{}'.format(idx))
        else:
            results_dir = os.path.join('.', results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        create_log_file(os.path.join(results_dir, 'log.txt'))
        
        # Multiple trials may be specified in a single config file.
        #  If so, break up into one set of arguments per trial.
        number_of_trials = 1
        for key in config_args.keys():
            if type(config_args[key]) == list:
                if number of trials == 1:
                    number_of_trials = len(config_args[key])
                elif number of trials == len(config_args[key]):
                    pass
                else:
                    assert False
        trial_args_list = [{} for _ in number_of_trials]
        for key in config_args.keys():
            if type(config_args[key]) == list:
                for (idx, item) in enumerate(list):
                    trial_args_list[idx].update({key: item})
            else:
                for idx in range(len(trial_args_list)):
                    trial_args_list[idx].update({key: config_args[key]})
        
        # Run and save the trials specified in the config file.
        for (idx, trial_args) in enumerate(trial_args_list):
            # Override number of epochs and batch size with command line arguments, if present.
            if cl_args.num_epochs != None:
                config_args['num_epochs'] = cl_args.num_epochs
            config_params['eval_initial_performance'] = (cl_args.init_meas and config_params['eval_initial_performance'])

            # Run the trial.
            print('Running trial {}'.format(trial_type))
            print('\tConfig file: {}'.format(os.path.join(config_dir, config_file)))
            print('\tResults directory: {}'.format(results_dir))
            if trial_type == 'imbalanced':
                from imbalanced_dataset_experiment import run_trial
            elif trial_type == 'noisy':
                from noisy_dataset_experiment import run_trial
            else:
                assert False
            results = run_trial(trial_args)
            
            # Save results from the trial. Append number to name corresponding to order in which this trial
            #  was defined in the config file.
            with open(os.path.join(results_dir, 'results_{}.pickle'.format(idx)), 'wb') as F:
                pickle.dump(results, F)