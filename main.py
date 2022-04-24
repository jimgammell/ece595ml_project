import os
import pickle
import argparse
import json
import random
import numpy as np
import torch
from train import run_trial

def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_initial_measure', dest='init_meas', action='store_false', default=True, help='Do not measure initial model performance prior to training.')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=None, help='Override the batch size in the .json files with this value.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=None, help='Override the batch size in the .json files with this value.')
    args = parser.parse_args()
    return args

def get_json_file_arguments(config_dir, config_name):
    with open(os.path.join(config_dir, config_name), 'r') as F:
        config_params = json.load(F)
    if 'results_dir' in config_params:
        results_dir = os.path.join('.', 'results', config_params['results_dir'])
        del config_params['results_dir']
    else:
        results_dir = os.path.join('.', 'results', config_name.split('.')[0])
    return results_dir, config_params

def main():
    cl_args = get_command_line_arguments()
    
    config_path = os.path.join('.', 'config')
    config_files = [f for f in os.listdir(config_path) if f.split('.')[-1]=='json']
    
    for config_file in config_files:
        results_dir, config_params = get_json_file_arguments(config_path, config_file)
        config_params['evaluate_initial_performance'] = cl_args.init_meas
        if cl_args.num_epochs != None:
            config_params['num_epochs'] = cl_args.num_epochs
        if cl_args.batch_size != None:
            config_params['dataloader_kwargs']['batch_size'] = cl_args.batch_size
        print('Preparing to run trial: {}'.format(config_file))
        print('\tResults directory: {}'.format(results_dir))
        print('\tConfig parameters:')
        for key in config_params.keys():
            val = config_params[key]
            print('\t\t{}: {}'.format(key, val))
        print()
        run_trial(config_params, results_dir)
        print('\n\n\n')

if __name__ == '__main__':
    main()