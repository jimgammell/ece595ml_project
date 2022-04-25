import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pickle
import os
import datasets
import models
import corrupted_dataset_trial

def run_trial(config_params, results_dir):
    random.seed(config_params['seed'])
    np.random.seed(config_params['seed'])
    torch.manual_seed(config_params['seed'])
    
    if type(config_params['dataset']) != list:
        config_params['dataset'] = [config_params['dataset']]
    if type(config_params['method']) != list:
        config_params['method'] = [config_params['method']]
    if type(config_params['trial_kwargs']) != list:
        config_params['trial_kwargs'] = [config_params['trial_kwargs']]
        
    for cp_dataset in config_params['dataset']:
        for cp_method in config_params['method']:
            for (tk_idx, trial_kwargs) in enumerate(config_params['trial_kwargs']):
                print('Running trial.')
                print('\tMethod: {}'.format(cp_method))
                print('\tDataset: {}'.format(cp_dataset))
                print('\tTrial kwargs {}: {}'.format(tk_idx, trial_kwargs))

                base_train_dataset, base_test_dataset = datasets.get_dataset(cp_dataset)
                dataset_constructor = getattr(corrupted_dataset_trial, config_params['trial_type'])
                train_dataset = dataset_constructor(base_train_dataset, **trial_kwargs)
                if cp_method == 'ltrwe':
                    validation_dataset = train_dataset.get_validation_dataset(config_params['num_validation_samples'])
                test_dataset = dataset_constructor(base_test_dataset, **config_params['trial_kwargs'])
                train_dataloader = DataLoader(train_dataset, **config_params['dataloader_kwargs'])
                if cp_method == 'ltrwe':
                    val_dataloader = DataLoader(validation_dataset, **config_params['dataloader_kwargs'])
                if 'shuffle' in config_params['dataloader_kwargs']:
                    config_params['dataloader_kwargs']['shuffle'] = False
                test_dataloader = DataLoader(test_dataset, **config_params['dataloader_kwargs'])

                model_constructor = getattr(models, config_params['model'])
                eg_input = next(iter(test_dataloader))[0]
                model = model_constructor(eg_input.shape, **config_params['model_kwargs'])
                model = model.to(config_params['device'])

                loss_fn_constructor = getattr(nn, config_params['loss_fn'])
                loss_fn = loss_fn_constructor(**config_params['loss_fn_kwargs'])
                loss_fn.reduction = 'none'

                optimizer_constructor = getattr(optim, config_params['optimizer'])
                optimizer = optimizer_constructor(model.parameters(), **config_params['optimizer_kwargs'])

                trial_constructor = getattr(corrupted_dataset_trial, config_params['trial_type']+'Trial')
                trial_kwargs = {'method': cp_method,
                                'train_dataloader': train_dataloader,
                                'test_dataloader': test_dataloader,
                                'model': model,
                                'loss_fn': loss_fn,
                                'optimizer': optimizer,
                                'device': config_params['device'],
                                'num_epochs': config_params['num_epochs'],
                                'batch_size': config_params['dataloader_kwargs']['batch_size'],
                                'evaluate_initial_performance': config_params['evaluate_initial_performance']}
                if cp_method == 'ltrwe':
                    trial_kwargs.update({'val_dataloader': val_dataloader})
                trial = trial_constructor(**trial_kwargs)
                results = trial()

                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)
                results_to_save = {}
                for key in results.keys():
                    results_to_save.update({key: results.get_traces(key)})
                with open(os.path.join(results_dir, 'results_{}_{}_{}.pickle'.format(cp_dataset, cp_method, tk_idx)), 'wb') as F:
                    pickle.dump(results_to_save, F)