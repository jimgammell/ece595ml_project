# To do:
#   Change metrics so we are tracking the number of nonzero samples which are correct and incorrect

import time
import random
from copy import deepcopy
import os
import numpy as np
from tqdm import tqdm as _tqdm
def tqdm(*args, **kwargs): return _tqdm(*args, ncols=50, **kwargs)
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from utils import set_random_seed, log_print as print
from datasets import get_dataset, RepetitiveDataset, extract_random_class_balanced_dataset, BinaryTargetDataset
from results import Results, mean
from train import sss_train_on_batch, ltrwe_train_on_batch, naive_train_on_batch
import models

def run_trial(config_params):
    trial_start_time = time.time()
    
    # Parse and validate trial_args
    method = config_params['method']
    assert method in ['naive', 'ltrwe', 'sss']
    dataset = config_params['dataset']
    assert dataset in ['CIFAR10', 'MNIST', 'FashionMNIST']
    total_samples = config_params['total_samples']
    majority_class = config_params['majority_class']
    minority_class = config_params['minority_class']
    minority_prop_of_majority = config_params['minority_prop_of_majority']
    if 'seed' in config_params:
        seed = config_params['seed']
    else:
        seed = time.time_ns() & 0xFFFFFFFF
    assert type(seed) == int
    train_dataloader_kwargs = config_params['train_dataloader_kwargs']
    assert type(train_dataloader_kwargs) == dict
    eval_dataloader_kwargs = config_params['eval_dataloader_kwargs']
    assert type(eval_dataloader_kwargs) == dict
    clean_dataloader_kwargs = config_params['clean_dataloader_kwargs']
    assert type(clean_dataloader_kwargs) == dict
    clean_dataset_samples_per_class = config_params['clean_samples_per_class']
    val_samples_per_class = config_params['val_samples_per_class']
    model_constructor_attr = config_params['model_constructor']
    assert model_constructor_attr in ['LeNet5', 'ResNet', 'AlexNet']
    if 'model_constructor_kwargs' in config_params:
        model_constructor_kwargs = config_params['model_constructor_kwargs']
    else:
        model_constructor_kwargs = {}
    assert type(model_constructor_kwargs) == dict
    loss_fn_constructor_attr = config_params['loss_fn_constructor']
    assert type(loss_fn_constructor_attr) == str
    if 'loss_fn_constructor_kwargs' in config_params:
        loss_fn_kwargs = config_params['loss_fn_constructor_kwargs']
    else:
        loss_fn_kwargs = {}
    assert type(loss_fn_kwargs) == dict
    optimizer_constructor_attr = config_params['optimizer_constructor']
    assert type(optimizer_constructor_attr) == str
    if 'optimizer_constructor_kwargs' in config_params:
        optimizer_kwargs = config_params['optimizer_constructor_kwargs']
    else:
        optimizer_kwargs = {}
    assert type(optimizer_kwargs) == dict
    if 'scheduler' in config_params:
        scheduler_constructor_attr = config_params['scheduler']
        scheduler_kwargs = config_params['scheduler_kwargs']
    else:
        scheduler_constructor_attr = None
    num_epochs = config_params['num_epochs']
    assert (type(num_epochs) == int) and (num_epochs >= 0)
    finetune_epochs = config_params['finetune_epochs']
    pretrain_epochs = config_params['pretrain_epochs']
    if 'device' in config_params:
        device = config_params['device']
    else:
        device = 'cpu'
    assert device in ['cpu', 'cuda']
    init_meas = config_params['eval_initial_performance']
    assert type(init_meas) == bool
    
    print('Beginning false positive dataset experiment.')
    print('\tMethod: {}'.format(method))
    print('\tDataset: {}'.format(dataset))
    print('\tTotal samples: {}'.format(total_samples))
    print('\tMajority class: {}'.format(majority_class))
    print('\tMinority class: {}'.format(minority_class))
    print('\tProportion of minority to majority samples: {}'.format(minority_prop_of_majority))
    print('\tRandom seed: {}'.format(seed))
    print('\tTraining dataloader kwargs: {}'.format(train_dataloader_kwargs))
    print('\tEval dataloader kwargs: {}'.format(eval_dataloader_kwargs))
    print('\tClean dataloader kwargs: {}'.format(clean_dataloader_kwargs))
    print('\tClean samples per class: {}'.format(clean_dataset_samples_per_class))
    print('\tValidation samples per class: {}'.format(val_samples_per_class))
    print('\tModel constructor: {}'.format(model_constructor_attr))
    print('\tModel constructor kwargs: {}'.format(model_constructor_kwargs))
    print('\tLoss function constructor: {}'.format(loss_fn_constructor_attr))
    print('\tLoss function constructor kwargs: {}'.format(loss_fn_kwargs))
    print('\tOptimizer constructor: {}'.format(optimizer_constructor_attr))
    print('\tOptimizer constructor kwargs: {}'.format(optimizer_kwargs))
    if scheduler_constructor_attr != None:
        print('\tScheduler constructor: {}'.format(scheduler_constructor_attr))
        print('\tScheduler constructor kwargs: {}'.format(scheduler_kwargs))
    print('\tNumber of epochs: {}'.format(num_epochs))
    print('\tPretraining epochs: {}'.format(pretrain_epochs))
    print('\tFine-tuning epochs: {}'.format(finetune_epochs))
    print('\tDevice: {}'.format(device))
    print('\tConduct initial measurements: {}'.format(init_meas))
    print()
    
    print('Setting random seed.')
    set_random_seed(seed)
    
    print('Initializing and partitioning datasets.')
    full_train_dataset, test_dataset = get_dataset(dataset)
    full_train_dataset = BinaryTargetDataset(full_train_dataset, majority_class, minority_class, total_samples)
    test_dataset = BinaryTargetDataset(test_dataset, majority_class, minority_class, total_samples)
    classes = np.unique(full_train_dataset.targets)
    train_dataset, full_train_dataset = extract_random_class_balanced_dataset(full_train_dataset, total_samples)
    clean_dataset, full_train_dataset = extract_random_class_balanced_dataset(full_train_dataset, clean_dataset_samples_per_class)
    val_dataset, _ = extract_random_class_balanced_dataset(full_train_dataset, val_samples_per_class)
    train_dataset = ImbalancedDataset(train_dataset, 1, 0, total_samples, minority_prop_of_majority)
    if method != 'sss':
        train_dataset.append_clean_dataset(clean_dataset)
    finetune_dataset = RepetitiveDataset(clean_dataset, len(train_dataset)//len(clean_dataset))
    finetune_dataset = ImbalancedDataset(finetune_dataset, 1, 0, len(finetune_dataset), 0.5)
    train_dataset = RepetitiveDataset(train_dataset, 5)
    
    print('Initializing dataloaders.')
    train_dataloader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, **eval_dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **eval_dataloader_kwargs)
    clean_dataloader = DataLoader(clean_dataset, **clean_dataloader_kwargs)
    finetune_dataloader = DataLoader(finetune_dataset, **clean_dataloader_kwargs)
        
    print('Initializing model.')
    model_constructor = getattr(models, model_constructor_attr)
    eg_input = next(iter(test_dataloader))[0]
    model = model_constructor(eg_input.shape, **model_constructor_kwargs)
    model = model.to(device)
    print(model)
    print()
    
    print('Initializing loss function.')
    loss_fn_constructor = getattr(nn, loss_fn_constructor_attr)
    loss_fn = loss_fn_constructor(**loss_fn_kwargs)
    loss_fn.reduction = 'none'
    print(loss_fn)
    print()
    
    print('Initializing optimizer.')
    optimizer_constructor = getattr(optim, optimizer_constructor_attr)
    optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)
    print(optimizer)
    print()
    
    if scheduler_constructor_attr != None:
        print('Initializing scheduler.')
        scheduler_constructor = getattr(optim.lr_scheduler, scheduler_constructor_attr)
        scheduler = scheduler_constructor(optimizer, **scheduler_kwargs)
        print(scheduler)
        print()
    else:
        scheduler = None
    
    results = Results()
    
    if init_meas:
        print('Measuring initial performance.')
        train_res = eval_epoch(train_dataloader, model, loss_fn, device)
        val_res = eval_epoch(val_dataloader, model, loss_fn, device)
        test_res = eval_epoch(test_dataloader, model, loss_fn, device)
        results.update(0, train_res, val_res, test_res)
    else:
        print('Skipping initial performance measurement.')
    
    if pretrain_epochs > 0:
        print('Pre-training model.')
        #model = best_model
        for epoch_num in range(1, pretrain_epochs+1):
            print('Beginning epoch {}.'.format(epoch_num))
            train_res = train_epoch(finetune_dataloader, model, loss_fn, optimizer, device, 'naive', clean_dataloader)
            val_res = eval_epoch(val_dataloader, model, loss_fn, device)
            test_res = eval_epoch(test_dataloader, model, loss_fn, device)
            results.update(epoch_num, train_res, val_res, test_res)
            scheduler.step(val_res['accuracy'])
    
    best_model = deepcopy(model)
    if num_epochs > 0:
        print('Training model.')
        best_accuracy = -np.inf
        final_test_accuracy = -np.inf
        for epoch_num in range(pretrain_epochs+1, pretrain_epochs+num_epochs+1):
            print('Beginning epoch {}.'.format(epoch_num))
            train_res = train_epoch(train_dataloader, model, loss_fn, optimizer, device, method, clean_dataloader)
            val_res = eval_epoch(val_dataloader, model, loss_fn, device)
            test_res = eval_epoch(test_dataloader, model, loss_fn, device)
            results.update(epoch_num, train_res, val_res, test_res)
            scheduler.step(val_res['accuracy'])
            if val_res['accuracy'] > best_accuracy:
                best_accuracy = val_res['accuracy']
                final_test_accuracy = test_res['accuracy']
                best_model = deepcopy(model)
        print('\tDone training. Final accuracy: {}'.format(final_test_accuracy))
    
    if finetune_epochs > 0:
        print('Fine-tuning model.')
        #model = best_model
        for epoch_num in range(pretrain_epochs+num_epochs+1, pretrain_epochs+num_epochs+finetune_epochs+1):
            print('Beginning epoch {}.'.format(epoch_num))
            train_res = train_epoch(finetune_dataloader, model, loss_fn, optimizer, device, 'naive', clean_dataloader)
            val_res = eval_epoch(val_dataloader, model, loss_fn, device)
            test_res = eval_epoch(test_dataloader, model, loss_fn, device)
            results.update(epoch_num, train_res, val_res, test_res)
            scheduler.step(val_res['accuracy'])
    
    trial_time = time.time() - trial_start_time
    results.add_single_pair('trial_time', trial_time)
    print('Trial complete.')
    print('\tTime taken: {} seconds.'.format(trial_time))
    
    return results.data, best_model

def eval_epoch(dataloader, model, loss_fn, device):
    batch_majority_loss = []
    batch_minority_loss = []
    batch_majority_acc = []
    batch_minority_acc = []
    for batch in tqdm(dataloader):
        images = batch[0]
        labels = batch[1]
        images = images.to(device)
        labels_d = labels.to(device)
        elementwise_loss, predictions = eval_on_batch(images, labels_d, model, loss_fn, device)
        batch_majority_loss.append(mean(elementwise_loss[labels==1]))
        batch_minority_loss.append(mean(elementwise_loss[labels==0]))
        batch_majority_acc.append(mean(np.equal(predictions, labels)[labels==1]))
        batch_minority_acc.append(mean(np.equal(predictions, labels)[labels==0]))
    epoch_majority_loss = mean(batch_majority_loss)
    epoch_minority_loss = mean(batch_minority_loss)
    epoch_majority_acc = mean(batch_majority_acc)
    epoch_minority_acc = mean(batch_minority_acc)
    return {'majority_loss': epoch_majority_loss,
            'minority_loss': epoch_minority_loss,
            'majority_acc': epoch_majority_acc,
            'minority_acc': epoch_minority_acc}

def train_epoch(dataloader, model, loss_fn, optimizer, device, method, val_dataloader=None):
    if method in ['ltrwe', 'sss']:
        assert val_dataloader != None
    
    batch_majority_loss = []
    batch_minority_loss = []
    batch_majority_accuracy = []
    batch_minority_accuracy = []
    batch_majority_nonzero = []
    batch_minority_nonzero = []
    
    for batch in tqdm(dataloader):
        images = batch[0]
        original_labels = batch[1]
        images = images.to(device)
        labels_d = original_labels.to(device)
        
        if method in ['naive']:
            elementwise_loss, predictions = naive_train_on_batch(images, labels_d, model, loss_fn, optimizer, device)
            weights = np.ones_like(elementwise_loss)
            labels = original_labels
        elif method in ['ltrwe', 'sss']:
            val_batch = next(iter(val_dataloader))
            val_image = val_batch[0]
            val_label = val_batch[1]
            val_image = val_image.to(device)
            val_label = val_label.to(device)
            if method == 'ltrwe':
                elementwise_loss, predictions, weights = ltrwe_train_on_batch(images, labels_d, val_image, val_label, model, loss_fn, optimizer, device)
                labels = original_labels
            elif method == 'sss':
                elementwise_loss, predictions, weights, labels = sss_train_on_batch(images, labels_d, val_image, val_label, model, loss_fn, optimizer, device)
        
        correct_labels = batch[2]
        batch_majority_loss.append(mean(elementwise_loss[label==1]))
        batch_minority_loss.append(mean(elementwise_loss[label==0]))
        batch_majority_accuracy.append(mean(np.equal(predictions, labels)[correct_label==1]))
        batch_minority_accuracy.append(mean(np.equal(predictions, labels)[correct_label==0]))
        batch_majority_nonzero.append(np.count_nonzero(weights[correct_label==1]))
        batch_minority_nonzero.append(np.count_nonzero(weights[correct_label==0]))
    
    epoch_majority_loss = mean(batch_majority_loss)
    epoch_minority_loss = mean(batch_minority_loss)
    epoch_majority_accuracy = mean(batch_majority_accuracy)
    epoch_minority_accuracy = mean(batch_minority_accuracy)
    epoch_majority_nonzero = np.sum(batch_majority_nonzero)
    epoch_minority_nonzero = np.sum(batch_minority_nonzero)
    return {'majority_loss': epoch_majority_loss,
            'minority_loss': epoch_minority_loss,
            'majority_acc': epoch_majority_accuracy,
            'minority_acc': epoch_minority_accuracy,
            'majority_nonzero': epoch_majority_nonzero,
            'minority_nonzero': epoch_minority_nonzero}

class ImbalancedDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 majority_class,
                 minority_class,
                 total_samples,
                 minority_prop_of_majority):
        super().__init__()
        self.minority_class_samples = int(majority_class_samples*minority_prop_of_majority)
        self.majority_class_samples = total_samples-self.minority_class_samples
        self.data = []
        self.targets = []
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        
        majority_to_go = self.majority_class_samples
        minority_to_go = self.minority_class_samples
        indices = [x for x in range(len(base_dataset))]
        random.shuffle(indices)
        for idx in indices:
            image, target = base_dataset[idx]
            if (target == majority_class) and (majority_to_go > 0):
                self.data.append(image)
                self.targets.append(1)
                majority_to_go -= 1
            elif (target == minority_class) and (minority_to_go > 0):
                self.data.append(image)
                self.targets.append(0)
                minority_to_go -= 1
        self.number_of_samples = len(self.data)
        assert self.number_of_samples == len(self.targets)
        assert majority_to_go == 0
        assert minority_to_go == 0
        
        base_dataset.transform = self.transform
        base_dataset.target_transform = self.target_transform
        
    def append_clean_dataset(self, clean_dataset):
        for data, target in zip(clean_dataset.data, clean_dataset.targets):
            self.number_of_samples += 1
            self.data.append(data)
            self.targets.append(target)
            
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform != None:
            image = self.transform(image)
        if self.target_transform != None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        return self.number_of_samples