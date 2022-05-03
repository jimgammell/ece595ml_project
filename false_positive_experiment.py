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
from train import sss_train_on_batch, ltrwe_train_on_batch, naive_train_on_batch, eval_on_batch
import models

def run_trial(config_params):
    trial_start_time = time.time()
    
    # Parse and validate trial_args
    method = config_params['method']
    assert method in ['naive', 'ltrwe', 'sss']
    dataset = config_params['dataset']
    assert dataset in ['CIFAR10', 'MNIST', 'FashionMNIST']
    samples_per_class = config_params['samples_per_class']
    positive_class = config_params['positive_class']
    negative_class = config_params['negative_class']
    proportion_false_positive = config_params['proportion_false_positive']
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
    print('\tSamples per class: {}'.format(samples_per_class))
    print('\tPositive class: {}'.format(positive_class))
    print('\tNegative class: {}'.format(negative_class))
    print('\tProportion of negative class with false positive label: {}'.format(proportion_false_positive))
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
    full_train_dataset = BinaryTargetDataset(full_train_dataset, positive_class, negative_class, 5000)
    test_dataset = BinaryTargetDataset(test_dataset, positive_class, negative_class, 800)
    classes = np.unique(full_train_dataset.targets)
    train_dataset, full_train_dataset = extract_random_class_balanced_dataset(full_train_dataset, samples_per_class)
    clean_dataset, full_train_dataset = extract_random_class_balanced_dataset(full_train_dataset, clean_dataset_samples_per_class)
    val_dataset, _ = extract_random_class_balanced_dataset(full_train_dataset, val_samples_per_class)
    train_dataset = FalsePositiveDataset(train_dataset, 1, 0, samples_per_class, proportion_false_positive)
    train_dataset.append_clean_dataset(clean_dataset)
    finetune_dataset = RepetitiveDataset(clean_dataset, len(train_dataset)//len(clean_dataset))
    finetune_dataset = FalsePositiveDataset(finetune_dataset, 1, 0, len(finetune_dataset)//len(classes), 0)
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
    
    trial_time = time.time() - trial_start_time
    results.add_single_pair('trial_time', trial_time)
    print('Trial complete.')
    print('\tTime taken: {} seconds.'.format(trial_time))
    
    return results.data, best_model

def eval_epoch(dataloader, model, loss_fn, device):
    batch_positive_loss = []
    batch_negative_loss = []
    batch_positive_acc = []
    batch_negative_acc = []
    for batch in tqdm(dataloader):
        images = batch[0]
        labels = batch[1]
        images = images.to(device)
        labels_d = labels.to(device)
        elementwise_loss, predictions = eval_on_batch(images, labels_d, model, loss_fn, device)
        batch_positive_loss.append(mean(elementwise_loss[labels==1]))
        batch_negative_loss.append(mean(elementwise_loss[labels==0]))
        batch_positive_acc.append(mean(np.equal(predictions, labels)[labels==1]))
        batch_negative_acc.append(mean(np.equal(predictions, labels)[labels==0]))
    epoch_positive_loss = mean(batch_positive_loss)
    epoch_negative_loss = mean(batch_negative_loss)
    epoch_positive_acc = mean(batch_positive_acc)
    epoch_negative_acc = mean(batch_negative_acc)
    return {'positive_loss': epoch_positive_loss,
            'negative_loss': epoch_negative_loss,
            'positive_acc': epoch_positive_acc,
            'negative_acc': epoch_negative_acc}

def train_epoch(dataloader, model, loss_fn, optimizer, device, method, val_dataloader=None):
    if method in ['ltrwe', 'sss']:
        assert val_dataloader != None
    
    batch_correct_loss = []
    batch_incorrect_loss = []
    batch_positive_loss = []
    batch_negative_loss = []
    batch_correct_accuracy = []
    batch_incorrect_accuracy = []
    batch_positive_accuracy = []
    batch_negative_accuracy = []
    batch_correct_nonzero = []
    batch_incorrect_nonzero = []
    batch_positive_nonzero = []
    batch_negative_nonzero = []
    
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
        correctness = np.equal(correct_labels, labels)
        batch_correct_loss.append(mean(elementwise_loss[correctness==1]))
        batch_incorrect_loss.append(mean(elementwise_loss[correctness==0]))
        batch_positive_loss.append(mean(elementwise_loss[labels==1]))
        batch_negative_loss.append(mean(elementwise_loss[labels==0]))
        batch_correct_accuracy.append(mean(np.equal(predictions, labels)[correctness==1]))
        batch_incorrect_accuracy.append(mean(np.equal(predictions, labels)[correctness==0]))
        batch_positive_accuracy.append(mean(np.equal(predictions, labels)[correct_labels==1]))
        batch_negative_accuracy.append(mean(np.equal(predictions, labels)[correct_labels==0]))
        batch_correct_nonzero.append(np.count_nonzero(weights[correctness==1]))
        batch_incorrect_nonzero.append(np.count_nonzero(weights[correctness==0]))
        batch_positive_nonzero.append(np.count_nonzero(weights[correct_labels==1]))
        batch_negative_nonzero.append(np.count_nonzero(weights[correct_labels==0]))
    
    epoch_correct_loss = mean(batch_correct_loss)
    epoch_incorrect_loss = mean(batch_incorrect_loss)
    epoch_positive_loss = mean(batch_positive_loss)
    epoch_negative_loss = mean(batch_negative_loss)
    epoch_correct_accuracy = mean(batch_correct_accuracy)
    epoch_incorrect_accuracy = mean(batch_incorrect_accuracy)
    epoch_positive_accuracy = mean(batch_positive_accuracy)
    epoch_negative_accuracy = mean(batch_negative_accuracy)
    epoch_correct_nonzero = np.sum(batch_correct_nonzero)
    epoch_incorrect_nonzero = np.sum(batch_incorrect_nonzero)
    epoch_positive_nonzero = np.sum(batch_positive_nonzero)
    epoch_negative_nonzero = np.sum(batch_negative_nonzero)
    return {'correct_loss': epoch_correct_loss,
            'incorrect_loss': epoch_incorrect_loss,
            'positive_loss': epoch_positive_loss,
            'negative_loss': epoch_negative_loss,
            'correct_acc': epoch_correct_accuracy,
            'incorrect_acc': epoch_incorrect_accuracy,
            'positive_acc': epoch_positive_accuracy,
            'negative_acc': epoch_negative_accuracy,
            'correct_nonzero': epoch_correct_nonzero,
            'incorrect_nonzero': epoch_incorrect_nonzero,
            'positive_nonzero': epoch_positive_nonzero,
            'negative_nonzero': epoch_negative_nonzero}

class FalsePositiveDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 positive_class,
                 negative_class,
                 samples_per_class,
                 proportion_false_positive):
        super().__init__()
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.samples_per_class = samples_per_class
        self.data = []
        self.noisy_targets = []
        self.correct_targets = []
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        
        false_positives = int(proportion_false_positive*samples_per_class)
        positive_samples_to_go = samples_per_class
        false_positive_samples_to_go = false_positives
        true_negative_samples_to_go = samples_per_class-false_positives
        indices = [x for x in range(len(base_dataset))]
        random.shuffle(indices)
        for idx in indices:
            image, target = base_dataset[idx]
            if target == positive_class:
                if positive_samples_to_go > 0:
                    self.data.append(image)
                    self.noisy_targets.append(positive_class)
                    self.correct_targets.append(positive_class)
                    positive_samples_to_go -= 1
            elif target == negative_class:
                if false_positive_samples_to_go > 0:
                    self.data.append(image)
                    self.noisy_targets.append(positive_class)
                    self.correct_targets.append(negative_class)
                    false_positive_samples_to_go -= 1
                elif true_negative_samples_to_go > 0:
                    self.data.append(image)
                    self.noisy_targets.append(negative_class)
                    self.correct_targets.append(negative_class)
                    true_negative_samples_to_go -= 1
        self.number_of_samples = len(self.data)
        assert self.number_of_samples == len(self.noisy_targets)
        assert self.number_of_samples == len(self.correct_targets)
        assert positive_samples_to_go == 0
        assert false_positive_samples_to_go == 0
        assert true_negative_samples_to_go == 0
        
        base_dataset.transform = self.transform
        base_dataset.target_transform = self.target_transform
    
    def append_clean_dataset(self, clean_dataset):
        for data, target in zip(clean_dataset.data, clean_dataset.targets):
            self.number_of_samples += 1
            self.data.append(data)
            self.noisy_targets.append(target)
            self.correct_targets.append(target)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        noisy_target = self.noisy_targets[idx]
        correct_target = self.correct_targets[idx]
        if self.transform != None:
            image = self.transform(image)
        if self.target_transform != None:
            noisy_target = self.transform(noisy_target)
        return image, noisy_target, correct_target
    
    def __len__(self):
        return self.number_of_samples