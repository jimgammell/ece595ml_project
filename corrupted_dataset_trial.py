from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from copy import deepcopy
import higher

class Results:
    def __init__(self):
        self.data = {}
    def keys(self):
        return self.data.keys()
    def add_key(self, key):
        self.data.update({key: {'epochs': [], 'values': []}})
    def append_value(self, key, epoch, value):
        self.data[key]['epochs'].append(epoch)
        self.data[key]['values'].append(value)
    def update(self, epoch, data):
        for key in data.keys():
            if not key in self.keys():
                self.add_key(key)
            self.append_value(key, epoch, data[key])
    def get_traces(self, key):
        x = np.array(self.data[key]['epochs'])
        y = np.array(self.data[key]['values'])
        return x, y

def dict_key_prepend(d, pref):
    keys_list = [k for k in d.keys()]
    for key in keys_list:
        d[pref+key] = d.pop(key)

class CleanDatasetTrial:
    def __init__(self,
                 method,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 num_epochs,
                 batch_size,
                 evaluate_initial_performance=True,
                 val_dataloader=None):
        self.results = Results()
        self.method = method
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.evaluate_initial_performance = evaluate_initial_performance
        self.val_dataloader = val_dataloader
    
    def __call__(self):
        if self.evaluate_initial_performance:
            train_loss, train_acc = self.eval_epoch(0, self.train_dataloader)
            test_loss, test_acc = self.eval_epoch(0, self.test_dataloader)
            self.results.update(0, {'train_loss': train_loss,
                                    'train_acc': train_acc,
                                    'test_loss': test_loss,
                                    'test_acc': test_acc})
        for epoch in range(1, self.num_epochs+1):
            if self.method == 'naive':
                train_loss, train_acc = self.naive_train_epoch(epoch)
                test_loss, test_acc = self.eval_epoch(epoch, self.test_dataloader)
                self.results.update(epoch, {'train_loss': train_loss,
                                            'train_acc': train_acc,
                                            'test_loss': test_loss,
                                            'test_acc': test_acc})
            elif self.method == 'ltrwe':
                res = self.ltrwe_train_epoch(epoch)
                test_loss, test_acc = self.eval_epoch(epoch, self.test_dataloader)
                self.results.update(epoch, {'train_loss': res[0],
                                            'train_acc': res[1],
                                            'weights_hist': res[2],
                                            'unused_samples': res[3],
                                            'p25_weight': res[4],
                                            'p50_weight': res[5],
                                            'p75_weight': res[6],
                                            'mn_weight': res[7],
                                            'std_weight': res[8],
                                            'test_loss': test_loss,
                                            'test_acc': test_acc})
        return self.results
        
    def eval_epoch(self, epoch_num, dataloader):
        print('Beginning evaluating epoch {}...'.format(epoch_num))
        batch_losses = []
        batch_accuracies = []
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(self.device)
            labels_d = labels.to(self.device)
            labels = labels.numpy()
            elementwise_loss, predictions = eval_on_batch(images, labels_d, self.model, self.loss_fn, self.optimizer, self.device)
            batch_loss = np.mean(elementwise_loss)
            batch_accuracy = np.mean(np.equal(predictions, labels))
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)
        loss = np.mean(batch_losses)
        accuracy = np.mean(batch_accuracies)
        print('\tLoss: {}'.format(loss))
        print('\tAccuracy: {}'.format(accuracy))
        return loss, accuracy
        
    def naive_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        batch_losses = []
        batch_accuracies = []
        for batch in tqdm(self.train_dataloader):
            images, labels = batch
            images = images.to(self.device)
            labels_d = labels.to(self.device)
            labels = labels.numpy()
            elementwise_loss, predictions = naive_train_on_batch(images, labels_d, self.model, self.loss_fn, self.optimizer, self.device)
            batch_loss = np.mean(elementwise_loss)
            batch_accuracy = np.mean(np.equal(predictions, labels))
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)
        loss = np.mean(batch_losses)
        accuracy = np.mean(batch_accuracies)
        print('\tLoss: {}'.format(loss))
        print('\tAccuracy: {}'.format(accuracy))
        return loss, accuracy
    
    def ltrwe_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        batch_losses = []
        batch_accuracies = []
        batch_weights = []
        for training_batch in tqdm(self.train_dataloader):
            training_images, training_labels = training_batch
            training_images = training_images.to(self.device)
            training_labels_d = training_labels.to(self.device)
            training_labels = training_labels.numpy()
            validation_batch = next(iter(self.val_dataloader))
            validation_images, validation_labels = validation_batch
            validation_images = validation_images.to(self.device)
            validation_labels_d = validation_labels.to(self.device)
            validation_labels = validation_labels.numpy()
            elementwise_loss, predictions, _, _, _, _, weights = ltrwe_train_on_batch(training_images, training_labels_d, validation_images, validation_labels_d, self.model, self.loss_fn, self.optimizer, self.device)
            batch_loss = np.mean(elementwise_loss)
            batch_accuracy = np.mean(np.equal(predictions, training_labels))
            batch_weight = list(weights)
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)
            batch_weights.extend(batch_weight)
        loss = np.mean(batch_losses)
        accuracy = np.mean(batch_accuracies)
        weight = np.histogram(batch_weights, bins=5*self.batch_size, range=(0.0, 1.0))
        unused_samples = len([w for w in batch_weights if w == 0])
        med_weight = np.percentile(weights, 50)
        p25_weight = np.percentile(weights, 25)
        p75_weight = np.percentile(weights, 75)
        mn_weight = np.mean(weights)
        std_weight = np.std(weights)
        print('\tLoss: {}'.format(loss))
        print('\tAccuracy: {}'.format(accuracy))
        print('\tWeight: {}'.format(weight))
        print('\tUnused samples: {}'.format(unused_samples))
        print('25/50/75 percentile weights: {} / {} / {}'.format(p25_weight, med_weight, p75_weight))
        print('Mean/std weights: {} / {}'.format(mn_weight, std_weight))
        return loss, accuracy, weight, unused_samples, p25_weight, med_weight, p75_weight, mn_weight, std_weight

class CleanDataset(Dataset):
    def __init__(self, *args):
        if len(args) == 1:
            self.init_from_base_dataset(*args)
        elif len(args) == 4:
            self.init_from_dataset_components(*args)
        elif len(args) == 0:
            self.init_for_test_set()
        else:
            assert False
    def init_from_base_dataset(self,
                 base_dataset):
        self.data_transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        self.data = []
        self.targets = []
        for (data, target) in base_dataset:
            self.data.append(data)
            self.targets.append(target)
        self.num_samples = len(self.data)
        assert self.num_samples == len(self.targets)
        
    def init_from_dataset_components(self,
                 data,
                 targets,
                 data_transform,
                 target_transform):
        self.data = data
        self.targets = targets
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.num_samples = len(self.data)
        assert self.num_samples == len(self.targets)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        if self.data_transform != None:
            data = self.data_transform(data)
        if self.target_transform != None:
            target = self.target_transform(target)
        return data, target
    
    def get_validation_dataset(self, samples_per_class):
        relevant_classes = np.unique(self.targets)
        number_to_go = {c: samples_per_class for c in relevant_classes}
        data = []
        targets = []
        for (image, target) in zip(self.data, self.targets):
            if number_to_go[target] > 0:
                data.append(image)
                targets.append(target)
                number_to_go[target] -= 1
        validation_dataset = CleanDataset(data,
                                          targets,
                                          self.data_transform,
                                          self.target_transform)
        return validation_dataset

class ImbalancedDatasetTrial:
    def __init__(self,
                 method,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 num_epochs,
                 batch_size,
                 input_shape,
                 evaluate_initial_performance=True,
                 val_dataloader=None,
                 self_reweight_model=False,
                 reweight_model_constructor=None,
                 reweight_model_kwargs=None,
                 reweight_model_period = 1,
                 exaustion_criteria = 0,
                 coarse_weights=False,
                 weights_propto_samples=False):
        self.results = Results()
        self.method = method
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.evaluate_initial_performance = evaluate_initial_performance
        self.val_dataloader = val_dataloader
        self.self_reweight_model = self_reweight_model
        self.reweight_model_constructor = reweight_model_constructor
        self.reweight_model_kwargs = reweight_model_kwargs
        self.reweight_model_period = reweight_model_period
        self.exaustion_criteria = exaustion_criteria
        self.coarse_weights = coarse_weights
        self.weights_propto_samples = weights_propto_samples
        self.input_shape = input_shape
    
    def __call__(self):
        if self.evaluate_initial_performance:
            training_results = self.eval_epoch(0, self.train_dataloader)
            test_results = self.eval_epoch(0, self.test_dataloader)
            dict_key_prepend(training_results, 'train_')
            dict_key_prepend(test_results, 'test_')
            self.results.update(0, training_results)
            self.results.update(0, test_results)
        best_test_accuracy = -np.inf
        epoch = 1
        epochs_without_improvement = 0
        while (epoch <= self.num_epochs) or (epochs_without_improvement < self.exaustion_criteria):
            if self.method == 'naive':
                training_results = self.naive_train_epoch(epoch)
                test_results = self.eval_epoch(epoch, self.test_dataloader)
                dict_key_prepend(training_results, 'train_')
                dict_key_prepend(test_results, 'test_')
                self.results.update(epoch, training_results)
                self.results.update(epoch, test_results)
            elif self.method == 'ltrwe':
                training_results = self.ltrwe_train_epoch(epoch)
                test_results = self.eval_epoch(epoch, self.test_dataloader)
                dict_key_prepend(training_results, 'train_')
                dict_key_prepend(test_results, 'test_')
                self.results.update(epoch, training_results)
                self.results.update(epoch, test_results)
            elif self.method == 'smltrwe':
                if ((epoch-1) % self.reweight_model_period) == 0:
                    reweight_model = self.reweight_model_constructor(self.input_shape, **self.reweight_model_kwargs)
                    if self.self_reweight_model:
                        reweight_model.load_state_dict(self.model.state_dict())
                    reweight_model = reweight_model.to(self.device)
                training_results = self.smltrwe_train_epoch(epoch, reweight_model)
                test_results = self.eval_epoch(epoch, self.test_dataloader)
                dict_key_prepend(training_results, 'train_')
                dict_key_prepend(test_results, 'test_')
                self.results.update(epoch, training_results)
                self.results.update(epoch, test_results)
            test_accuracy = test_results['test_minority_accuracy']
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epoch += 1
        return self.results
        
    def evaluation_metrics_dict(self):
        return {'majority_loss': [],
                'minority_loss': [],
                'majority_accuracy': [],
                'minority_accuracy': []}
        
    def compute_evaluation_metrics(self, elementwise_loss, predictions, labels):
        def mean(x):
            if len(x) == 0:
                return 0
            return np.mean(x)
        majority_loss = mean(elementwise_loss[labels==0])
        minority_loss = mean(elementwise_loss[labels==1])
        majority_accuracy = mean(np.equal(predictions, labels)[labels==0])
        minority_accuracy = mean(np.equal(predictions, labels)[labels==1])
        return majority_loss, minority_loss, majority_accuracy, minority_accuracy
    
    def append_evaluation_metrics(self, em_dict, metrics):
        for (key, metric) in zip(em_dict.keys(), metrics):
            if not(np.isnan(metric)):
                em_dict[key].append(metric)
    
    def ltrwe_metrics_dict(self):
        return {'majority_loss': [],
                'minority_loss': [],
                'majority_accuracy': [],
                'minority_accuracy': [],
                'majority_weights_mean': [],
                'minority_weights_mean': [],
                'majority_nonzero_samples': [],
                'minority_nonzero_samples': []}
    
    def compute_ltrwe_metrics(self,
                              elementwise_loss,
                              predictions,
                              labels,
                              weights):
        def mean(x):
            if len(x) == 0:
                return np.nan
            return np.mean(x)
        majority_loss = mean(elementwise_loss[labels==0])
        minority_loss = mean(elementwise_loss[labels==1])
        majority_accuracy = mean(np.equal(predictions, labels)[labels==0])
        minority_accuracy = mean(np.equal(predictions, labels)[labels==1])
        majority_weights_mean = mean(weights[labels==0])
        minority_weights_mean = mean(weights[labels==1])
        majority_nonzero_samples = np.count_nonzero(weights[labels==0])
        minority_nonzero_samples = np.count_nonzero(weights[labels==1])
        return majority_loss, minority_loss, majority_accuracy, minority_accuracy, majority_weights_mean, minority_weights_mean, majority_nonzero_samples, minority_nonzero_samples
    
    def append_ltrwe_metrics(self, em_dict, metrics):
        for (key, metric) in zip(em_dict.keys(), metrics):
            if not(np.isnan(metric)):
                em_dict[key].append(metric)
    
    def eval_epoch(self, epoch_num, dataloader):
        print('Beginning evaluating epoch {}...'.format(epoch_num))
        results = self.evaluation_metrics_dict()
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(self.device)
            labels_d = labels.to(self.device)
            labels = labels.numpy()
            elementwise_loss, predictions = eval_on_batch(images, labels_d, self.model, self.loss_fn, self.optimizer, self.device)
            metrics = self.compute_evaluation_metrics(elementwise_loss, predictions, labels)
            self.append_evaluation_metrics(results, metrics)
        for key in results.keys():
            results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
        
    def naive_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.evaluation_metrics_dict()
        for batch in tqdm(self.train_dataloader):
            images, labels = batch
            images = images.to(self.device)
            labels_d = labels.to(self.device)
            labels = labels.numpy()
            elementwise_loss, predictions = naive_train_on_batch(images, labels_d, self.model, self.loss_fn, self.optimizer, self.device)
            metrics = self.compute_evaluation_metrics(elementwise_loss, predictions, labels)
            self.append_evaluation_metrics(results, metrics)
        for key in results.keys():
            results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
    def smltrwe_train_epoch(self, epoch_num, reweight_model):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.ltrwe_metrics_dict()
        for training_batch in tqdm(self.train_dataloader):
            training_images, training_labels = training_batch
            training_images = training_images.to(self.device)
            training_labels_d = training_labels.to(self.device)
            training_labels = training_labels.numpy()
            validation_batch = next(iter(self.val_dataloader))
            validation_images, validation_labels = validation_batch
            validation_images = validation_images.to(self.device)
            validation_labels_d = validation_labels.to(self.device)
            validation_labels = validation_labels.numpy()
            values = smltrwe_train_on_batch(training_images, training_labels_d, validation_images, validation_labels_d, self.model, reweight_model, self.loss_fn, self.optimizer, self.device, self.weights_propto_samples, self.coarse_weights)
            metrics = self.compute_ltrwe_metrics(*values)
            self.append_ltrwe_metrics(results, metrics)
        for key in results:
            if key in ['majority_nonzero_samples', 'minority_nonzero_samples']:
                results[key] = np.sum(results[key])
            else:
                results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
    def ltrwe_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.ltrwe_metrics_dict()
        for training_batch in tqdm(self.train_dataloader):
            training_images, training_labels = training_batch
            training_images = training_images.to(self.device)
            training_labels_d = training_labels.to(self.device)
            training_labels = training_labels.numpy()
            validation_batch = next(iter(self.val_dataloader))
            validation_images, validation_labels = validation_batch
            validation_images = validation_images.to(self.device)
            validation_labels_d = validation_labels.to(self.device)
            validation_labels = validation_labels.numpy()
            values = ltrwe_train_on_batch(training_images, training_labels_d, validation_images, validation_labels_d, self.model, self.loss_fn, self.optimizer, self.device, self.weights_propto_samples, self.coarse_weights)
            metrics = self.compute_ltrwe_metrics(*values)
            self.append_ltrwe_metrics(results, metrics)
        for key in results:
            if key in ['majority_nonzero_samples', 'minority_nonzero_samples']:
                results[key] = np.sum(results[key])
            else:
                results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
class ImbalancedDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 majority_class=9,
                 minority_class=4,
                 num_majority_samples=2500,
                 num_minority_samples=2500):
        self.data = []
        self.targets = []
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        self.majority_class = majority_class
        self.minority_class = minority_class
        
        majority_to_go = num_majority_samples
        minority_to_go = num_minority_samples
        for (image, target) in base_dataset:
            if (target==majority_class) and (majority_to_go>0):
                self.data.append(image)
                self.targets.append(0)
                majority_to_go -= 1
            if (target==minority_class) and (minority_to_go>0):
                self.data.append(image)
                self.targets.append(1)
                minority_to_go -= 1
        self.number_of_samples = len(self.data)
        assert self.number_of_samples == len(self.targets)
        
    def __len__(self):
        return self.number_of_samples
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        if self.transform != None:
            data = self.transform(data)
        if self.target_transform != None:
            target = self.target_transform(data)
        return data, target
    
    def get_validation_dataset(self,
                               samples_per_class):
        majority_to_go = samples_per_class
        minority_to_go = samples_per_class
        data = []
        targets = []
        for (image, target) in zip(self.data, self.targets):
            if (target==0) and (majority_to_go>0):
                data.append(image)
                targets.append(0)
                majority_to_go -= 1
            if (target==1) and (minority_to_go>0):
                data.append(image)
                targets.append(1)
                minority_to_go -= 1
        validation_dataset = CleanDataset(data,
                                          targets,
                                          self.transform,
                                          self.target_transform)
        return validation_dataset

class NoisyLabelsDatasetTrial(Dataset):
    def __init__(self,
                 method,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 num_epochs, 
                 batch_size,
                 input_shape,
                 evaluate_initial_performance=True,
                 val_dataloader=None,
                 pretrain_dataloader=None,
                 reweight_model_constructor=None,
                 reweight_model_kwargs=None,
                 reweight_model_period=1,
                 self_reweight_model=False,
                 exaustion_criteria=0):
        self.method = method
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.evaluate_initial_performance = evaluate_initial_performance
        self.val_dataloader = val_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.reweight_model_constructor = reweight_model_constructor
        self.reweight_model_kwargs = reweight_model_kwargs
        self.reweight_model_period = reweight_model_period
        self.self_reweight_model = self_reweight_model
        self.exaustion_criteria = exaustion_criteria
        self.results = Results()
    
    def __call__(self):
        epoch = 0
        epochs_without_improvement = 0
        best_test_accuracy = -np.inf
        
        # Evaluate initial model performance
        if self.evaluate_initial_performance:
            train_results = self.eval_epoch(0, self.train_dataloader)
            test_results = self.eval_epoch(0, self.test_dataloader)
            dict_key_prepend(training_results, 'train_')
            dict_key_prepend(test_results, 'test_')
            self.results.update(0, training_results)
            self.results.update(0, test_results)
        epoch += 1
        
        if self.method == 'semi-self-supervised':
            while epoch <= self.pretrain_epochs:
                training_results = self.naive_train_epoch(0, self.pretrain_dataloader)
                test_results = self.eval_epoch(0, self.test_dataloader)
                dict_key_prepend(training_results, 'pretrain_')
                dict_key_prepend(test_results, 'test_')
                self.results.update(epoch, training_results)
                self.results.update(epoch, test_results)
                epoch += 1
        
        while (epoch <= self.pretrain_epochs+self.num_epochs) or (epochs_without_improvement < self.exaustion_criteria):
            if self.method == 'naive':
                training_results = self.naive_train_epoch(epoch)
            elif self.method == 'ltrwe':
                training_results = self.ltrwe_train_epoch(epoch)
            elif self.method == 'smltrwe':
                if epoch % self.reweight_model_period == 0:
                    reweight_model = self.reweight_model_constructor(self.input_shape, **self.reweight_model_kwargs)
                    if self.self_reweight_model:
                        reweight_model.load_state_dict(model.state_dict())
                    reweight_model = reweight_model.to(self.device)
                training_results = self.smltrwe_train_epoch(epoch, reweight_model)
            elif self.method == 'semi-self-supervised':
                training_results = self.sss_train_epoch(epoch)
            else:
                assert False
                
            test_results = self.eval_epoch(epoch, self.test_dataloader)
            dict_key_prepend(training_results, 'train_')
            dict_key_prepend(test_results, 'test_')
            self.results.update(epoch, training_results)
            self.results.update(epoch, test_results)
            
            test_accuracy = test_results['test_accuracy']
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epoch += 1
        return self.results
    
    def evaluation_metrics_dict(self):
        return {'correct_loss': [],
                'incorrect_loss': [],
                'correct_accuracy': [],
                'incorrect_accuracy': []}
    
    def compute_evaluation_metrics(self, noise_presence, elementwise_loss, predictions, labels):
        def mean(x):
            if len(x) == 0:
                return np.nan
            return np.mean(x)
        correct_loss = np.mean(elementwise_loss[noise_presence==0])
        incorrect_loss = np.mean(elementwise_loss[noise_presence==1])
        correct_accuracy = np.mean(np.equal(predictions, labels)[noise_presence==0])
        incorrect_accuracy = np.mean(np.equal(predictions, labels)[noise_presence==1])
        return correct_loss, incorrect_loss, correct_accuracy, incorrect_accuracy
    
    def append_evaluation_metrics(self, em_dict, metrics):
        for (key, metric) in zip(em_dict.keys(), metrics):
            em_dict[key].append(metric)
    
    def ltrwe_metrics_dict(self):
        return {'correct_loss': [],
                'incorrect_loss': [],
                'correct_accuracy': [],
                'incorrect_accuracy': [],
                'correct_weights_mean': [],
                'incorrect_weights_mean': [],
                'correct_nonzero_samples': [],
                'incorrect_nonzero_samples': []}
    
    def compute_ltrwe_metrics(self, noise_presence, elementwise_loss, predictions, labels, weights):
        def mean(x):
            if len(x) == 0:
                return np.nan
            return np.mean(x)
        correct_loss = mean(elementwise_loss[noise_presence==0])
        incorrect_loss = mean(elementwise_loss[noise_presence==1])
        correct_accuracy = mean(np.equal(predictions, labels)[noise_presence==0])
        incorrect_accuracy = mean(np.equal(predictions, labels)[noise_presence==1])
        correct_weights_mean = mean(weights[noise_presence==0])
        incorrect_weights_mean = mean(weights[noise_presence==1])
        correct_nonzero_samples = np.count_nonzero(weights[noise_presence==0])
        incorrect_nonzero_samples = np.count_nonzero(weights[noise_presence==1])
        return correct_loss, incorrect_loss, correct_accuracy, incorrect_accuracy, correct_weights_mean, incorrect_weights_mean, correct_nonzero_samples, incorrect_nonzero_samples
    
    def append_ltrwe_metrics(self, em_dict, metrics):
        for (key, metric) in zip(em_dict.keys(), metrics):
            em_dict[key].append(metric)
    
    def eval_epoch(self, epoch_num, dataloader):
        print('Beginning evaluating epoch {}...'.format(epoch_num))
        results = self.evaluation_metrics_dict()
        for batch in tqdm(dataloader):
            images, labels, noise_presence = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            values = eval_on_batch(images, labels, self.model, self.loss_fn, self.optimizer, self.device)
            metrics = self.compute_evaluation_metrics(noise_presence, *values)
            self.append_evaluation_metrics(results, metrics)
        for key in results.keys():
            results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
    def naive_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.evaluation_metrics_dict()
        for batch in tqdm(self.train_dataloader):
            images, labels, noise_presence = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            values = naive_train_on_batch(images, labels, self.model, self.loss_fn, self.optimizer, self.device)
            metrics = self.compute_evaluation_metrics(noise_presence, *values)
            self.append_evaluation_metrics(results, metrics)
        for key in results.keys():
            results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
    def ltrwe_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.ltrwe_metrics_dict()
        for training_batch in tqdm(self.training_dataloader):
            training_images, training_labels, noise_presence = training_batch
            training_images = training_images.to(self.device)
            training_labels = training_labels.to(self.device)
            validation_batch = next(iter(self.val_dataloader))
            validation_images, validation_labels = validation_batch
            validation_images = validation_images.to(self.device)
            validation_labels = validation_labels.to(self.device)
            values = ltrwe_train_on_batch(train_images, train_labels, validation_images, validation_labels, self.model, self.loss_fn, self.optimizer, self.device, self.weights_propto_samples, self.coarse_weights)
            metrics = self.compute_ltrwe_metrics(noise_presence, *values)
            self.append_ltrwe_metrics(results, metrics)
        for key in results:
            if key in ['clean_nonzero_samples', 'noisy_nonzero_samples']:
                results[key] = np.sum(results[key])
            else:
                results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
    def smltrwe_train_epoch(self, epoch_num, reweight_model):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.ltrwe_metrics_dict()
        for training_batch in tqdm(self.training_dataloader):
            training_images, training_labels, noise_presence = training_batch
            training_images = training_images.to(self.device)
            training_labels = training_labels.to(self.device)
            validation_batch = next(iter(self.val_dataloader))
            validation_images, validation_labels = validation_batch
            validation_images = validation_images.to(self.device)
            validation_labels = validation_labels.to(self.device)
            values = ltrwe_train_on_batch(train_images, train_labels, validation_images, validation_labels, self.model, reweight_model, self.loss_fn, self.optimizer, self.device, self.weights_propto_samples, self.coarse_weights)
            metrics = self.compute_ltrwe_metrics(noise_presence, *values)
            self.append_ltrwe_metrics(results, metrics)
        for key in results:
            if key in ['clean_nonzero_samples', 'noisy_nonzero_samples']:
                results[key] = np.sum(results[key])
            else:
                results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
    def sss_train_epoch(self, epoch_num):
        print('Beginning training epoch {}...'.format(epoch_num))
        results = self.ltrwe_metrics_dict()
        for training_batch in tqdm(self.training_dataloader):
            training_images, training_labels, noise_presence = training_batch
            training_images = training_images.to(self.device)
            training_labels = training_labels.to(self.device)
            validation_batch = next(iter(self.val_dataloader))
            validation_images, validation_labels = validation_batch
            validation_images = validation_images.to(self.device)
            validation_labels = validation_labels.to(self.device)
            values = sss_train_on_batch(train_images, train_labels, validation_images, validation_labels, self.model, self.loss_fn, self.optimizer, self.device, self.weights_propto_samples, self.coarse_weights)
            metrics = self.compute_ltrwe_metrics(noise_presence, *values)
            self.append_ltrwe_metrics(results, metrics)
        for key in results:
            if key in ['clean_nonzero_samples', 'noisy_nonzero_samples']:
                results[key] = np.sum(results[key])
            else:
                results[key] = np.mean(results[key])
            print('\t{}: {}'.format(key, results[key]))
        return results
    
class NoisyLabelsDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 num_clean_samples,
                 num_noisy_samples,
                 relevant_classes):
        self.data = []
        self.targets = []
        self.noise_presence = []
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        self.relevant_classes = relevant_classes
        
        clean_samples_to_go = {c: num_clean_samples for c in self.relevant_classes}
        noisy_samples_to_go = {c: num_noisy_samples for c in self.relevant_classes}
        for (image, target) in base_dataset:
            if target in relevant_classes:
                if clean_samples_to_go[target] > 0:
                    self.data.append(data)
                    self.targets.append(target)
                    self.noise_presence.append(0)
                    clean_samples_to_go[target] -= 1
                elif noisy_samples_to_go[target] > 0:
                    self.data.append(data)
                    self.targets.append(random.choice(self.relevant_classes))
                    self.noise_presence.append(1)
                    noisy_samples_to_go[target] -= 1
        self.number_of_samples = len(self.data)
        assert self.number_of_samples == len(self.targets)
        assert self.number_of_samples == len(self.noise_presence)
        
    def __len__(self):
        return self.number_of_samples
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        noise_presence = self.noise_presence[idx]
        if self.transform != None:
            data = self.transform(data)
        if self.target_transform != None:
            target = self.target_transform(target)
        return data, target, noise_presence
    
    def get_validation_dataset(self, samples_per_class):
        number_to_go = {c: samples_per_class for c in self.relevant_classes}
        data = []
        targets = []
        for (image, target) in zip(self.data, self.targets):
            if number_to_go[target] > 0:
                data.append(image)
                targets.append(target)
                number_to_go[target] -= 1
        validation_dataset = CleanDataset(data,
                                          targets,
                                          self.transform,
                                          self.target_transform)
        return validation_dataset

def eval_on_batch(images,
                  labels,
                  model,
                  loss_fn,
                  optimizer,
                  device):
    model.eval()
    with torch.no_grad():
        logits = model(images)
        elementwise_loss = loss_fn(logits, labels)
    
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return elementwise_loss, predictions

def smltrwe_train_on_batch(training_images,
                           training_labels,
                           validation_images,
                           validation_labels,
                           model,
                           reweight_model,
                           loss_fn,
                           optimizer,
                           device,
                           reweight_by_nonzero_examples=False,
                           coarse_example_reweighting=False):
    model.train()
    
    reweight_state = deepcopy(reweight_model.state_dict())
    dummy_optimizer = optim.SGD(reweight_model.parameters(), lr=.001)
    with higher.innerloop_ctx(reweight_model, dummy_optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        training_loss = loss_fn(training_logits, training_labels)
        eps = torch.zeros_like(training_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_loss = loss_fn(validation_logits, validation_labels)
        reduced_validation_loss = torch.mean(validation_loss)
    eps_grad = torch.autograd.grad(reduced_validation_loss, eps)[0].detach()
    weights = nn.functional.relu(-eps_grad)
    if torch.norm(weights) != 0:
        if coarse_example_reweighting:
            weights[torch.nonzero(weights)] = 1
        weights /= torch.sum(weights)
        if reweight_by_nonzero_examples:
            weights *= torch.norm(weights, p=0)/len(training_images)
    reweight_model.load_state_dict(reweight_state)
    
    optimizer.zero_grad()
    logits = model(training_images)
    elementwise_loss = loss_fn(logits, training_labels)
    reweighted_loss = torch.sum(elementwise_loss*weights)
    reweighted_loss.backward()
    optimizer.step()
    
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    labels = training_labels.detach().cpu().numpy()
    validation_loss = validation_loss.detach().cpu().numpy()
    validation_predictions = np.argmax(validation_logits.detach().cpu().numpy(), axis=1)
    validation_labels = validation_labels.detach().cpu().numpy()
    weights = weights.detach().cpu().numpy()
    return elementwise_loss, predictions, labels, weights

def ltrwe_train_on_batch(training_images,
                         training_labels, 
                         validation_images,
                         validation_labels,
                         model,
                         loss_fn,
                         optimizer,
                         device,
                         reweight_by_nonzero_examples=False,
                         coarse_example_reweighting=False):
    model.train()
    
    model_params_backup = deepcopy(model.state_dict())
    dummy_optimizer = optim.SGD(model.parameters(), lr=.001)
    with higher.innerloop_ctx(model, dummy_optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        training_loss = loss_fn(training_logits, training_labels)
        eps = torch.zeros_like(training_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_loss = loss_fn(validation_logits, validation_labels)
        reduced_validation_loss = torch.mean(validation_loss)
    eps_grad = torch.autograd.grad(reduced_validation_loss, eps)[0].detach()
    weights = nn.functional.relu(-eps_grad)
    if torch.norm(weights) != 0:
        if coarse_example_reweighting:
            weights[torch.nonzero(weights)] = 1
        weights /= torch.sum(weights)
        if reweight_by_nonzero_examples:
            weights *= torch.norm(weights, p=0)/len(training_images)
    model.load_state_dict(model_params_backup)
    
    optimizer.zero_grad()
    logits = model(training_images)
    elementwise_loss = loss_fn(logits, training_labels)
    reweighted_loss = torch.sum(elementwise_loss*weights)
    reweighted_loss.backward()
    optimizer.step()
    
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    labels = training_labels.detach().cpu().numpy()
    validation_loss = validation_loss.detach().cpu().numpy()
    validation_predictions = np.argmax(validation_logits.detach().cpu().numpy(), axis=1)
    validation_labels = validation_labels.detach().cpu().numpy()
    weights = weights.detach().cpu().numpy()
    return elementwise_loss, predictions, labels, weights

def sss_train_on_batch(training_images,
                       training_labels,
                       validation_images,
                       validation_labels,
                       model,
                       loss_fn,
                       optimizer,
                       device):
    model.train()
    
    ## TODO: Can these two computations be done in parallel?
    
    # Compute epsilon gradients on labels in dataset
    model_params_backup = deepcopy(model.state_dict())
    dummy_optimizer = optim.SGD(model.parameters(), lr=.001)
    with higher.innerloop_ctx(model, dummy_optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        training_loss = loss_fn(training_logits, training_labels)
        eps = torch.zeros_like(training_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_loss = loss_fn(validation_logits, validation_labels)
        reduced_validation_loss = torch.mean(validation_loss)
    dataset_labels_eps_grad = torch.autograd.grad(reduced_validation_loss, eps)[0].detach()
    model.load_state_dict(model_params_backup)
    
    # Compute epsilon gradients on self-generated labels
    self_labels = torch.argmax(training_logits.detach(), dim=-1)
    dummy_optimizer = optim.SGD(model.parameters(), lr=.001)
    with higher.innerloop_ctx(model, dummy_optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        training_loss = loss_fn(training_logits, self_labels)
        eps = torch.zeros_like(training_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_loss = loss_fn(validation_logits, validation_labels)
        reduced_validation_loss = torch.mean(validation_loss)
    self_labels_eps_grad = torch.autograd.grad(reduced_validation_loss, eps)[0].detach()
    model.load_state_dict(model_params_backup)
    
    # Choose labels which maximize epsilon grads
    label_options = torch.stack((training_labels, self_labels))
    eg_options = torch.stack((dataset_labels_eps_grad, self_labels_eps_grad))
    max_eg_idx = torch.argmax(-eg_options, dim=0).unsqueeze(0)
    labels = torch.gather(label_options, 0, max_eg_idx)
    eps_grad = torch.gather(eg_options, 0, max_eg_idx)
    
    # Compute example weights
    weights = nn.functional.relu(-eps_grad)
    if torch.norm(weights) != 0:
        weights /= torch.sum(weights)
    
    # Train model using computed labels and example weights
    optimizer.zero_grad()
    logits = model(training_images)
    elementwise_loss = loss_fn(logits, labels)
    reweighted_loss = torch.sum(elementwise_loss*weights)
    reweighted_loss.backward()
    optimizer.step()
    
    # Detach results and convert to numpy
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy())
    labels = labels.detach().cpu().numpy()
    weights = weights.detach().cpu().numpy()
    return elementwise_loss, predictions, labels, weights

def naive_train_on_batch(images,
                         labels,
                         model,
                         loss_fn,
                         optimizer,
                         device):
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    elementwise_loss = loss_fn(logits, labels)
    reduced_loss = torch.mean(elementwise_loss)
    reduced_loss.backward()
    optimizer.step()
    
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return elementwise_loss, predictions