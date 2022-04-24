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
        x = self.data[key]['epochs']
        y = self.data[key]['values']
        return x, y

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
            elementwise_loss, predictions, weights = ltrwe_train_on_batch(training_images, training_labels_d, validation_images, validation_labels_d, self.model, self.loss_fn, self.optimizer, self.device)
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

class ImbalancedDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 majority_class,
                 minority_class,
                 num_majority_samples,
                 num_minority_samples):
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
                self.targets.append(target)
                majority_to_go -= 1
            if (target==minority_class) and (minority_to_go>0):
                self.data.append(image)
                self.targets.append(target)
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
            if (target==self.majority_class) and (majority_to_go>0):
                data.append(image)
                targets.append(target)
                majority_to_go -= 1
            if (target==self.minority_class) and (minority_to_go>0):
                data.append(image)
                targets.append(target)
                minority_to_go -= 1
        validation_dataset = CleanDataset(data,
                                          targets,
                                          self.transform,
                                          self.target_transform)
        return validation_dataset

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
                    possible_targets = [c for c in self.relevant_classes if c!=target]
                    self.targets.append(random.choice(possible_targets))
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

def ltrwe_train_on_batch(training_images,
                         training_labels, 
                         validation_images,
                         validation_labels,
                         model,
                         loss_fn,
                         optimizer,
                         device):
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
        validation_loss = torch.mean(validation_loss)
    eps_grad = torch.autograd.grad(validation_loss, eps)[0].detach()
    weights = nn.functional.relu(-eps_grad)
    if torch.norm(weights) != 0:
        weights /= torch.sum(weights)
    model.load_state_dict(model_params_backup)
    
    optimizer.zero_grad()
    logits = model(training_images)
    elementwise_loss = loss_fn(logits, training_labels)
    reweighted_loss = torch.sum(elementwise_loss*weights)
    reweighted_loss.backward()
    optimizer.step()
    
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    weights = weights.detach().cpu().numpy()
    return elementwise_loss, predictions, weights

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