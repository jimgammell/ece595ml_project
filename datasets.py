import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

class RepetitiveDataset(Dataset):
    def __init__(self, base_dataset, num_repetitions):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_samples = self.base_dataset.__len__() * num_repetitions
        self.transform = None
        self.target_transform = None
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        idx %= self.base_dataset.__len__()
        item = self.base_dataset.__getitem__(idx)
        return item

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        subset_transform = subset.dataset.transform
        subset_target_transform = subset.dataset.target_transform
        subset.dataset.transform = None
        subset.dataset.target_transform = None
        self.data = []
        self.targets = []
        for data, target in subset:
            self.data.append(data)
            self.targets.append(target)
        subset.dataset.transform = subset_transform
        subset.dataset.target_transform = subset_target_transform
        self.num_samples = len(subset)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        if self.transform != None:
            data = self.transform(data)
        if self.target_transform != None:
            target = self.target_transform(target)
        return data, target
    
    def __len__(self):
        return self.num_samples
    
def extract_random_class_balanced_dataset(dataset, samples_per_class):
    classes, class_counts = np.unique(dataset.targets, return_counts=True)
    extracted_indices = []
    remainder_indices = []
    assert all([class_count>=samples_per_class for class_count in class_counts])
    samples_to_go = {c: samples_per_class for c in classes}
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    for sidx in indices:
        target = dataset.targets[sidx]
        if samples_to_go[target] > 0:
            samples_to_go[target] -= 1
            extracted_indices.append(sidx)
        else:
            remainder_indices.append(sidx)
    extracted_subset = Subset(dataset, extracted_indices)
    remainder_subset = Subset(dataset, remainder_indices)
    extracted_dataset = DatasetFromSubset(extracted_subset, dataset.transform, dataset.target_transform)
    remainder_dataset = DatasetFromSubset(remainder_subset, dataset.transform, dataset.target_transform)
    return extracted_dataset, remainder_dataset
    
def get_dataset(name):
    dataset_path = os.path.join('.', 'datasets')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    if name == 'CIFAR10':
        training_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2021))])
        evaluation_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2021))])
        dataset_constructor = datasets.CIFAR10
    elif name == 'MNIST':
        training_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081))])
        evaluation_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081))])
        dataset_constructor = datasets.MNIST
    elif name == 'FashionMNIST':
        training_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.3814,), (0.3994,))])
        evaluation_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.3814,), (0.3994,))])
        dataset_constructor = datasets.FashionMNIST
    else:
        assert False
        
    train_dataset = dataset_constructor(root=dataset_path,
                                        train=True,
                                        transform=training_transform,
                                        download=True)
    train_dataset.targets = np.array(train_dataset.targets)
    
    test_dataset = dataset_constructor(root=dataset_path,
                                       train=False,
                                       transform=evaluation_transform,
                                       download=True)
    test_dataset.targets = np.array(test_dataset.targets)
    
    return train_dataset, test_dataset
        