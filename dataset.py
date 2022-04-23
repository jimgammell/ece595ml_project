import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from matplotlib import pyplot as plt

class GrayscaleToRgb:
    def __init__(self):
        pass
    def __call__(self, x):
        x = x.repeat(3, 1, 1)
        return x

def compute_minimal_perturbation(data, target, model, loss_fn, device, eps, max_steps=100):
    data.requires_grad = True
    data = data.to(device)
    target = target.to(device)
    model.eval()
    logits = model(data)
    loss = loss_fn(logits, target)
    data_grad = torch.autograd.grad(loss, data)[0].detach()
    perturbation = torch.sign(data_grad)
    data.requires_grad = False
    
    a, b = 0, 1
    while (b-a)/2 >= eps:
        c = (a+b)/2
        with torch.no_grad():
            logits = model(data+c*perturbation)
        logits = logits.cpu().numpy()
        prediction = np.argmax(logits, axis=-1)
        if prediction == target:
            a = c
        else:
            b = c
        max_steps -= 1
        if max_steps < 0:
            return np.nan
    return c

class ValidationDataset(Dataset):
    def __init__(self,
                 dataset,
                 data_transform=None,
                 target_transform=None):
        super().__init__()
        self.data = []
        self.targets = []
        self.data_transform = data_transform
        self.target_transform = target_transform
        for (image, target) in dataset:
            self.data.append(np.array(image))
            self.targets.append(np.array(target, dtype=int))
        self.num_samples = len(self.data)
        assert self.num_samples == len(self.targets)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

class FewClassDataset(Dataset):
    def __init__(self,
                 dataset,
                 classes,
                 data_transform,
                 target_transform):
        self.data = []
        self.targets = []
        self.data_transform = data_transform
        self.target_transform = target_transform
        
        for (idx, (data, target)) in enumerate(dataset):
            if target in classes:
                self.data.append(np.array(data))
                self.targets.append(np.array(target, dtype=int))
        self.num_samples = len(self.data)
        assert self.num_samples == len(self.targets)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

class LowQualityDataset(Dataset):
    def __init__(self,
                 high_quality_dataset,
                 relevant_classes,
                 irrelevant_classes,
                 proportion_correct=1.0,
                 data_transform=None,
                 target_transform=None,
                 relevance_transform=None,
                 correctness_transform=None):
        super().__init__()
        
        # Original dataset which will be modified
        self.data = []
        self.targets = []
        self.relevances = []
        self.correctnesses = []
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.relevance_transform = relevance_transform
        self.correctness_transform = correctness_transform
        
        for (idx, (data, target)) in enumerate(high_quality_dataset):
            if target in relevant_classes:
                # Randomly choose an incorrect label with probability p=1-proportion_correct
                if np.random.uniform(0, 1) > proportion_correct:
                    relevant_classes.remove(target)
                    self.targets.append(np.array(np.random.choice(relevant_classes), dtype=int))
                    relevant_classes.append(target)
                    self.correctnesses.append(np.array(0, dtype=int))
                else:
                    self.correctnesses.append(np.array(1, dtype=int))
                    self.targets.append(np.array(target, dtype=int))
                self.data.append(np.array(data))
                self.relevances.append(np.array(1, dtype=int))
            elif target in irrelevant_classes:
                # Randomly choose a valid label for this irrelevant example
                self.data.append(np.array(data))
                self.targets.append(np.array(np.random.choice(relevant_classes), dtype=int))
                self.correctnesses.append(np.array(0, dtype=int))
                self.relevances.append(np.array(0, dtype=int))
            else:
                continue
                
        self.num_samples = len(self.data)
        assert self.num_samples == len(self.targets)
        assert self.num_samples == len(self.relevances)
        assert self.num_samples == len(self.correctnesses)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        relevance = self.relevances[idx]
        correctness = self.correctnesses[idx]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.relevance_transform is not None:
            relevance = self.relevance_transform(relevance)
        if self.correctness_transform is not None:
            correctness = self.correctness_transform(correctness)
        return data, target, relevance, correctness
    
    def display_examples(self, num_examples):
        indices = np.random.choice(self.__len__(), (num_examples), replace=False)
        for idx in indices:
            img, target, relevance, correctness = self.__getitem__(idx)
            
            # Preprocess image to have channels as last dimension, and pixel intensity in [0, 1]
            img = img.permute(1, 2, 0)
            img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            
            # Plot image
            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(img, cmap='binary')
            ax.set_axis_off()
            fig.suptitle('Label: %d. Relevance: %d. Correctness: %d.'%(target, relevance, correctness))