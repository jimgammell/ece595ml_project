import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class GrayscaleToRgb:
    def __init__(self):
        pass
    def __call__(self, x):
        x = x.repeat(3, 1, 1)
        return x

class LowQualityDataset(Dataset):
    def __init__(self,
                 high_quality_dataset,
                 relevant_classes=[0, 1, 2, 3, 4],
                 proportion_correct=1.0,
                 relevance_transform=None,
                 correctness_transform=None):
        super().__init__()
        
        # Original dataset which will be modified
        self.high_quality_dataset = high_quality_dataset
        self.relevances = torch.ones_like(self.high_quality_dataset.targets, dtype=torch.int)
        self.correctnesses = torch.ones_like(self.high_quality_dataset.targets, dtype=torch.int)
        self.relevance_transform = relevance_transform
        self.correctness_transform = correctness_transform
        
        for (idx, target) in enumerate(self.high_quality_dataset.targets):
            if target in relevant_classes:
                # Randomly choose an incorrect label with probability p=1-proportion_correct
                if np.random.uniform(0, 1) > proportion_correct:
                    relevant_classes.remove(target)
                    self.high_quality_dataset.targets[idx] = np.random.choice(relevant_classes)
                    relevant_classes.append(target)
                    self.correctnesses[idx] = 0
            else:
                # Randomly choose a valid label for this irrelevant example
                self.high_quality_dataset.targets[idx] = np.random.choice(relevant_classes)
                self.correctnesses[idx] = 0
                self.relevances[idx] = 0
        
    def __len__(self):
        return self.high_quality_dataset.__len__()
    
    def __getitem__(self, idx):
        img, target = self.high_quality_dataset.__getitem__(idx)
        relevance = int(self.relevances[idx])
        correctness = int(self.correctnesses[idx])
        if self.relevance_transform is not None:
            relevance = self.relevance_transform(relevance)
        if self.correctness_transform is not None:
            correctness = self.correctness_transform(correctness)
        return img, target, relevance, correctness
    
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