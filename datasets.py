import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split

# Default PyTorch subsets must have the same transforms as the base dataset which is inconvenient for partitioning a dataset into a
#  training and validation set. This is a wrapper for a subset which allows its transforms to be uniquely specified. Base dataset
#  should not have transforms in this case.
class SubsetWithTransforms(Dataset):
    def __init__(self, subset,
                 transform=None,
                 target_transform=None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        image, target = self.subset.__getitem__(idx)
        if self.transform != None:
            image = self.transform(image)
        if self.target_transform != None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        return self.subset.__len__()

def get_dataset_classes(name):
    if name == 'CIFAR10':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif name == 'MNIST':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif name == 'FashionMNIST':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        assert False
    
def get_dataset(name, val_size=10000):
    dataset_path = os.path.join('.', 'datasets')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    if name == 'CIFAR10':
        training_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
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
        
    full_dataset = dataset_constructor(root=dataset_path,
                                       train=True,
                                       transform=None,
                                       download=True)
    train_dataset, val_dataset = random_split(full_dataset, [len(full_dataset)-val_size, val_size])
    training_dataset = SubsetWithTransforms(training_dataset, transform=training_transform)
    val_dataset = SubsetWithTransforms(val_dataset, transform=evaluation_transform)
    test_dataset = dataset_constructor(root=dataset_path,
                                       train=True,
                                       transform=evaluation_transform,
                                       download=True)
        
    return train_dataset, val_dataset, test_dataset
        