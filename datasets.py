import os
from torchvision import datasets, transforms

def get_dataset(name):
    dataset_path = os.path.join('.', 'datasets')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    if name == 'CIFAR10':
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((.4914, .4822, .4465),
                                                                  (247, .243, .261))])
        training_dataset = datasets.CIFAR10(root=dataset_path,
                                            train=True,
                                            transform=data_transform,
                                            download=True)
        testing_dataset = datasets.CIFAR10(root=dataset_path,
                                           train=False,
                                           transform=data_transform,
                                           download=True)
    elif name == 'MNIST':
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((.1307,), (.3081,))])
        training_dataset = datasets.MNIST(root=dataset_path,
                                          train=True,
                                          transform=data_transform,
                                          download=True)
        testing_dataset = datasets.MNIST(root=dataset_path,
                                         train=False,
                                         transform=data_transform,
                                         download=True)
    elif name == 'FashionMNIST':
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((.5,), (.5,))])
        training_dataset = datasets.FashionMNIST(root=dataset_path,
                                                 train=True,
                                                 transform=data_transform,
                                                 download=True)
        testing_dataset = datasets.FashionMNIST(root=dataset_path,
                                                train=False,
                                                transform=data_transform,
                                                download=True)
    return training_dataset, testing_dataset