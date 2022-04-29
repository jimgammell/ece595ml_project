import os
from torchvision import datasets, transforms

def get_dataset(name):
    dataset_path = os.path.join('.', 'datasets')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    if name == 'CIFAR10':
        train_transform = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                              #transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                  (0.2023, 0.1994, 0.2010))])
        training_dataset = datasets.CIFAR10(root=dataset_path,
                                            train=True,
                                            transform=train_transform,
                                            download=True)
        testing_dataset = datasets.CIFAR10(root=dataset_path,
                                           train=False,
                                           transform=test_transform,
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