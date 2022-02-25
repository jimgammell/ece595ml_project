import numpy as np
import torch
from torchvision import datasets, transforms

def dataset_to_numpy(dataset):
    X = []
    Y = []
    for (x, y) in dataset:
        X.append(np.array(x))
        Y.append(np.array(y))
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def dataset_to_tensors(dataset):
    (X, Y) = dataset
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y).to(torch.long)
    dataset = torch.utils.data.TensorDataset(X, Y)
    return dataset

def load_mnist(to_numpy=False):
    train_dataset = datasets.MNIST('./data',
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())
    test_dataset  = datasets.MNIST('./data',
                                   train=False,
                                   download=True,
                                   transform=transforms.ToTensor())
    if to_numpy:
        train_dataset = dataset_to_numpy(train_dataset)
        test_dataset = dataset_to_numpy(test_dataset)
    return (train_dataset, test_dataset)

def load_fashion_mnist(to_numpy=False):
    train_dataset = datasets.FashionMNIST('./data',
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())
    test_dataset  = datasets.FashionMNIST('./data',
                                          train=False,
                                          download=True,
                                          transform=transforms.ToTensor())
    if to_numpy:
        train_dataset = dataset_to_numpy(train_dataset)
        test_dataset = dataset_to_numpy(test_dataset)
    return (train_dataset, test_dataset)

def extract_classes(dataset, classes):
    X, Y = dataset
    class_datasets = {}
    for c in classes:
        X_c = []
        for (x, y) in zip(X, Y):
            if y == c:
                X_c.append(x)
        X_c = np.array(X_c)
        Y_c = c*(np.ones(X_c.shape[0]))
        class_datasets.update({c: (X_c, Y_c)})
    return class_datasets

def generate_unbalanced_dataset(dataset, classes, props, output_size):
    if len(classes) != len(props):
        raise Exception('Classes {} and proportions {} must have same dimensions.'.format(classes, props))
    dataset = dataset_to_numpy(dataset)
    class_datasets = extract_classes(dataset, classes)
    samples_per_class = [int(output_size*p) for p in props]
    output_X = []
    output_Y = []
    for n, class_key in zip(samples_per_class, class_datasets):
        samples_available = len(class_datasets[class_key][0])
        if n > samples_available:
            raise Exception('Tried to extract %d samples from class %d, but only %d samples exist.'%(n, class_key, samples_available))
        else:
            indices = np.random.choice(np.arange(samples_available), n, replace=False)
            output_X.append(class_datasets[class_key][0][indices])
            output_Y.append(class_datasets[class_key][1][indices])
    output_X = np.concatenate(output_X, axis=0)
    output_Y = np.concatenate(output_Y, axis=0)
    dataset = dataset_to_tensors((output_X, output_Y))
    return dataset