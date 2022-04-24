import os
import pickle
import argparse
from copy import copy
import json
from tqdm import tqdm
import shutil
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import get_resnet_feature_extractor, SelfTrainingModel, get_lenet_feature_extractor
from dataset import LowQualityDataset, GrayscaleToRgb, ValidationDataset
from train import naive_train_on_batch, naive_eval_on_batch, eval_epoch, train_epoch, update_res, display_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_initial_measure', dest='init_meas', action='store_false', default=True, help='Do not take initial model performance measurements prior to training.')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=None, help='Override the number of epochs in the .json files with this value.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=None, help='Override the batch size in the .json files with this value.')
    args = parser.parse_args()
    
    config_path = os.path.join('.', 'config')
    config_files = [f for f in os.listdir(config_path) if f.split('.')[-1]=='json']
    
    for config_file in config_files:
        with open(os.path.join(config_path, config_file), 'r') as F:
            config_params = json.load(F)
        dataset = config_params['dataset']
        relevant_classes = config_params['relevant_classes']
        irrelevant_classes = config_params['irrelevant_classes']
        proportion_correct = config_params['proportion_correct']
        if 'results_dir' in config_params:
            results_dir = config_params['results_dir']
        else:
            results_dir = config_file.split('.')[0]
        if args.num_epochs != None:
            num_epochs = args.num_epochs
        else:
            num_epochs = config_params['num_epochs']
        if args.batch_size != None:
            batch_size = args.batch_size
        else:
            batch_size = config_params['batch_size']
        if not 'technique' in config_params:
            technique = 'naive'
        else:
            technique = config_params['technique']
        if technique == 'ltrwe':
            num_validation_samples = config_params['num_validation_samples']
        
        print('Config file:', config_file)
        print('Dataset:', dataset)
        print('Relevant classes:', relevant_classes)
        print('Irrelevant_classes:', irrelevant_classes)
        print('Proportion correct:', proportion_correct)
        print('Results directory:', results_dir)
        print('Number of epochs:', num_epochs)
        print('Batch size:', batch_size)
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        num_channels = 3 if dataset == 'cifar10' else 1
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             #transforms.RandomAffine(20, translate=(.1, .1)),
                                             #transforms.RandomHorizontalFlip(p=.5),
                                             transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.247, 0.243, 0.261])])
        validation_data_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                             std=[0.247, 0.243, 0.261])])
        target_transform = None
        relevance_transform = None
        correctness_transform = None
        if dataset in ['mnist', 'fmnist', 'cifar10']:
            dataset_constructor = {'mnist': datasets.MNIST,
                                   'fmnist': datasets.FashionMNIST,
                                   'cifar10': datasets.CIFAR10}[dataset]
            training_dataset = dataset_constructor(root='./data',
                                                   train=True,
                                                   download=True,
                                                   transform=None)
            if technique == 'ltrwe':
                validation_indices = []
                for c in relevant_classes:
                    for example in range(num_validation_samples):
                        for idx in range(len(training_dataset)):
                            _, target = training_dataset[idx]
                            if (target == c) and not(idx in validation_indices):
                                validation_indices.append(idx)
                                break
                validation_dataset = Subset(training_dataset, validation_indices)
                validation_dataset = ValidationDataset(validation_dataset, validation_data_transform, target_transform)
                unclean_dataset = Subset(training_dataset, [i for i in range(len(training_dataset)) if not(i in validation_indices)])
            else:
                unclean_dataset = training_dataset
            testing_dataset = dataset_constructor(root='./data',
                                                  train=False,
                                                  download=True,
                                                  transform=None)
        low_quality_training_dataset = LowQualityDataset(unclean_dataset,
                                                         relevant_classes=relevant_classes,
                                                         irrelevant_classes=irrelevant_classes,
                                                         proportion_correct=proportion_correct,
                                                         data_transform=data_transform,
                                                         target_transform=target_transform,
                                                         relevance_transform=relevance_transform,
                                                         correctness_transform=correctness_transform)
        low_quality_training_dataloader = DataLoader(low_quality_training_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     pin_memory=True)
        low_quality_testing_dataset = LowQualityDataset(testing_dataset,
                                                        relevant_classes=relevant_classes,
                                                        irrelevant_classes=irrelevant_classes,
                                                        proportion_correct=proportion_correct,
                                                        data_transform=validation_data_transform,
                                                        target_transform=target_transform,
                                                        relevance_transform=relevance_transform,
                                                        correctness_transform=correctness_transform)
        low_quality_testing_dataloader = DataLoader(low_quality_testing_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True)
        if technique == 'ltrwe':
            validation_dataloader = DataLoader(validation_dataset,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               pin_memory=True)
        else:
            validation_dataloader = None
            
        (feature_extractor, num_features) = get_resnet_feature_extractor(num_blocks=18)#get_lenet_feature_extractor(num_channels)
        eg_image, _, _, _ = next(iter(low_quality_training_dataloader))
        _ = feature_extractor(eg_image)
        model = SelfTrainingModel(feature_extractor, num_features,
                                  num_classes=5,
                                  predict_quality=False,
                                  predict_correctness=False)
            
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        #optimizer = optim.SGD(model.parameters(), momentum=.9, lr=.001)
        optimizer = optim.Adam(model.parameters())
        device = 'cuda'
        model = model.to(device)
        training_results = {'loss_metrics': [[], [], [], []],
                            'acc_metrics': [[], [], [], []]}
        testing_results = {'loss_metrics': [[], [], [], []],
                           'acc_metrics': [[], [], [], []]}
        
        if args.init_meas:
            print('Calculating initial model performance...')
            lm, am = eval_epoch(low_quality_training_dataloader, model, loss_fn, device)
            update_res(training_results, lm, am)
            print('Training results:')
            display_res(lm, am)
            # Evaluate over all batches in testing dataset and record performance
            lm, am = eval_epoch(low_quality_testing_dataloader, model, loss_fn, device)
            update_res(testing_results, lm, am)
            print('Testing results:')
            display_res(lm, am)
            print()

        for epoch in range(num_epochs):
            print('Beginning epoch %d...'%(epoch+1))

            # Train over all batches in training dataset and record performance
            lm, am = train_epoch(low_quality_training_dataloader, model, loss_fn, optimizer, device, validation_dataloader)
            update_res(training_results, lm, am)
            print('Training results:')
            display_res(lm, am)

            # Evaluate over all batches in testing dataset and record performance
            lm, am = eval_epoch(low_quality_testing_dataloader, model, loss_fn, device)
            update_res(testing_results, lm, am)
            print('Testing results:')
            display_res(lm, am)
            print()
    
        results_dir = os.path.join('.', 'results', results_dir)
        if not(os.path.exists(results_dir)):
            os.mkdir(results_dir)

        with open(os.path.join(results_dir, 'training_results.pickle'), 'wb') as F:
            pickle.dump(training_results, F)
        with open(os.path.join(results_dir, 'testing_results.pickle'), 'wb') as F:
            pickle.dump(testing_results, F)

        #torch.save(model.state_dict(), os.path.join(results_dir, 'trained_model'))

        shutil.copy(os.path.join(config_path, config_file), os.path.join(results_dir, config_file))
        

if __name__ == '__main__':
    main()