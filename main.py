import os
import pickle
import copy
import json
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import get_resnet_feature_extractor, SelfTrainingModel
from dataset import LowQualityDataset, GrayscaleToRgb
from train import naive_train_on_batch, naive_eval_on_batch

def main():
    config_path = os.path.join('.', 'config')
    config_files = [f for f in os.listdir(config_path) if '.json' in f]
    
    for config_file in config_files:
        with open(os.path.join(config_path, config_file), 'r') as F:
            config_params = json.load(F)
        dataset = config_params['dataset']
        relevant_classes = config_params['relevant_classes']
        irrelevant_classes = config_params['irrelevant_classes']
        proportion_correct = config_params['proportion_correct']
        results_dir = config_params['results_dir']
        num_epochs = config_params['num_epochs']
        batch_size = config_params['batch_size']
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
        
        (feature_extractor, num_features) = get_resnet_feature_extractor(num_blocks=18, pretrained=True, freeze_weights=True)
        model = SelfTrainingModel(feature_extractor, num_features,
                                  num_classes=5,
                                  predict_quality=False,
                                  predict_correctness=False)
        
        if dataset in ['mnist', 'fmnist']:
            data_transform = transforms.Compose([transforms.ToTensor(),
                                                 GrayscaleToRgb(),
                                                 transforms.Resize((224, 224)),
                                                 transforms.Normalize(mean=[.485, .456, .406],
                                                                      std=[.229, .224, .225])])
        else:
            data_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((224, 224)),
                                                 transforms.Normalize(mean=[.485, .456, .406],
                                                                      std=[.229, .224, .225])])
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
                targets = list(copy(training_dataset.targets))
                for c in range(10):
                    for example in range(num_validation_samples):
                        for (idx, target) in enumerate(targets):
                            if target == c:
                                del targets[idx]
                                if c in relevant_classes:
                                    validation_indices.append(idx)
                validation_dataset = Subset(training_dataset, validation_indices)
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
                                                     shuffle=True)
        low_quality_testing_dataset = LowQualityDataset(testing_dataset,
                                                        relevant_classes=relevant_classes,
                                                        irrelevant_classes=[],
                                                        proportion_correct=proportion_correct,
                                                        data_transform=data_transform,
                                                        target_transform=target_transform,
                                                        relevance_transform=relevance_transform,
                                                        correctness_transform=correctness_transform)
        low_quality_testing_dataloader = DataLoader(low_quality_testing_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        if technique == 'ltrwe':
            validation_dataloader = DataLoader(validation_dataset,
                                               batch_size = batch_size,
                                               shuffle=True)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        device = 'cuda'
        model = model.to(device)
        Results = {'train_loss': [],
                   'test_loss': [],
                   'train_acc': [],
                   'test_acc': []}
        print('Calculating initial model performance...')
        train_loss_over_epoch = []
        train_acc_over_epoch = []
        for batch in tqdm(low_quality_training_dataloader):
            loss, acc = naive_eval_on_batch(batch, model, loss_fn, device)
            train_loss_over_epoch.append(loss)
            train_acc_over_epoch.append(acc)
        Results['train_loss'].append(np.mean(train_loss_over_epoch))
        Results['train_acc'].append(np.mean(train_acc_over_epoch))
        # Evaluate over all batches in testing dataset and record performance
        test_loss_over_epoch = []
        test_acc_over_epoch = []
        for batch in tqdm(low_quality_testing_dataloader):
            loss, acc = naive_eval_on_batch(batch, model, loss_fn, device)
            test_loss_over_epoch.append(loss)
            test_acc_over_epoch.append(acc)
        Results['test_loss'].append(np.mean(test_loss_over_epoch))
        Results['test_acc'].append(np.mean(test_acc_over_epoch))
        print('\tTraining loss: %e'%(Results['train_loss'][-1]))
        print('\tTraining accuracy: %f'%(100*Results['train_acc'][-1]))
        print('\tTesting loss: %e'%(Results['test_loss'][-1]))
        print('\tTesting accuracy: %f'%(100*Results['test_acc'][-1]))

        for epoch in range(num_epochs):
            print('Beginning epoch %d...'%(epoch+1))

            # Train over all batches in training dataset and record performance
            train_loss_over_epoch = []
            train_acc_over_epoch = []
            if technique == 'ltrwe':
                for (training_batch, validation_batch) in tqdm(zip(low_quality_training_dataloader, validation_dataloader)):
                    loss, acc = ltrwe_train_on_batch(training_batch, validation_batch, model, loss_fn, optimizer, device)
                    train_loss_over_epoch.append(loss)
                    train_acc_over_epoch.append(acc)
            else:
                for batch in tqdm(low_quality_training_dataloader):
                    loss, acc = naive_train_on_batch(batch, model, loss_fn, optimizer, device)
                    train_loss_over_epoch.append(loss)
                    train_acc_over_epoch.append(acc)
            Results['train_loss'].append(np.mean(train_loss_over_epoch))
            Results['train_acc'].append(np.mean(train_acc_over_epoch))

            # Evaluate over all batches in testing dataset and record performance
            test_loss_over_epoch = []
            test_acc_over_epoch = []
            for batch in tqdm(low_quality_testing_dataloader):
                loss, acc = naive_eval_on_batch(batch, model, loss_fn, device)
                test_loss_over_epoch.append(loss)
                test_acc_over_epoch.append(acc)
            Results['test_loss'].append(np.mean(test_loss_over_epoch))
            Results['test_acc'].append(np.mean(test_acc_over_epoch))

            # Report results
            print('\tTraining loss: %e'%(Results['train_loss'][-1]))
            print('\tTraining accuracy: %f'%(100*Results['train_acc'][-1]))
            print('\tTesting loss: %e'%(Results['test_loss'][-1]))
            print('\tTesting accuracy: %f'%(100*Results['test_acc'][-1]))
    
        results_dir = os.path.join('.', 'results', results_dir)
        if not(os.path.exists(results_dir)):
            os.mkdir(results_dir)

        with open(os.path.join(results_dir, 'results.pickle'), 'wb') as F:
            pickle.dump(Results, F)

        torch.save(model.state_dict(), os.path.join(results_dir, 'trained_model'))

        os.rename(os.path.join(config_path, config_file), os.path.join(results_dir, config_file))
        

if __name__ == '__main__':
    main()