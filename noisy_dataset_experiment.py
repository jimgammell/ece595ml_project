import time
from torch.utils.data import Dataset

from utils import set_random_seed, log_print as print
from datasets import get_dataset, get_dataset_classes

def run_trial(config_params):
    trial_start_time = time.time()
    
    # Parse and validate trial_args
    method = config_params['method']
    assert method in ['naive', 'ltrwe', 'sss']
    dataset = config_params['dataset']
    assert dataset in ['CIFAR10', 'MNIST', 'FashionMNIST']
    if 'val_dataset_size' in config_params:
        val_set_size = config_params['val_dataset_size']
    else:
        val_set_size = 10000
    assert (type(val_set_size) == int) and (val_set_size > 0)
    correct_samples_per_class = config_params['correct_samples_per_class']
    assert (type(correct_samples_per_class) == int) and (correct_samples_per_class >= 0)
    incorrect_samples_per_class = config_params['incorrect_samples_per_class']
    assert (type(incorrect_samples_per_class) == int) and (incorrect_samples_per_class >= 0)
    if seed in config_params:
        seed = config_params['seed']
    else:
        seed = time.time_ns() & 0xFFFFFFFF
    assert type(seed) == int
    train_dataloader_kwargs = config_params['train_dataloader_kwargs']
    assert type(train_dataloader_kwargs) == dict
    eval_dataloader_kwargs = config_params['eval_dataloader_kwargs']
    assert type(eval_dataloader_kwargs) == dict
    if method in ['ltrwe', 'sss']:
        clean_dataloader_kwargs = config_params['clean_dataloader_kwargs']
        assert type(clean_dataloader_kwargs) == dict
        clean_dataset_samples_per_class = config_params['clean_samples_per_class']
    else:
        clean_dataloader_kwargs = None
        clean_dataset_samples_per_class = None
    model_constructor_attr = config_params['model_constructor']
    assert model_constructor_attr in ['LeNet5', 'ResNet']
    if 'model_constructor_kwargs' in config_params:
        model_constructor_kwargs = config_params['model_constructor_kwargs']
    else:
        model_constructor_kwargs = {}
    assert type(model_constructor_kwargs) == dict
    loss_fn_constructor_attr = config_params['loss_fn_constructor']
    assert type(loss_fn_constructor_attr) == str
    if 'loss_fn_kwargs' in config_params:
        loss_fn_kwargs = config_params['loss_fn_kwargs']
    else:
        loss_fn_kwargs = {}
    assert type(loss_fn_kwargs) == dict
    optimizer_constructor_attr = config_params['optimizer_constructor']
    assert type(optimizer_constructor_attr) == str
    if optimizer_kwargs in config_params:
        optimizer_kwargs = config_params['optimizer_kwargs']
    else:
        optimizer_kwargs = {}
    assert type(optimizer_kwargs) == dict
    num_epochs = config_params['num_epochs']
    assert (type(num_epochs) == int) and (num_epochs >= 0)
    if 'device' in config_params:
        device = config_params['device']
    else:
        device = 'cpu'
    assert device in ['cpu', 'cuda']
    init_meas = config_params['eval_initial_performance']
    assert type(init_meas) == bool
    
    print('Beginning noisy dataset experiment.')
    print('\tMethod: {}'.format(method))
    print('\tDataset: {}'.format(dataset))
    print('\tValidation set size: {}'.format(val_set_size))
    print('\tCorrect samples per class: {}'.format(correct_samples_per_class))
    print('\tIncorrect samples per class: {}'.format(incorrect_samples_per_class))
    print('\tRandom seed: {}'.format(seed))
    print('\tTraining dataloader kwargs: {}'.format(train_dataloader_kwargs))
    print('\tEval dataloader kwargs: {}'.format(eval_dataloader_kwargs))
    if clean_dataloader_kwargs != None:
        print('\tClean dataloader kwargs: {}'.format(clean_dataloader_kwargs))
        print('\tClean samples per class: {}'.format(clean_dataset_samples_per_class))
    print('\tModel constructor: {}'.format(model_constructor_attr))
    print('\tModel constructor kwargs: {}'.format(model_constructor_kwargs))
    print('\tLoss function constructor: {}'.format(loss_fn_constructor_attr))
    print('\tLoss function constructor kwargs: {}'.format(loss_fn_kwargs))
    print('\tOptimizer constructor: {}'.format(optimizer_constructor))
    print('\tOptimizer constructor kwargs: {}'.format(optimizer_constructor_kwargs))
    print('\tNumber of epochs: {}'.format(num_epochs))
    print('\tDevice: {}'.format(device))
    print('\tConduct initial measurements: {}'.format(init_meas))
    print()
    
    set_random_seed(seed)
    
    train_dataset, val_dataset, test_dataset = get_dataset(dataset, val_set_size)
    classes = get_dataset_classes(dataset)
    if method in ['ltrwe', 'sss']:
        clean_dataset = CleanDataset(train_dataset, classes, clean_dataset_samples_per_class)

class CleanDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 classes,
                 samples_per_class,
                 transform=None,
                 target_transform=None):
        self.images = []
        self.targets = []
        self.trasform = transform
        self.target_transform = target_transform
        
        samples_to_go = {c: samples_per_class for c in classes}
        base_dataset_transform = base_dataset.transform
        base_dataset_target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        for (image, target) in base_dataset:
            if samples_to_go[target] > 0:
                self.images.append(image)
                self.targets.append(target)
                samples_to_go[target] -= 1
        base_dataset.transform = base_dataset_transform
        base_dataset.target_transform = base_dataset_target_transform
        self.number_of_samples = len(self.images)
        assert self.number_of_samples == len(self.targets)
        assert all([samples_to_go[c] == 0 for c in samples_to_go.keys()])
    
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform != None:
            image = self.transform(image)
        if self.target_transform != None:
            target = self.target_transform(target)
        return image, target

class NoisyLabelsDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 classes,
                 correct_samples_per_class,
                 incorrect_samples_per_class):
        self.images = []
        self.noisy_targets = []
        self.correct_targets = []
        self.correctness = []
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        base_dataset.transform = None
        base_dataset.target_transform = None
        
        correct_samples_to_go = {c: correct_samples_per_class for c in classes}
        incorrect_samples_to_go = {c: incorrect_samples_per_class for c in classes}
        for (image, target) in base_dataset:
            if correct_samples_to_go[target] > 0:
                self.images.append(image)
                self.noisy_targets.append(target)
                self.correct_targets.append(target)
                self.correctness.append(1)
                correct_samples_to_go[target] -= 1
            elif noisy_samples_to_go[target] > 0:
                self.images.append(image)
                incorrect_classes = [c for c in self.classes if c != target]
                self.noisy_targets.append(random.choice(incorrect_classes))
                self.correct_targets.append(target)
                self.correctness.append(0)
                self.incorrect_samples_to_go[target] -= 1
        self.number_of_samples = len(self.images)
        assert self.number_of_samples == len(self.noisy_targets)
        assert self.number_of_samples == len(self.correct_targets)
        assert self.number_of_samples == len(self.correctness)
        assert all([correct_samples_to_go[c] == 0 for c in correct_samples_to_go.keys()])
        assert all([incorrect_samples_to_go[c] == 0 for c in incorrect_samples_to_go.keys()])
    
    def __getitem__(self, idx):
        image = self.images[idx]
        noisy_target = self.noisy_targets[idx]
        correct_target = self.correct_targets[idx]
        correctness = self.correctness[idx]
        if self.transform != None:
            image = self.transform(image)
        if self.target_transform != None:
            noisy_target = self.target_transform(noisy_target)
        return image, noisy_target, correct_target, correctness
    
    def __len__(self):
        return self.number_of_samples