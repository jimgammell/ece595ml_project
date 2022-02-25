import numpy as np
import torch
from torch import nn
from tqdm import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device):
    loss_values = []
    model.train()
    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        #y = nn.functional.one_hot(y, 10)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.cpu().detach().numpy()/len(y))
    return loss_values

def evaluate_step(model, dataloader, loss_fn, device, roc):
    model.eval()
    results = {'true_0': 0,
               'false_0': 0,
               'true_1': 0,
               'false_1': 0,
               'loss': 0,
               'accuracy': 0}
    with torch.no_grad():
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            #y = nn.functional.one_hot(y, 10)
            output = model(x)
            loss = loss_fn(output, y)
            results['loss'] += loss.cpu().detach().numpy()
            prediction = np.argmax(output, axis=1)
            results['accuracy'] += np.count_nonzero(prediction==y)
            if roc:
                results['true_0'] += np.count_nonzero((prediction==0)[y==0])
                results['false_0'] += np.count_nonzero((prediction==0)[y!=0])
                results['true_1'] += np.count_nonzero((prediction==1)[y==1])
                results['false_1'] += np.count_nonzero((prediction==1)[y!=1])
    for key in results:
        results[key] /= len(dataloader.dataset)
    return results

def train(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          device,
          n_epochs,
          roc=False):
    results = {
        'train_loss': {},
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []}
    if roc:
        results.update({'test_roc': {'true_0': [], 'false_0': [], 'true_1': [], 'false_1': []}})
    res = evaluate_step(model, train_dataloader, loss_fn, device, roc)
    results['train_loss'].update({0: [res['loss']]})
    results['train_accuracy'].append(res['accuracy'])
    res = evaluate_step(model, test_dataloader, loss_fn, device, roc)
    results['test_loss'].append(res['loss'])
    results['test_accuracy'].append(res['accuracy'])
    if roc:
        for key in results['test_roc']:
            results['test_roc'][key].append(res[key])
    with tqdm(range(1, n_epochs+1)) as t:
        for epoch in t:
            res = train_step(model, train_dataloader, loss_fn, optimizer, device)
            results['train_loss'].update({epoch: res})
            res = evaluate_step(model, train_dataloader, loss_fn, device, roc)
            results['train_accuracy'].append(res['accuracy'])
            res = evaluate_step(model, test_dataloader, loss_fn, device, roc)
            results['test_loss'].append(res['loss'])
            results['test_accuracy'].append(res['accuracy'])
            if roc:
                for key in results['test_accuracy']:
                    results['test_accuracy'][key].append(res[key])
    return results
        