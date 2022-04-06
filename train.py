import numpy as np
import torch
from torch import nn
import higher
import copy
from tqdm import tqdm

def compute_ltrwe_example_weights(training_batch, validation_batch, model, loss_fn, optimizer, device):
    training_images, training_labels = training_batch
    training_images = training_images.to(device)
    training_labels = training_labels.to(device)
    validation_images, validation_labels = validation_batch
    validation_images = validation_images.to(device)
    validation_labels = validation_labels.to(device)
    
    model_params_backup = copy.deepcopy(model.state_dict())
    optimizer_params_backup = copy.deepcopy(optimizer.state_dict())
    
    with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        training_loss = loss_fn(training_logits, training_labels)
        eps = torch.zeros_like(training_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_loss = torch.mean(loss_fn(validation_logits, validation_labels))
    eps_grad = torch.autograd.grad(validation_loss, eps)
    weights = nn.functional.relu(eps_grad)
    if torch.norm(weights) == 0:
        pass
    else:
        weights /= torch.sum(weights)
    
    model.load_state_dict(model_params_backup)
    optimizer.load_state_dict(optimizer_params_backup)
    
    return weights

def ltrwe_train_on_batch(training_batch, validation_batch, model, loss_fn, optimizer, device):
    example_weights = compute_ltrwe_example_weights(training_batch, validation_batch, model, loss_fn, optimizer, device)
    
    training_images, training_labels = training_batch
    batch_size = len(training_images)
    assert batch_size == len(training_labels)
    training_images = training_images.to(device)
    training_labels = training_labels.to(device)
    
    model.train()
    optimizer.zero_grad()
    (predictions, _, _) = model(training_images)
    loss = loss_fn(predictions, training_labels)
    reweighted_loss = torch.sum(loss*example_weights)
    reweighted_loss.backward()
    optimizer.step()
    
    reweighted_loss = float(reweighted_loss.detach().cpu())
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(training_labels.detach().cpu())
    num_correct = np.count_nonzero(np.equal(predictions, labels))
    accuracy = num_correct / batch_size
    
    return loss, accuracy

def eval_epoch(dataloader, model, loss_fn, device, print_output=False):
    rll, raa, ill, iaa = [], [], [], []
    for batch in tqdm(dataloader):
        rl, ra, il, ia = naive_eval_on_batch(batch, model, loss_fn, device, print_output)
        rll.append(rl)
        raa.append(ra)
        ill.append(il)
        iaa.append(ia)
    return {'relevant_loss': np.mean(rll),
            'relevant_acc': np.mean(raa),
            'irrelevant_loss': np.mean(ill),
            'irrelevant_acc': np.mean(iaa)}

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    rll, raa, ill, iaa = [], [], [], []
    for batch in tqdm(dataloader):
        rl, ra, il, ia = naive_train_on_batch(batch, model, loss_fn, optimizer, device)
        rll.append(rl)
        raa.append(ra)
        ill.append(il)
        iaa.append(ia)
    return {'relevant_loss': np.mean(rll),
            'relevant_acc': np.mean(raa),
            'irrelevant_loss': np.mean(ill), 
            'irrelevant_acc': np.mean(iaa)}

def update_res(res_d, res):
    for k in res_d.keys():
        res_d[k].append(res[k])

def display_res(res):
    for k in res:
        print(k, ':', res[k])

def naive_train_on_batch(batch, model, loss_fn, optimizer, device):
    # Get input data ready for the model
    images, labels, relevance, correctness = batch
    batch_size = len(images)
    assert len(labels) == batch_size
    images = images.to(device)
    labels = labels.to(device)
    
    # Make prediction and update model parameters based on results
    model.train()
    optimizer.zero_grad()
    (predictions, _, _) = model(images)
    elementwise_loss = loss_fn(predictions, labels)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    # Record how well the model performed
    elementwise_loss = elementwise_loss.detach().cpu()
    relevant_loss = torch.mean(elementwise_loss[relevance == 1])
    irrelevant_loss = torch.mean(elementwise_loss[relevance == 0])
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(labels.detach().cpu())
    correct = np.equal(predictions, labels)
    relevant_accuracy = np.mean(correct[relevance == 1])
    irrelevant_accuracy = np.mean(correct[relevance == 0])
    
    return relevant_loss, relevant_accuracy, irrelevant_loss, irrelevant_accuracy

def naive_eval_on_batch(batch, model, loss_fn, device, print_output=False):
    # Get input data ready for the model
    images, labels, relevance, correctness = batch
    batch_size = len(images)
    assert len(labels) == batch_size
    images = images.to(device)
    labels = labels.to(device)
    
    # Make prediction on batch
    model.eval()
    with torch.no_grad():
        (predictions, _, _) = model(images)
        elementwise_loss = loss_fn(predictions, labels)
        loss = torch.mean(elementwise_loss)
    
    # Record how well the model performed
    elementwise_loss = elementwise_loss.detach().cpu()
    relevant_loss = torch.mean(elementwise_loss[relevance == 1])
    irrelevant_loss = torch.mean(elementwise_loss[relevance == 0])
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(labels.detach().cpu())
    correct = np.equal(predictions, labels)
    relevant_accuracy = np.mean(correct[relevance == 1])
    irrelevant_accuracy = np.mean(correct[relevance == 0])
    if print_output:
        pass
    return relevant_loss, relevant_accuracy, irrelevant_loss, irrelevant_accuracy