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

def eval_epoch(dataloader, model, loss_fn, device):
    loss_metrics, accuracy_metrics = [[] for _ in range(4)], [[] for _ in range(4)]
    for batch in tqdm(dataloader):
        l, a = naive_eval_on_batch(batch, model, loss_fn, device)
        for (idx, (ll, aa)) in enumerate(zip(l, a)):
            loss_metrics[idx].append(ll)
            accuracy_metrics[idx].append(aa)
    for (idx, (l, a)) in enumerate(zip(loss_metrics, accuracy_metrics)):
        loss_metrics[idx] = np.mean(l)
        accuracy_metrics[idx] = np.mean(a)
    return (loss_metrics, accuracy_metrics)

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    loss_metrics, accuracy_metrics = [[] for _ in range(4)], [[] for _ in range(4)]
    for batch in tqdm(dataloader):
        l, a = naive_train_on_batch(batch, model, loss_fn, optimizer, device)
        for (idx, (ll, aa)) in enumerate(zip(l, a)):
            loss_metrics[idx].append(ll)
            accuracy_metrics[idx].append(aa)
    for (idx, (l, a)) in enumerate(zip(loss_metrics, accuracy_metrics)):
        loss_metrics[idx] = np.mean(l)
        accuracy_metrics[idx] = np.mean(a)
    return (loss_metrics, accuracy_metrics)

def update_res(res_dict, loss_res, acc_res):
    for (l, idx) in zip(loss_res, range(len(res_dict['loss_metrics']))):
        res_dict['loss_metrics'][idx].append(l)
    for (a, idx) in zip(acc_res, range(len(res_dict['acc_metrics']))):
        res_dict['acc_metrics'][idx].append(a)

def display_res(loss_res, acc_res):
    for (res, s) in [(loss_res, 'Loss:'), (acc_res, 'Accuracy:')]:
        print(s)
        print('\trelevant/correct:', res[0])
        print('\tirrelevant/correct:', res[1])
        print('\trelevant/incorrect:', res[2])
        print('\tirrelevant/incorrect:', res[3])

def compute_metrics(m, relevance, correctness):
    m11 = np.mean(m[relevance*correctness == 1])
    m01 = np.mean(m[(1-relevance)*correctness == 1])
    m10 = np.mean(m[relevance*(1-correctness) == 1])
    m00 = np.mean(m[(1-relevance)*(1-correctness) == 1])
    return (m11, m01, m10, m00)
        
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
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    loss_metrics = compute_metrics(elementwise_loss, relevance, correctness)
    
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(labels.detach().cpu())
    correct = np.equal(predictions, labels)
    accuracy_metrics = compute_metrics(correct, relevance, correctness)
    
    return (loss_metrics, accuracy_metrics)

def naive_eval_on_batch(batch, model, loss_fn, device):
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
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    loss_metrics = compute_metrics(elementwise_loss, relevance, correctness)
    
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(labels.detach().cpu())
    correct = np.equal(predictions, labels)
    accuracy_metrics = compute_metrics(correct, relevance, correctness)
    
    return (loss_metrics, accuracy_metrics)