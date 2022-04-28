from copy import deepcopy
import numpy as np
import torch
from torch import optim
import higher

def eval_on_batch(images, labels, model, loss_fn, device):
    model.eval()
    
    # Compute model logits and loss
    with torch.no_grad():
        logits = model(images)
        elementwise_loss = loss_fn(logits, labels)
    
    # Detach and return results of evaluating on this batch
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return elementwise_loss, predictions

def sss_train_on_batch(training_images, training_labels,
                       validation_images, validation_labels,
                       model,
                       loss_fn,
                       optimizer,
                       device):
    model.train()
    
    # Compute example weight gradients based on dataset labels and model's predictions as labels
    model_params_backup = deepcopy(model.state_dict())
    dummy_optimizer = optim.SGD(model.parameters(), lr=0.001)
    with higher.innerloop_ctx(model, dummy_optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        self_generated_labels = torch.argmax(training_logits.detach(), dim=-1)
        labels = torch.cat((self_generated_labels, training_labels))
        training_elementwise_loss = loss_fn(torch.cat((training_logits, training_logits)), labels)
        eps = torch.zeros_like(training_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_elementwise_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_elementwise_loss = loss_fn(validation_logits, validation_labels)
        reduced_validation_loss = torch.mean(validation_loss)
    eps_grad = torch.autograd.grad(reduced_validation_loss, eps)[0].detach()
    model.load_state_dict(model_params_backup)
    
    # Find the label/eps_grad pairs which maximize the eps_grads
    self_generated_labels_eps_grad, dataset_labels_eps_grad = torch.tensor_split(eps_grad, 2)
    label_options = torch.stack((self_generated_labels, training_labels))
    eps_grad_options = torch.stack((self_generated_labels_eps_grad, dataset_labels_eps_grad))
    use_dataset_labels_idx = torch.argmax(-eps_grad_options, dim=0).unsqueeze(0)
    labels = torch.gather(label_options, 0, use_dataset_labels_idx).squeeze()
    eps_grad = torch.gather(eps_grad_options, 0, use_dataset_labels_idx).squeeze()
    
    # Compute example weights
    weights = nn.functional.relu(-eps_grad)
    if torch.norm(weights) != 0:
        weights /= torch.sum(weights)
    
    # Train model using computed labels and example weights
    optimizer.zero_grad()
    logits = model(training_images)
    elementwise_loss = loss_fn(logits, labels)
    reweighted_loss = torch.sum(elementwise_loss*weights)
    reweighted_loss.backward()
    optimizer.step()
    
    # Detach and return the results of training on this batch
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    weights = weights.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return elementwise_loss, predictions, weights, labels

# Based on code here:
#  https://github.com/TinfoilHat0/Learning-to-Reweight-Examples-for-Robust-Deep-Learning-with-PyTorch-Higher
def ltrwe_train_on_batch(training_images, training_labels,
                         validation_images, validation_labels,
                         model,
                         loss_fn,
                         optimizer,
                         device):
    model.train()
    
    # Compute example weights using technique described in LTRWE paper
    model_params_backup = deepcopy(model.state_dict())
    dummy_optimizer = optim.SGD(model.parameters(), lr=0.001)
    with higher.innerloop_ctx(model, dummy_optimizer) as (fmodel, diffopt):
        training_logits = fmodel(training_images)
        training_elementwise_loss = loss_fn(training_logits, training_labels)
        eps = torch.zeros_like(training_elementwise_loss, device=device, requires_grad=True)
        reweighted_loss = torch.sum(training_elementwise_loss*eps)
        diffopt.step(reweighted_loss)
        validation_logits = fmodel(validation_images)
        validation_elementwise_loss = loss_fn(validation_logits, validation_labels)
        reduced_validation_loss = torch.mean(validation_elementwise_loss)
    eps_grad = torch.autograd.grad(reduced_validation_loss, eps)[0].detach()
    weights = nn.functional.relu(-eps_grad)
    if torch.norm(weights) != 0:
        weights /= torch.sum(weights)
    model.load_state_dict(model_params_backup)
    
    # Optimize model on reweighted batch
    optimizer.zero_grad()
    logits = model(training_images)
    elementwise_loss = loss_fn(logits, training_labels)
    reweighted_loss = torch.sum(elementwise_loss*weights)
    reweighted_loss.backward()
    optimizer.step()
    
    # Detach and return results of training on this batch
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    weights = weights.detach().cpu().numpy()
    return elementwise_loss, predictions, weights

def naive_train_on_batch(images, labels, model, loss_fn, optimizer, device):
    # Optimize model on batch
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    elementwise_loss = loss_fn(logits, labels)
    reduced_loss = torch.mean(elementwise_loss)
    reduced_loss.backward()
    optimizer.step()
    
    # Detach and return results of training on this batch
    elementwise_loss = elementwise_loss.detach().cpu().numpy()
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return elementwise_loss, predictions