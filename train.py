import numpy as np
import torch

def naive_train_on_batch(batch, model, loss_fn, optimizer, device):
    # Get input data ready for the model
    images, labels, _, _ = batch
    batch_size = len(images)
    assert len(labels) == batch_size
    images = images.to(device)
    labels = labels.to(device)
    
    # Make prediction and update model parameters based on results
    model.train()
    optimizer.zero_grad()
    (predictions, _, _) = model(images)
    loss = loss_fn(predictions, labels)
    loss.backward()
    optimizer.step()
    
    # Record how well the model performed
    loss = float(loss.detach().cpu())
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(labels.detach().cpu())
    num_correct = np.count_nonzero(np.equal(predictions, labels))
    accuracy = num_correct / batch_size
    
    return loss, accuracy

def naive_eval_on_batch(batch, model, loss_fn, device):
    # Get input data ready for the model
    images, labels, _, _ = batch
    batch_size = len(images)
    assert len(labels) == batch_size
    images = images.to(device)
    labels = labels.to(device)
    
    # Make prediction on batch
    model.eval()
    with torch.no_grad():
        (predictions, _, _) = model(images)
        loss = loss_fn(predictions, labels)
    
    # Record how well the model performed
    loss = float(loss.detach().cpu())
    predictions = np.array(np.argmax(predictions.detach().cpu(), axis=1))
    labels = np.array(labels.detach().cpu())
    num_correct = np.count_nonzero(np.equal(predictions, labels))
    accuracy = num_correct / batch_size
    
    return loss, accuracy