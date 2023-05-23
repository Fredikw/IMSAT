#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Libraries

"""

import csv
from datetime import datetime
from typing import Callable

import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score

# In[3]:


"""
Setting generic hyperparameters

"""

num_epochs: int = 25
batch_size: int = 128   # Should be set to a power of 2.
# Learning rate
lr:         float = 1e-4 # Learning rate used in the IIC paper: lr=1e-4.

"""
GPU utilization

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specifications
if torch.cuda.is_available():
    print(f"Number of available devices: {torch.cuda.device_count()}\n",
          f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}\n",
          f"Total GPU memory device 0: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.2f} GB\n")


# In[4]:


'''
Store data to .csv file

'''

# open the file for writing
f = open(f'logs/{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv', 'w')
# create a CSV writer object
writer = csv.writer(f)
# write the header row to the CSV file
writer.writerow(['epoch', 'loss', 'running_acc', 'acc', 'running_nmi', 'nmi'])


# In[5]:


"""
Unsupervised Machine Learning Framework

"""

def train(model, data_loader: DataLoader, criterion: Callable, optimizer: torch.optim, num_epochs: int) -> None:
    """
    Trains a given model using the provided training data, optimizer and loss criterion for a given number of epochs.

    Args:
        model: Neural network model to train.
        data_loader: PyTorch data loader containing the training data.
        criterion: Loss criterion used for training the model.
        optimizer: Optimizer used to update the model's parameters.
        num_epochs: Number of epochs to train the model.

    Returns:
        None
    """

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_acc  = 0.0
        running_nmi  = 0.0

        # Initialize tensors for storing true and predicted labels
        labels_true = torch.zeros(len(data_loader.dataset))
        labels_pred = torch.zeros(len(data_loader.dataset))

        # Loop over the mini-batches in the data loader
        for i, data in enumerate(data_loader):
        
            # Get the inputs and labels for the mini-batch
            inputs, labels = data

            # Use GPU if available
            inputs = inputs.to(device)

            # Image augmentation
            if data_loader.dataset.augment_data:
                inputs_trans = torch.stack([data_loader.dataset.transform_list(input) for input in inputs])
                # # Flatten input data for the feed forward model
                # inputs       = [inputs.view(inputs.size(0), -1), inputs_trans.view(inputs_trans.size(0), -1)]
                inputs       = [inputs, inputs_trans]
            # else:
                # inputs = inputs.view(inputs.size(0), -1)
        
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the model
            if data_loader.dataset.augment_data:
                outputs = [F.softmax(model(inputs[0]), dim=1), F.softmax(model(inputs[1]), dim=1)]
            else:
                outputs = F.softmax(model(inputs), dim=1)

            # Set arguments for objective function
            # kwargs = {key: value for key, value in locals().items() if key in criterion.__code__.co_varnames}
            kwargs = {"model": model, "inputs": inputs, "outputs": outputs}
            kwargs = {key: value for key, value in kwargs.items() if key in criterion.__code__.co_varnames}
            
            # Compute the loss
            loss = criterion(**kwargs)
            # Backward pass through the model and compute gradients
            loss.backward()
        
            # Update the weights
            optimizer.step()

            # Accumulate the loss for the mini-batch
            running_loss += loss.item()

            outputs = outputs[0] if data_loader.dataset.augment_data else outputs

            running_acc  += unsupervised_clustering_accuracy(labels, torch.argmax(outputs.cpu(), dim=1))
            running_nmi  += normalized_mutual_info_score(labels, torch.argmax(outputs.cpu(), dim=1))

            # Store predicted and true labels in tensors
            labels_true[i*len(labels):(i+1)*len(labels)] = labels
            labels_pred[i*len(labels):(i+1)*len(labels)] = torch.argmax(outputs, dim=1)

        acc = unsupervised_clustering_accuracy(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)

        # Compute the average loss and accuracy for the epoch and print
        print(f"Epoch {epoch+1} loss: {running_loss/len(data_loader):.4f},              running_acc: {running_acc/len(data_loader):.4f}, acc: {acc:.4f},              running_nmi: {running_nmi/len(data_loader):.4f}, nmi: {nmi:.4f}")
        # Store data to file
        writer.writerow([epoch+1, running_loss/len(data_loader), running_acc/len(data_loader), acc, running_nmi/len(data_loader), nmi])

def unsupervised_clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, C: int=121) -> float:
    """
    Computes the unsupervised clustering accuracy between two clusterings.
    Uses the Hungarian algorithm to find the best matching between true and predicted labels.

    Args:
        y_true: true cluster labels as a 1D torch.Tensor
        y_pred: predicted cluster labels as a 1D torch.Tensor
        C:      number of classes

    Returns:
        accuracy: unsupervised clustering accuracy as a float
    """
    # Create confusion matrix
    cm = confusion_matrix(y_pred, y_true, labels=list(range(C)))

    # Compute best matching between true and predicted labels using the Hungarian algorithm
    _, col_ind = linear_sum_assignment(-cm)

    # Reassign labels for the predicted clusters
    y_pred_reassigned = torch.tensor(col_ind)[y_pred.long()]

    # Compute accuracy as the percentage of correctly classified samples
    acc = accuracy_score(y_true, y_pred_reassigned)

    return acc


def test_classifier(model, data_loader: DataLoader) -> float:
    """
    Testing a classifier given the model and a test set.

    Args:
        model: Neural network model to train.
        data_loader: PyTorch data loader containing the test data.
    
    Returns:
        None
    """
    
    # Disable gradient computation, not needed for inference
    model.eval()
    # Initialize tensors for storing true and predicted labels
    y_true = torch.zeros(len(data_loader.dataset))
    y_pred = torch.zeros(len(data_loader.dataset))

    with torch.no_grad():
        # Iterate over the mini-batches in the data loader
        for i, data in enumerate(data_loader):
            # Get the inputs and true labels for the mini-batch and reshape
            inputs, labels_true = data
            
            # Use GPU if available
            inputs      = inputs.to(device)
                                    
            # # TODO flattening should be done in the feed forward model, else statement should be removed
            # inputs = inputs.view(inputs.size(0), -1)
            
            # Forward pass through the model to get predicted labels
            labels_pred = F.softmax(model(inputs), dim=1)

            # Store predicted and true labels in tensors
            y_pred[i*len(labels_true):(i+1)*len(labels_true)] = torch.argmax(labels_pred.cpu(), dim=1)
            y_true[i*len(labels_true):(i+1)*len(labels_true)] = labels_true

    # Compute unsupervised clustering accuracy score
    acc = unsupervised_clustering_accuracy(y_true, y_pred)

    print(f"\nThe unsupervised clustering accuracy score of the classifier is: {acc}")
    
    return acc


# In[6]:


"""

"""

from archt import get_model

# Information Maximizing Self-Augmented Training
from IMSAT import regularized_information_maximization

# Invariant Information Clustering
from IIC import invariant_information_clustering

from datasets.dataset_classes import NDSBDataset, MNISTDataset


# In[7]:


"""

"""

# Create the train and test datasets
train_dataset = NDSBDataset(train=True, augment_data=True)
test_dataset  = NDSBDataset(train=False)

# Create the train and test data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)


# In[8]:


# Initialize model
model = get_model("vgg11").to(device)

# Initialize loss function, and optimizer
criterion = invariant_information_clustering
optimizer = optim.Adam(model.parameters(), lr=lr)

# Store metadata to .log file
logger = logging.getLogger(__name__)
# Set the logging level
logger.setLevel(logging.INFO)
# Add handler to the logger
logger.addHandler(logging.FileHandler(f'logs/{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log'))

# Write metadata to .log file
logger.info(f'Optimization criterion: {criterion.__name__}')
logger.info(f'Learning rate: {lr}')
logger.info(f'Number of epochs: {num_epochs}')
logger.info(f'Batch size: {batch_size}')
logger.info(f'Optimizer: {optimizer}')
logger.info(f'Model: {model}')

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs)

# Test model
acc = test_classifier(model, test_loader)

logger.info(f'Accuracy: {acc}')
# Close data file
f.close()
