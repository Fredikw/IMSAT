"""
Libraries

"""

from typing import Callable

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

"""
Setting generic hyperparameters

"""

num_epochs: int = 20
batch_size: int = 250   # Should be set to a power of 2.
# Learning rate
lr:         float = 0.002

"""
Data Preprocessing

"""

#TODO Preprocess the AILARON dataset to a suitable format.

# #TODO Implement custome dataset for AILARON data. Should inherit from torch.utils.data.Dataset
# class AILARONDataset(torchvision.Dataset):

#     def __init__(self):
#         # Load data
#         pass

#     def __getitem__(self, index):
#         # TODO
#         pass
#     def __len__(self):
#         # TODO
#         pass 

# ailaron_train = AILARONDataset()
# dataloader = DataLoader(dataset=ailaron_train, batch_size=batch_size, shuffle=True)

# Load MNIST dataset, normalizes data and transform to tensor.
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test  = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())


# # Create a subset of the MNIST dataset with the first 100 examples
# mnist_train_subset = torch.utils.data.Subset(mnist_train, range(3000))
# mnist_test_subset  = torch.utils.data.Subset(mnist_test, range(32))

# # Get a random image from the dataset
# image, label = mnist_train[np.random.randint(0, len(mnist_train))]

# # Plot the image
# plt.imshow(image[0], cmap='gray')
# plt.title(f'Label: {label}')
# plt.show()

# Create DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

"""
Multi-output probabilistic classifier that maps similar inputs into similar representations.

"""

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        # Add first fully connected layer with 28 * 28 = 784 input neurons and 1200 output neurons
        self.fc1 = nn.Linear(28 * 28, 1200)
        # Initialize the weights of the first fully connected layer using the He normal initialization
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        # Add first batch normalization layer with 1200 neurons and epsilon = 2e-5
        self.bn1   = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        # Add first ReLU activation function
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(1200, 1200)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.bn2   = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)

        self.relu2 = nn.ReLU()
        
        # Add output layer of size 10 
        self.fc3 = nn.Linear(1200, 10)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')
        
    # Define the forward pass through the network
    def forward(self, x):
        # Pass the input through the first fully connected layer
        x = self.fc1(x)
        # Pass the output of the first fully connected layer through the first batch normalization layer
        x = self.bn1(x)
        # Pass the output of the first batch normalization layer through the first ReLU activation function
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x
 
"""
Training the model

"""

# Define the training function
def train(model: NeuralNet, train_loader: DataLoader, criterion: Callable, optimizer: torch.optim, num_epochs: int) -> None:
    """
    Trains a given model using the provided training data, optimizer and loss criterion for a given number of epochs.

    Args:
        model: Neural network model to train.
        train_loader: PyTorch data loader containing the training data.
        criterion: Loss criterion used for training the model.
        optimizer: Optimizer used to update the model's parameters.
        num_epochs: Number of epochs to train the model.

    Returns:
        None
    """
    # Loop over the epochs
    for epoch in range(num_epochs):

        # Initialize running loss for the epoch
        running_loss = 0.0
        running_acc  = 0.0

        # Loop over the mini-batches in the data loader
        for _, data in enumerate(train_loader):
        
            # Get the inputs and labels for the mini-batch and reshape
            inputs, labels = data
            inputs         = inputs.view(-1, 28*28)
        
            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward pass through the model
            outputs = F.softmax(model(inputs), dim=1)
        
            # Compute the loss
            loss = criterion(model, inputs, outputs)
            # Backward pass through the model and compute gradients
            loss.backward()
        
            # Update the weights
            optimizer.step()

            # Accumulate the loss for the mini-batch
            running_loss += loss.item()
            # Accumulate the accuracy for the mini-batch
            running_acc  += unsupervised_clustering_accuracy(labels, torch.argmax(outputs, dim=1))

        # Compute the average loss for the epoch and print
        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader):.4f}, ACC: {running_acc/len(train_loader):.4f}")

"""
Evaluation Metric

"""

def unsupervised_clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the unsupervised clustering accuracy between two clusterings.
    Uses the Hungarian algorithm to find the best matching between true and predicted labels.

    Args:
        y_true: true cluster labels as a 1D torch.Tensor
        y_pred: predicted cluster labels as a 1D torch.Tensor

    Returns:
        accuracy: unsupervised clustering accuracy as a float
    """
    # Create confusion matrix
    cm = confusion_matrix(y_pred, y_true)

    # Compute best matching between true and predicted labels using the Hungarian algorithm
    _, col_ind = linear_sum_assignment(-cm)

    # Reassign labels for the predicted clusters
    y_pred_reassigned = torch.tensor(col_ind)[y_pred.long()]

    # Compute accuracy as the percentage of correctly classified samples
    acc = accuracy_score(y_true, y_pred_reassigned)

    return acc

"""
Testing

"""

def test_classifier(model: NeuralNet, test_loader: DataLoader) -> None:
    """
    Testing a classifier given the model and a test set.

    Args:
        model: Neural network model to train.
        test_loader: PyTorch data loader containing the test data.
    
    Returns:
        None
    """
    
    # Disable gradient computation, as we don't need it for inference
    model.eval()
    # Initialize tensors for true and predicted labels
    y_true = torch.zeros(len(test_loader.dataset))
    y_pred = torch.empty(len(test_loader.dataset))

    with torch.no_grad():
        # Iterate over the mini-batches in the data loader
        for i, data in enumerate(test_loader):
            # Get the inputs and true labels for the mini-batch and reshape
            inputs, labels_true = data
            inputs = inputs.view(-1, 28*28)

            # Forward pass through the model to get predicted labels
            labels_pred = F.softmax(model(inputs), dim=1)

            # Store predicted and true labels in tensors
            y_pred[i*len(labels_true):(i+1)*len(labels_true)] = torch.argmax(labels_pred, dim=1)
            y_true[i*len(labels_true):(i+1)*len(labels_true)] = labels_true

    # Compute unsupervised clustering accuracy score
    acc = unsupervised_clustering_accuracy(y_true, y_pred)

    print(f"\nThe unsupervised clustering accuracy score of the classifier is: {acc}")

"""
Invariant Information Clustering for Unsupervised Image Classification

"""

def invariant_information_clustering(z: torch.Tensor, zt: torch.Tensor, C: int = 10, eps: float=1e-8) -> float:
    """
    Calculate the invariant information clustering (IIC) loss.

    Args:
        z (torch.Tensor): Representation of the input data.
        zt (torch.Tensor): Representation of the transformed input data.
        C (int, optional): Number of clusters. Default is 10.

    Returns:
        float: Invariant Information Clustering (IIC) loss.

    """

    # Compute the joint probability matrix
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    # Symmetrize matrix
    P = ((P + P.t()) / 2) / P.sum()
    # Prevent numerical issues from p values close to zero. 
    P[(P < eps).data] = eps
    
    # Compute the marginals
    Pi = P.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P.sum(dim=0).view(1, C).expand(C, C)

    return (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()

"""

"""

# # Initialize the model, loss function, and optimizer
# model     = NeuralNet()
# criterion = regularized_information_maximization
# optimizer = optim.Adam(model.parameters(), lr=lr)

# # Train the model
# train(model, train_loader, criterion, optimizer, num_epochs)
# # Test model
# test_classifier(model, test_loader)