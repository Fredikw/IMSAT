"""
Libraries

"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment

from sklearn.metrics import confusion_matrix, accuracy_score
from torch.autograd import Variable
import torchvision


"""
Setting the hyperparameters

"""

num_epochs: int = 20
batch_size: int = 250    # Should be set to a power of 2.
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


# Create a subset of the MNIST dataset with the first 100 examples
mnist_train_subset = torch.utils.data.Subset(mnist_train, range(3000))
mnist_test_subset  = torch.utils.data.Subset(mnist_test, range(32))

# # Get a random image from the dataset
# image, label = mnist_train[np.random.randint(0, len(mnist_train))]

# # Plot the image
# plt.imshow(image[0], cmap='gray')
# plt.title(f'Label: {label}')
# plt.show()

# Create DataLoader
train_loader = DataLoader(mnist_train_subset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

"""
Settings

"""

# Trade-off parameter for mutual information and smooth regularization
lam:        float = 0.1

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
Approximating the marginal distribution

"""

def mariginal_distribution(conditionals: torch.Tensor) -> torch.Tensor:
    """
    Approximates the mariginal probability according to Eq (15).
    
    Args:
        conditionals: conditional probabilities

    Returns
        An approximation of mariginal probabilities.
    """

    # Calculate the sums for each columns.
    return torch.sum(conditionals, dim=0) / conditionals.size()[0]

"""
Mutual Information

"""

def shannon_entropy(probabilities: torch.Tensor) -> float:
    """
    Calculates the Shannon entropy of a set of probabilities.
    
    Args:
        probabilities:
    
    Returns:
        The Shannon entropy
    """

    return -torch.sum(probabilities * torch.log(probabilities))


def mutual_information(mariginals: torch.Tensor, conditionals: torch.Tensor) -> float:
    """
    Calculate the mutual information between two discrete random variables. According to Eq. (7).
    
    Args:
        mariginals: Mariginal probabilities of X.
        conditionals: Conditional probabilities, X|Y.
    
    Returns:
        The mutual information between the two random variables.
    """

    # TODO Remove
    mu: float = 4
    
    marginal_entropy    = shannon_entropy(mariginals)
    conditional_entropy = shannon_entropy(conditionals)

    return mu * marginal_entropy - conditional_entropy

"""
Self-Augmented Training (SAT)

"""

def self_augmented_training(model: NeuralNet, X: torch.Tensor, Y: torch.Tensor, eps: float = 1.0, ksi: float = 1e1, num_iters: int = 1) -> float:
    """
    Self Augmented Training by Virtual Adversarial Training.
    
    Args:
        model: multi-output probabilistic classifier. 
        X: Input samples.
        Y: output when applying model on x.
        eps: Perturbation size.
        ksi: A small constant used for computing the finite difference approximation of the KL divergence.
        num_iters: The number of iterations to use for computing the perturbation.

    Returns:
        The total loss (sum of cross-entropy loss on original input and perturbed input) for the batch.

    """

    """
    Virtual Adversarial Training
    """

    r = torch.randn_like(X)
    r = F.normalize(r, p=2, dim=1)
    # r = r.renorm(2, dim=1, maxnorm=1.0)
    # r.requires_grad_(True)

    # # TODO remove
    # print("r.grad", r.grad)
    # print("r.grad_fn", r.grad_fn)
    # print("r.is_leaf", r.is_leaf)
    # print("r.requires_grad", r.requires_grad)


    for i in range(num_iters):

        r_var = Variable(r)
        r_var.requires_grad_(True)

        # Compute output of the perturbed datapoints.
        Y_p = F.softmax(model(X + r_var * ksi), dim=1)
   
        # Compute the KL divergence between the probabilities
        kl_div = F.kl_div(Y, Y_p, reduction='batchmean')

        # Compute the gradient of current tensor w.r.t. graph leaves.
        grad_r = torch.autograd.grad(kl_div, r_var, create_graph=True)[0]

        # Set the perturbation as the gradient of the KL divergence w.r.t. r
        r = grad_r.detach()
        r = F.normalize(r, p=2, dim=1)


    vad = r * eps

    """
    Self Augmented Training
    """

    Y_p = F.softmax(model(X + vad), dim=1)

    loss = F.kl_div(Y.log(), Y_p, reduction='batchmean')
    
    return loss

def regularized_information_maximization(model: NeuralNet, X: torch.Tensor, Y: torch.Tensor) -> float():
    """
    Computes the loss using regularized information maximization.

    Args:
        model: multi-output probabilistic classifier. 
        X: Input samples.
        Y: output when applying model on x.

    Returns:
        The loss given by mutual information and the regularization penalty.
    """

    conditionals = Y
    mariginals   = mariginal_distribution(Y)

    I = mutual_information(mariginals, conditionals)

    R_sat = self_augmented_training(model, X, Y)

    return R_sat - lam * I


"""
Training the model

"""

# Define the training function
def train(model: NeuralNet, train_loader: DataLoader, criterion: Callable, optimizer: torch.optim, num_epochs: int):
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
        
        # Loop over the mini-batches in the data loader
        for _, data in enumerate(train_loader):
        
            # Get the inputs and labels for the mini-batch and reshape
            inputs, _ = data
            inputs    = inputs.view(-1, 28*28)
        
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

        # Compute the average loss for the epoch and print
        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")


"""
Evaluation Metric

"""
# TODO consider including Normalized Information Score as an evaluation metric.

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

"""

# Initialize the model, loss function, and optimizer
model     = NeuralNet()
criterion = regularized_information_maximization
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs)
# Test model
test_classifier(model, test_loader)