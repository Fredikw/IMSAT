"""
Libraries

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

    return torch.sum(conditionals) / conditionals.size()[0]

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

    marginal_entropy    = shannon_entropy(mariginals)
    conditional_entropy = shannon_entropy(conditionals)

    return marginal_entropy - conditional_entropy

"""
Self-Augmented Training (SAT)

"""

def self_augmented_training(model: NeuralNet, X: torch.Tensor, Y: torch.Tensor, eps: float = 1.0, ksi: float = 1e-6, num_iters: int = 1) -> float:
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
    # Initial perturbation with the same shape as input samples.
    r = torch.randn_like(X, requires_grad=True)

    for i in range(num_iters):

        # Compute the discrete representation of the perturbed datapoints.
        Y_p = F.softmax(model(X + r * ksi), dim=1)
   
        # Compute the KL divergence between the probabilities
        kl_div = F.kl_div(Y.log(), Y_p, reduction='batchmean')

        # Compute the gradient of current tensor w.r.t. graph leaves.
        grad_r = torch.autograd.grad(kl_div, r, create_graph=True)[0]

        # Set the perturbation as the gradient of the KL divergence w.r.t. r
        r = grad_r.detach()

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