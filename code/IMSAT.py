import torch
import torch.nn.functional as F

"""
Setting hyperparameters for the IMSAT algorithm 

"""

# Trade-off parameter for mutual information and smooth regularization
lam: float = 0.1
        
"""
...

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
    return torch.sum(conditionals, dim=0) / conditionals.shape[0]

def shannon_entropy(probabilities: torch.Tensor) -> float:
    """
    Calculates the Shannon entropy of a set of probabilities.
    
    Args:
        probabilities:
    
    Returns:
        The Shannon entropy (float)
    """

    if probabilities.dim() == 1:
        # Calculates the entropy of mariginals according to the definition of Shannon Entropy.
        return -torch.sum(probabilities * torch.log(probabilities))
    
    else:
        # Calculates the entropy of conditionals, according to (9)
        return -torch.sum(probabilities * torch.log(probabilities)) / probabilities.shape[0]

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

def self_augmented_training(model, inputs: torch.Tensor, outputs: torch.Tensor, eps: float = 1.0, ksi: float = 1e0, num_iters: int = 1) -> float:
    """
    Self Augmented Training by Virtual Adversarial Perturbation.
    
    Args:
        model: multi-output probabilistic classifier. 
        inputs: Input samples.
        outputs: output when applying model on X.
        eps: Magnitude of perturbation.
        ksi: A small constant used for computing the finite difference approximation of the KL divergence.
        num_iters: The number of iterations to use for computing the perturbation.

    Returns:
        The total loss (sum of cross-entropy loss on original input and perturbed input) for the batch.

    """

    """
    Generate Virtual Adversarial Perturbation

    """

    #TODO Consider removing without breaking code
    y = model(inputs)
    
    # Generate random unit tensor for perturbation direction
    d = torch.randn_like(inputs, requires_grad=True)
    d = F.normalize(d, p=2, dim=1)
    
    # Use finite difference method to estimate adversarial perturbation
    for _ in range(num_iters):
        # Forward pass with perturbation
        y_p = model(inputs + ksi * d)
        
        # Calculate the KL divergence between the predictions with and without perturbation
        # To avoid underflow, loss expects the argument input in the log-space.
        kl_div = F.kl_div(F.log_softmax(y_p, dim=1), F.softmax(y, dim=1), reduction='batchmean')
        
        # Calculate the gradient of the KL divergence w.r.t the perturbation
        grad = torch.autograd.grad(kl_div, d)[0]
        
        # Update the perturbation
        d = grad
        d = F.normalize(d, p=2, dim=1)
        d.requires_grad_()

    """
    Apply Perturbation and calculate the Kullback-Leibler divergence Loss

    """
    outputs_p = F.log_softmax(model(inputs + eps * d), dim=1)

    loss = F.kl_div(outputs_p, outputs, reduction='batchmean')
    
    return loss

def regularized_information_maximization(model, inputs: torch.Tensor, outputs: torch.Tensor) -> float():
    """
    Computes the loss using regularized information maximization.

    Args:
        model: multi-output probabilistic classifier. 
        X: Input samples.
        Y: output when applying model on x.

    Returns:
        The loss given by mutual information and the regularization penalty.
    """

    conditionals = outputs
    mariginals   = mariginal_distribution(conditionals)

    I = mutual_information(mariginals, conditionals)

    R_sat = self_augmented_training(model, inputs, outputs)

    return R_sat - lam * I