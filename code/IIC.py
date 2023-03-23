"""
Libraries

"""

from sys import float_info

import torch
import torch.nn.functional as F

def invariant_information_clustering(model, inputs: torch.Tensor, y: torch.Tensor, C: int = 10, EPS: float=float_info.epsilon) -> float:
    """
    Calculate the invariant information clustering (IIC) loss.

    Args:
        x: Representation of the input data.
        y: Representation of the transformed input data.
        C: Number of clusters. Default is 10.
        EPS: Epsilon value to prevent numerical issues. Default is torch.finfo().epsilon.

    Returns:
        float: Invariant Information Clustering (IIC) loss.
    """

    # Get the inputs and the augmented inputs 
    _, xt = inputs
    
    # Compute representation of perturbed input.
    yt = model(xt)

    # Compute the joint probability matrix, symmetrize and normalize matrix
    P = (y.unsqueeze(2) * yt.unsqueeze(1)).sum(dim=0)
    P = (P + P.t()) / 2
    P =  P / P.sum()
    P.clamp_(min=EPS)
    
    # Compute the marginals
    Pi = P.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P.sum(dim=0).view(1, C).expand(C, C)

    # Compute the loss
    loss = P * (torch.log(P) - torch.log(Pi) - torch.log(Pj))
    return -loss.sum()