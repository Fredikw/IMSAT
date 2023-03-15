"""
Libraries

"""

from sys import float_info

import torch
import torch.nn.functional as F

"""
Setting hyperparameters for the IMSAT algorithm 

"""

"""
...

"""

def invariant_information_clustering(model, x: torch.Tensor, y: torch.Tensor, C: int = 10, EPS: float=float_info.epsilon) -> float:
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
    # TODO rewirte
    # Generate random unit tensor for perturbation direction
    d = torch.randn_like(x, requires_grad=True)
    d = F.normalize(d, p=2, dim=1)
    
    # TODO rewrite
    # Apply perturbation
    yt = model(x + d)

    # Compute the joint probability matrix, symmetrize and normalize matrix
    P = (y.unsqueeze(2) * yt.unsqueeze(1)).sum(dim=0)
    P = ((P + P.t()) / 2) / P.sum()
    P.clamp_(min=EPS)
    
    # Compute the marginals
    Pi = P.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P.sum(dim=0).view(1, C).expand(C, C)

    # Compute the loss
    loss = P * (torch.log(P) - torch.log(Pi) - torch.log(Pj))
    return -loss.sum()