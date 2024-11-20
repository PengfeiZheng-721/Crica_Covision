# loss.py
import torch
import torch.nn as nn
import math

def compute_loss(features, label_matrix):
    """
    Compute loss using scaled dot product and BCEWithLogitsLoss.

    Args:
        features (torch.Tensor): Feature vectors with shape (N, D).
        label_matrix (torch.Tensor): Ground truth labels with shape (N, N).

    Returns:
        torch.Tensor: Computed loss value.
    """
    N, D = features.shape

    # Compute dot products between all feature vectors
    dot_product_matrix = torch.matmul(features, features.T)  # Shape: (N, N)

    # Scale dot products, similar to the scaled dot product in multi-head attention
    scaling_factor = 1.0 / math.sqrt(D)
    scaled_dot_product = dot_product_matrix * scaling_factor  # Shape: (N, N)

    # Flatten the scaled dot product matrix into a 1D tensor
    logits = scaled_dot_product.view(-1)  # Shape: (N*N,)

    # Flatten the label matrix into a 1D tensor
    labels = label_matrix.view(-1)        # Shape: (N*N,)

    # Define the loss function; BCEWithLogitsLoss internally applies sigmoid
    criterion = nn.BCEWithLogitsLoss()

    # Compute the loss
    loss = criterion(logits, labels)

    return loss
