import torch
import numpy as np

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features (torch.Tensor).

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return torch.mm(x, x.T, )

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix (torch.Tensor).
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
                  estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not torch.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")

    n = gram.size(0)
    dtype = torch.float64 #gram.dtype
    device = gram.device
    unit = torch.ones((n, n), device=device, dtype=dtype)
    #identity = torch.eye(n, device=device, dtype=dtype)

    if unbiased:
        gram = gram - unit.mm(gram) / (n - 2)
        gram = gram - gram.mm(unit) / (n - 2)
        gram = gram + unit.mm(gram).mm(unit) / ((n - 1) * (n - 2))
    else:
        gram = gram - unit.mm(gram) / n
        gram = gram - gram.mm(unit) / n
        gram = gram + unit.mm(gram).mm(unit) / (n * n)

    return gram

def cka(gram_x, gram_y, unbiased=False):
    """Compute Centered Kernel Alignment (CKA).

    Args:
        gram_x: Gram matrix for dataset X (torch.Tensor).
        gram_y: Gram matrix for dataset Y (torch.Tensor).
        unbiased: Whether to adjust the Gram matrices for an unbiased HSIC estimate.

    Returns:
        The CKA similarity value (scalar).
    """
    gram_x = center_gram(gram_x, unbiased=unbiased)
    gram_y = center_gram(gram_y, unbiased=unbiased)
    
    # Hilbert-Schmidt Independence Criterion (HSIC)
    hsic = torch.sum(gram_x * gram_y)

    # Normalization terms
    normalization_x = torch.sqrt(torch.sum(gram_x * gram_x))
    normalization_y = torch.sqrt(torch.sum(gram_y * gram_y))

    return hsic / (normalization_x * normalization_y)

def compute_cka(x, y, unbiased=False):
    """Compute CKA similarity between two datasets.

    Args:
        x: A num_examples x num_features matrix of features (torch.Tensor).
        y: A num_examples x num_features matrix of features (torch.Tensor).
        unbiased: Whether to adjust the Gram matrices for an unbiased HSIC estimate.

    Returns:
        The CKA similarity value (scalar).
    """
    gram_x = gram_linear(x)
    gram_y = gram_linear(y)
    return cka(gram_x, gram_y, unbiased=unbiased)
