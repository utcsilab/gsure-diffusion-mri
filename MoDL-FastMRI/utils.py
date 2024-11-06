#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.fft as torch_fft
import numpy as np

def ifft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    
    return x
    
def fft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    
    return x

def itemize(x):
    """Converts a Tensor into a list of Python numbers.
    """
    if len(x.shape) < 1:
        x = x[None]
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()
    
# Complex dot product of two complex-valued multidimensional Tensors
def zdot_batch(x1, x2):
    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)

# Same, applied to self --> squared L2-norm
def zdot_single_batch(x):
    return zdot_batch(x, x)

# The following two functions are from utils_SSDU.py for creating gaussian masks
def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """
    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]