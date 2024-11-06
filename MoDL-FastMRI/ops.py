#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:29:31 2021

@author: yanni
"""

import torch
import torch.fft as torch_fft
import sigpy as sp
import numpy as np

def nrmse(x, y):
    num = torch.norm(x-y, p=2)
    denom = torch.norm(x,p=2)
    return num/denom

# Centered, orthogonal ifft in torch >= 1.7
def ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

# Centered, orthogonal fft in torch >= 1.7
def fft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * \
                  np.conj(s_maps), axis=1)

def A_forward(image, maps, mask):
    # image shape: [B,1,H,W]
    # maps shape: [B,C,H,W]
    # mask shape: [B,1,H,W]

    coil_imgs = maps*image
    coil_ksp = fft(coil_imgs)
    sampled_ksp = mask*coil_ksp
    return sampled_ksp

def A_adjoint(ksp, maps, mask):
    # ksp shape: [B,1,H,W]
    # maps shape: [B,C,H,W]
    # mask shape: [B,1,H,W]

    sampled_ksp = mask*ksp
    coil_imgs = ifft(sampled_ksp)
    img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1)
    return img_out[:,None,...]

def A_normal(image, maps, mask):
    meas_ksp = A_forward(image=image,maps=maps,mask=mask)
    out = A_adjoint(ksp=meas_ksp, maps=maps, mask=mask)
    return out