#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import copy as copy

from core_ops import TorchHybridSense, TorchHybridImage
from core_ops import TorchMoDLSense, TorchMoDLImage

from opt import ZConjGrad
from unet import NormUnet
from CG import conjgrad_single
from ops import A_adjoint


class MoDL(torch.nn.Module):
    def __init__(self, hparams):
        super(MoDL, self).__init__()
        # Storage
        self.verbose    = hparams.verbose
        self.batch_size = hparams.batch_size
        self.block2_max_iter = hparams.max_cg_steps
        self.cg_eps          = hparams.cg_eps

        # Logging
        self.logging  = hparams.logging

        # ImageNet parameters
        self.img_channels = hparams.img_channels
        self.img_blocks   = hparams.img_blocks
        self.img_arch     = hparams.img_arch

        # Attention parameters
        self.att_config   = hparams.att_config
        if hparams.img_arch != 'Unet':
            self.latent_channels = hparams.latent_channels
            self.kernel_size     = hparams.kernel_size

        # Get useful values
        self.ones_mask = torch.ones((1)).cuda()

        # Initialize trainable parameters
        if hparams.l2lam_train:
            self.block2_l2lam = torch.nn.Parameter(torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda())
        else:
            self.block2_l2lam = torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda()

        # self.image_net = Unet(in_chans=2, out_chans=2, chans=self.img_channels,num_pool_layers=self.img_blocks)
        self.image_net = NormUnet(chans=self.img_channels, num_pools=self.img_blocks)

    # Get torch operators for the entire batch
    def get_core_torch_ops(self, mps_kernel, img_kernel, mask, direction):
        # List of output ops
        normal_ops, adjoint_ops, forward_ops = [], [], []

        # For each sample in batch
        for idx in range(self.batch_size):
            # Type
            if direction == 'ConvSense':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLSense(mps_kernel[idx], mask[idx])
            elif direction == 'ConvImage':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLImage(img_kernel[idx], mask[idx])

            # Add to lists
            normal_ops.append(normal_op)
            adjoint_ops.append(adjoint_op)
            forward_ops.append(forward_op)

        # Return operators
        return normal_ops, adjoint_ops, forward_ops

    # Given a batch of inputs and ops, get a single batch operator
    def get_batch_op(self, input_ops, batch_size):
        # Inner function trick
        def core_function(x):
            # Store in list
            output_list = []
            for idx in range(batch_size):
                output_list.append(input_ops[idx](x[idx])[None, ...])
            # Stack and return
            return torch.cat(output_list, dim=0)
        return core_function

    def forward(self, ksp, maps, mask, meta_unrolls=1):
        mask      = mask
        ksp       = ksp
        # Initializers
        with torch.no_grad():
            maps = maps


        if self.logging:
            img_logs = []

        normal_ops, adjoint_ops, forward_ops = \
            self.get_core_torch_ops(maps, None,
                    mask, 'ConvSense')

        # Get joint batch operators for adjoint and normal
        normal_batch_op, adjoint_batch_op = \
            self.get_batch_op(normal_ops, self.batch_size), \
            self.get_batch_op(adjoint_ops, self.batch_size)


        # get initial image x = A^H(y)
        est_img_kernel = adjoint_batch_op(ksp) #flipped order of was ksp[:,mask_idx]: same below
        # print(est_img_kernel.shape)
        # For each outer unroll
        # print(est_img_kernel.shape)
        for meta_idx in range(meta_unrolls):
            # Convert to reals
            # est_img_kernel_prev = est_img_kernel.clone()
            est_img_kernel = torch.view_as_real(est_img_kernel).float()# shape: [B,H,W,2]
            # print(est_img_kernel.shape)
            # Apply image denoising network in image space
            # stack images to be 4 channel input
            # print(meta_idx, '   ', est_img_kernel.shape)
            # est_img_kernel = self.image_net(est_img_kernel.permute(0,-1,-3,-2)) + est_img_kernel.permute(0,-1,-3,-2) UNet
            est_img_kernel = self.image_net(est_img_kernel[None,...])[0] #NormUNet
            # Convert to complex
            # est_img_kernel = est_img_kernel.permute(0,-2,-1,1).contiguous() # shape: [B,H,W,4] UNet
            # # Convert to complex
            est_img_kernel = torch.view_as_complex(est_img_kernel)

            rhs = adjoint_batch_op(ksp) + \
                self.block2_l2lam[0] * est_img_kernel

            # Get unrolled CG op
            cg_op = ZConjGrad(rhs, normal_batch_op,
                             l2lam=self.block2_l2lam[0],
                             max_iter=self.block2_max_iter,
                             eps=self.cg_eps, verbose=self.verbose)
            # Run CG
            est_img_kernel = cg_op(est_img_kernel)

            # Log
            if self.logging:
                img_logs.append(est_img_kernel)

        if self.logging:
            return est_img_kernel, img_logs
        else:
            return est_img_kernel



class MoDL_clean(torch.nn.Module):
    def __init__(self, hparams):
        super(MoDL_clean, self).__init__()
        # Storage
        self.verbose    = hparams.verbose
        self.batch_size = hparams.batch_size
        self.block2_max_iter = hparams.max_cg_steps
        self.cg_eps          = hparams.cg_eps

        # Logging
        self.logging  = hparams.logging

        # ImageNet parameters
        self.img_channels = hparams.img_channels
        self.img_blocks   = hparams.img_blocks
        self.img_arch     = hparams.img_arch

        # Attention parameters
        self.att_config   = hparams.att_config
        if hparams.img_arch != 'Unet':
            self.latent_channels = hparams.latent_channels
            self.kernel_size     = hparams.kernel_size

        # Get useful values
        self.ones_mask = torch.ones((1)).cuda()

        # Initialize trainable parameters
        if hparams.l2lam_train:
            self.block2_l2lam = torch.nn.Parameter(torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda())
        else:
            self.block2_l2lam = torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda()

        # self.image_net = Unet(in_chans=2, out_chans=2, chans=self.img_channels,num_pool_layers=self.img_blocks)
        self.image_net = NormUnet(chans=self.img_channels, num_pools=self.img_blocks)


    def forward(self, ksp, maps, mask, meta_unrolls=1):
        mask      = mask
        ksp       = ksp
        maps      = maps
        # print('ksp shape: ', ksp.shape)
        # print('mask shape: ', mask.shape)
        # print('maps shape: ', maps.shape)

        if self.logging:
            img_logs = []

        # get initial image x = A^H(y)
        est_img_kernel = A_adjoint(ksp,maps,mask) 
        
        for meta_idx in range(meta_unrolls):
            # Convert to reals
            # print(est_img_kernel.shape)
            est_img_kernel = torch.view_as_real(est_img_kernel[:,0,...]).float()# shape: [B,H,W,2]
            # Apply image denoising network in image space
            est_img_kernel = self.image_net(est_img_kernel[None,...])[0] #NormUNet
            # Convert to complex
            est_img_kernel = torch.view_as_complex(est_img_kernel)

            rhs = A_adjoint(ksp,maps,mask) + \
                self.block2_l2lam[0] * est_img_kernel

            # Run CG
            est_img_kernel,_ = conjgrad_single(x=est_img_kernel, b=rhs,maps=maps, mask=mask,
                             l2lam=self.block2_l2lam[0],
                             max_iter=self.block2_max_iter,
                             eps=self.cg_eps, verbose=self.verbose)

            # Log
            if self.logging:
                img_logs.append(est_img_kernel)

        if self.logging:
            return est_img_kernel, img_logs
        else:
            return est_img_kernel