def forward_fs(img, m):
    coil_imgs = img*m
    return bart(1, 'fft -u 3', coil_imgs.transpose(1, 2, 0)).transpose(2, 0, 1)

def normalization_const(s, gt):                   
    # Get normalization constant from undersampled RSS
    gt_maps_cropped = sp.resize(s, [s.shape[0], 384, 320])
    gt_ksp_cropped = forward_fs(gt[None,...], gt_maps_cropped)
    # zero out everything but ACS
    gt_ksp_acs_only = sp.resize(sp.resize(gt_ksp_cropped, (s.shape[0], ACS_size, ACS_size)), gt_ksp_cropped.shape)
    # make RCS img
    ACS_img = sp.rss(sp.ifft(gt_ksp_acs_only, axes =(-2,-1)), axes=(0,))
    norm_const_99 = np.percentile(np.abs(ACS_img), 99)
    
    return norm_const_99

import sys
import os
os.environ['TOOLBOX_PATH'] = '/home/asad/bart'
sys.path.append('bart/python')
import numpy as np
import h5py
import sigpy as sp
import glob
import random
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

from bart import bart
import torch

center_slice     = 2
ACS_size         = 20
indexes = [i for i in range(500)]

snr = "32dB"

if snr == "32dB":
    noise_amp = np.sqrt(0)
elif snr == "22dB":
    noise_amp = np.sqrt(10)
elif snr == "12dB":
    noise_amp = np.sqrt(100)

ksp_files_train = sorted(glob.glob("/csiNAS/mridata/fastmri_brain/multicoil_val/**.h5"))
ksp_files = []

for files in ksp_files_train:
    if 'AXT2' in files:
        ksp_files.append(files)

ksp_files = sorted(ksp_files)[0:500]

for i in tqdm(range(500)):
    idx = indexes[i]
    slice_idx  = center_slice

    # Load MRI samples and maps
    with h5py.File(ksp_files[idx], 'r') as contents:
        # Get k-space for specific slice
        ksp = np.asarray(contents['kspace'][slice_idx]).transpose(1, 2, 0)

    cimg = bart(1, 'fft -iu 3', ksp) # compare to `bart fft -iu 3 ksp cimg`
    noise = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])[0:30,0:30]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))
    cimg = sp.resize(cimg, [384, 320, cimg.shape[2]])
    
    cimg_white = bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:]).squeeze()
    cimg_white = cimg_white + (noise_amp / np.sqrt(2))*(np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape))
    ksp_white = bart(1, 'fft -u 3', cimg_white)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()
    
    gt_img_white_cropped = sp.resize(bart(1, 'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:]), [384, 320])

    ksp_white = ksp_white.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)  
    cimg_white = cimg_white.transpose(2, 0, 1)  

    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped)
    ksp_white = ksp_white / norm_const_99_white
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = sp.resize(bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:,:,None,:], s_maps_white.transpose(1, 2, 0)[:,:,None,:]), [384, 320])
    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    var = np.var(cimg_white[:, 0:30, 0:30])
    
    total_lines = 320
    R = 2
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_2 = mask[None]
    
    total_lines = 320
    R = 3
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_3 = mask[None]
    
    total_lines = 320
    R = 4
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_4 = mask[None]
    
    total_lines = 320
    R = 5
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_5 = mask[None]
    
    total_lines = 320
    R = 6
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_6 = mask[None]
    
    total_lines = 320
    R = 7
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_7 = mask[None]
    
    total_lines = 320
    R = 8
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_8 = mask[None]
    
    total_lines = 320
    R = 9
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_9 = mask[None]

    total_lines = 320
    R = 10
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    mask_10 = mask[None]

    print('\n')
    print('white SNR: ' + str(10*np.log10(1/var)))
    print('gt norm: ' + str(np.linalg.norm(gt_img_white_cropped)))
    print("Mask R=2: " + str((384*320)/np.sum(mask_2)))
    print("Mask R=3: " + str((384*320)/np.sum(mask_3)))
    print("Mask R=4: " + str((384*320)/np.sum(mask_4)))
    print("Mask R=5: " + str((384*320)/np.sum(mask_5)))
    print("Mask R=6: " + str((384*320)/np.sum(mask_6)))
    print("Mask R=7: " + str((384*320)/np.sum(mask_7)))
    print("Mask R=8: " + str((384*320)/np.sum(mask_8)))
    print("Mask R=9: " + str((384*320)/np.sum(mask_9)))
    print("Mask R=10: " + str((384*320)/np.sum(mask_10)))
    print('\nStep ' + str(i) + ' Done')

    path = '/csiNAS/asad/DATA-FastMRI/brain/val/' + str(snr) + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save({'gt': torch.tensor(gt_img_white_cropped, dtype=torch.complex64),
                'ksp': torch.tensor(ksp_white, dtype=torch.complex64),
                's_map': torch.tensor(s_maps_white, dtype=torch.complex64),
                'mask_2': torch.tensor(mask_2),
                'mask_3': torch.tensor(mask_3),
                'mask_4': torch.tensor(mask_4),
                'mask_5': torch.tensor(mask_5),
                'mask_6': torch.tensor(mask_6),
                'mask_7': torch.tensor(mask_7),
                'mask_8': torch.tensor(mask_8),
                'mask_9': torch.tensor(mask_9),
                'mask_10': torch.tensor(mask_10),
                'norm_consts_99': norm_const_99_white,},
                path + 'sample_' + str(i) + '.pt')
    
    torch.save({"noise_var_noisy": var},
               path + 'noise_var_' + str(i) + '.pt')