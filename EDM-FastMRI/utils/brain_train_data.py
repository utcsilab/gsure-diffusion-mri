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

def task(i):
    idx = indexes[i]
    sample_idx = idx // num_slices
    slice_idx  = center_slice + np.mod(idx, num_slices) - num_slices // 2

    # Load MRI samples and maps
    with h5py.File(ksp_files[sample_idx], 'r') as contents:
        # Get k-space for specific slice
        ksp = np.asarray(contents['kspace'][slice_idx]).transpose(1, 2, 0)
        cimg = bart(1, 'fft -iu 3', ksp) # compare to `bart fft -iu 3 ksp cimg`
        cimg = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])

    noise = cimg[0:30,0:30]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))
    cimg_white = sp.resize(bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:]).squeeze(), [384, 320, cimg_white.shape[2]])
    cimg_white_noisy = cimg_white + (noise_amp / np.sqrt(2))*(np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape))
    
    ksp_white = bart(1, 'fft -u 3', cimg_white)
    ksp_white_noisy = bart(1, 'fft -u 3', cimg_white_noisy)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy[:,:,None,:]).squeeze()
    
    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:])
    gt_img_white_cropped_noisy = bart(1, 'pics -S -i 30', ksp_white_noisy[:,:,None,:], s_maps_white_noisy[:,:,None,:])

    ksp_white = ksp_white.transpose(2, 0, 1)
    ksp_white_noisy = ksp_white_noisy.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)  
    s_maps_white_noisy = s_maps_white_noisy.transpose(2, 0, 1)  
    cimg_white = cimg_white.transpose(2, 0, 1)  
    cimg_white_noisy = cimg_white_noisy.transpose(2, 0, 1)  

    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped)
    norm_const_99_white_noisy = normalization_const(s_maps_white_noisy, gt_img_white_cropped_noisy)
    ksp_white = ksp_white / norm_const_99_white
    ksp_white_noisy = ksp_white_noisy / norm_const_99_white_noisy
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:,:,None,:], s_maps_white.transpose(1, 2, 0)[:,:,None,:])
    gt_img_white_cropped_noisy=bart(1, 'pics -S -i 30',ksp_white_noisy.transpose(1,2,0)[:,:,None,:],s_maps_white_noisy.transpose(1,2,0)[:,:,None,:])

    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    cimg_white_noisy = bart(1, 'fft -iu 3', ksp_white_noisy.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    
    var = np.var(cimg_white[:, 0:30, 0:30])
    var_noisy = np.var(cimg_white_noisy[:, 0:30, 0:30])
    
    coil_imgs_with_maps_white_noisy = cimg_white_noisy * np.conj(s_maps_white_noisy)
    u_white_noisy = np.sum(coil_imgs_with_maps_white_noisy, axis = -3)
    u_cropped_white_noisy = u_white_noisy

    print('\n')
    print('white SNR: ' + str(10*np.log10(1/var)))
    print('white_noisy SNR: ' + str(10*np.log10(1/var_noisy)))
    print('gt norm: ' + str(np.linalg.norm(gt_img_white_cropped_noisy)))
    print('u norm: ' + str(np.linalg.norm(u_cropped_white_noisy)))
    
    return i, gt_img_white_cropped, gt_img_white_cropped_noisy, u_cropped_white_noisy, norm_const_99_white_noisy, var_noisy, ksp_white_noisy, s_maps_white_noisy

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
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
from multiprocessing import Pool

device           = sp.cpu_device
n_proc           = 30 # number of cpu cores to use, when possible
num_slices       = 5
center_slice     = 2
ACS_size         = 24

db = "32dB"

if db == "32dB":
    noise_amp = np.sqrt(0)
elif db == "22dB":
    noise_amp = np.sqrt(10)
elif db == "12dB":
    noise_amp = np.sqrt(100)

with open('/home/asad/Old/mri-score/data/ksp_files.npy', 'rb') as f:
    ksp_files = np.load(f)

indexes = [i for i in range(10000)]

x_est_gt = torch.zeros(10000, 384, 320, dtype=torch.complex64)
x_est = torch.zeros(10000, 384, 320, dtype=torch.complex64)
u_images = torch.zeros(10000, 384, 320, dtype=torch.complex64)
norm_consts_99 = torch.zeros(10000, dtype=torch.float32)
noise_var_noisy = torch.zeros(10000, dtype=torch.float32)

path = "/csiNAS/asad/DATA-FastMRI/brain/train/" + db
if not os.path.exists(path + "/ksp/"):
    os.makedirs(path + "/ksp/")

with Pool(n_proc) as p:
    for i, gt_img_white_cropped, gt_img_white_cropped_noisy, u_cropped_white_noisy, norm_const_99, var_noisy, ksp_white_noisy, s_maps_white_noisy in tqdm(p.imap(task, range(10000))):
        x_est_gt[i] = torch.tensor(gt_img_white_cropped, dtype=torch.complex64)
        x_est[i] = torch.tensor(gt_img_white_cropped_noisy, dtype=torch.complex64)
        u_images[i] = torch.tensor(u_cropped_white_noisy, dtype=torch.complex64)
        norm_consts_99[i] = torch.tensor(norm_const_99, dtype=torch.float32)
        noise_var_noisy[i] = torch.tensor(var_noisy, dtype=torch.float32)
        ksp_white_noisy = torch.tensor(ksp_white_noisy, dtype=torch.complex64)
        s_maps_white_noisy = torch.tensor(s_maps_white_noisy, dtype=torch.complex64)
        
        torch.save({
            "ksp_white_noisy": ksp_white_noisy,
            "s_maps_white_noisy": s_maps_white_noisy},
            path + "/ksp/" + str(i) + ".pt")
        print('Step ' + str(i) + ' Done')

torch.save({'x_est_gt': x_est_gt,
            'x_est': x_est,
            'u_images': u_images,
            'norm_consts_99': norm_consts_99,
            'noise_var_noisy': noise_var_noisy},
            path + "/noisy.pt")