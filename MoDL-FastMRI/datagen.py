import numpy as np
import torch
from torch.utils.data import Dataset
import sigpy as sp
from ops import A_adjoint

def mask_gen(R, acs_lines, dim1, dim2):
    num_sampled_lines = np.floor(dim2 / R)
    center_line_idx = np.arange((dim2 - acs_lines) // 2,(dim2 + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(dim2), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((dim2, dim2))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    mask = sp.resize(mask, [dim1, dim2])
    mask[0:int((dim1-dim2)/2)] = mask[int((dim1-dim2)/2):int(dim1-dim2)]
    mask[dim1-int((dim1-dim2)/2):dim1] = mask[int((dim1-dim2)/2):int(dim1-dim2)]
    return torch.tensor(mask[None,None])
    
class FastMRI(Dataset):
    def __init__(self, downsample, ksp_path, data_path, data_type):
        self.downsample = downsample
        self.ksp_dir = ksp_path
        self._data = torch.load(data_path)
        self.noise_var = self._data['noise_var_noisy']
        self.images = self._data['x_est']
        
        for i in range(len(self.noise_var)):
            self.noise_var[i] = self.noise_var[i] / 2
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gt_cont = self.images[idx]
        ksp_cont = torch.load(self.ksp_dir + '%d.pt'%idx)
        fs_ksp = ksp_cont['ksp_white_noisy'][None,...] # shape: [1,C,H,W]
        maps = ksp_cont['s_maps_white_noisy'][None,...] # shape: [1,C,H,W]
        mask = mask_gen(R=self.downsample, acs_lines=20, dim1=gt_cont.shape[-2], dim2=gt_cont.shape[-1])
        gt_img = gt_cont[None,None] # shape: [1,1,H,W] 
        
        meas_ksp = fs_ksp*mask
        adj_img = A_adjoint(ksp=meas_ksp, maps=maps, mask=mask) 

        return {'idx': idx,
                'fs_ksp': fs_ksp[0],
                'adj_img': adj_img[0],
                'meas_ksp': meas_ksp[0],
                'maps': maps[0],
                'full_mask': mask[0],
                'noise_var': self.noise_var[idx],
                'gt_img': gt_img[0]}