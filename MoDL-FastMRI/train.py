import numpy as np
import torch
from tqdm import tqdm
import os
import argparse
from dotmap import DotMap
from torch.optim import Adam
from losses import nrmse
from models import MoDL
from datagen import FastMRI
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--data_R', type=int, default=4)
parser.add_argument('--seed', type=int, default=8)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--recon_lr' , type=float, default=3e-4)
parser.add_argument('--decay_ep', type=int, default=2)
parser.add_argument('--save_interval',type=int, default=1)
parser.add_argument('--comp_val', type=int, default=1)
parser.add_argument('--snr', type=str)
parser.add_argument('--anatomy', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--ksp_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--data_type', type=str, default='noisy') # ['noisy', 'denoised']

args   = parser.parse_args()

#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# configure reconstruction model settings
recon_hparams = DotMap()
recon_hparams.verbose      = False
recon_hparams.batch_size   = 1
recon_hparams.max_cg_steps = 6
recon_hparams.cg_eps       = 1e-6
recon_hparams.unrolls      = 6
recon_hparams.logging      = False
recon_hparams.img_channels = 16
recon_hparams.img_blocks   = 4
recon_hparams.img_arch     = 'UNet'
recon_hparams.l2lam_train = True
recon_hparams.l2lam_init  = 0.1

# Create Model and Load weights
model = MoDL(recon_hparams).cuda()
model.train()

# create optimzer for recon model and mask
optimizer = Adam(
    [
        {"params": model.parameters(), "lr": args.recon_lr}
    ],
    lr=args.recon_lr,
)

# Count parameters
total_params = np.sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
print('Recon Model Total parameters %d' % total_params)

# create dataloader
train_dataset = FastMRI(downsample=args.data_R, ksp_path=args.ksp_path, data_path=args.data_path, data_type=args.data_type)
train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)

#create results folder
results_dir = args.root + "/models/" + args.anatomy + "/" + args.data_type + args.snr + "/R=" + str(args.data_R)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

running_training   = 0.0
training_log = []

for epoch_idx in range(args.epochs):
    for sample_idx, sample in tqdm(enumerate(train_loader)):
        # Move to CUDA
        for key in sample.keys():
            try:
                sample[key] = sample[key].cuda()
            except:
                pass

        # for first loss term
        x_train = model(ksp=sample['meas_ksp'], maps=sample['maps'], mask=sample['full_mask'][0], meta_unrolls=6)
        full_loss = nrmse(abs(sample['gt_img']), abs(x_train))

        # Backprop
        optimizer.zero_grad()
        full_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            running_training = 0.99 * running_training + 0.01 * full_loss.item() if running_training > 0. else full_loss.item()
            training_log.append(running_training)

        # Verbose
        print('Epoch %d, sample %d, Train loss %.3f,  Avg. Train loss %.3f' % (
            epoch_idx,  sample_idx, full_loss.item(), running_training))

    if (epoch_idx%args.save_interval==0):
        save_dict = {
                'gt_img': sample['gt_img'].detach().cpu(),
                'x_train': x_train.detach().cpu(),
                'recon_model_state_dict': model.state_dict(),
                'recon_hparams':recon_hparams,
                'mask': sample['full_mask'].cpu(),
                'training_log':training_log,
                'hparams': recon_hparams
        }
        torch.save(save_dict, results_dir+'/ckpt_%d.pt'%(epoch_idx))