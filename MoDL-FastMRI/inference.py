import torch
from models import MoDL
from dotmap import DotMap
import os
from tqdm import tqdm
import argparse
from ops import A_adjoint

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--sample_start', type=int, default=0)
parser.add_argument('--sample_end', type=int, default=100)
parser.add_argument('--inference_R', type=int, default=4)
parser.add_argument('--measurements_path', type=str, default='') 
parser.add_argument('--ksp_path', type=str, default='') 
parser.add_argument('--outdir', type=str, default='none')
parser.add_argument('--network', type=str, default='none')
parser.add_argument('--inference_snr', type=str, default="")
parser.add_argument('--anatomy', type=str, default="brain")
parser.add_argument('--method', type=str, default='modl') # ['modl', 'ensure']
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

# MoDL
cont = torch.load(args.network, weights_only=False)
model = MoDL(recon_hparams).cuda()
model.load_state_dict(cont['recon_model_state_dict'])

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

for sample in tqdm(range(args.sample_start, args.sample_end)):
    val_cont = torch.load(args.measurements_path + '/sample_%d.pt'%sample, weights_only=False)
    val_ksp = torch.load(args.ksp_path + "/sample_%d.pt"%sample, weights_only=False)
    gt_img = val_cont['gt'][None,None].cuda()
    fs_ksp = val_ksp['ksp'][None].cuda()
    a_mask = val_cont['mask_' + str(args.inference_R)][None].cuda()
    fm_ksp = fs_ksp*a_mask
    maps = val_ksp['s_map'][None].cuda()
    
    if args.method == "modl":
        x_MoDL = model(ksp=fm_ksp, maps=maps, mask=a_mask[0], method=args.method, meta_unrolls=6)
    
    elif args.method == "ensure":
        v = A_adjoint(ksp=fm_ksp, maps=maps, mask=a_mask)
        v = torch.view_as_real(v[0]).float()
        x_MoDL = model(ksp=fm_ksp, maps=maps, mask=a_mask[0], method=args.method, adjoint=v)
        x_MoDL = torch.view_as_complex(x_MoDL)

    torch.save({"gt_img": gt_img.cpu().detach().numpy(), "recon": x_MoDL[None].cpu().detach().numpy()}, args.outdir + "/sample_" + str(sample) + ".pt")