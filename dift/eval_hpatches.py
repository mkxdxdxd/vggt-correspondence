import torch
import sys
import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import skimage.io as io
import torch.nn.functional as F
# from src.models.dift_sd import SDFeaturizer4Eval
# from src.models.dift_adm import ADMFeaturizer4Eval
from vggt.models.vggt_geo import VGGT
import os
import json
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from croco.models.croco import CroCoNet

transform = transforms.Compose([
    transforms.Resize((700, 700)),
    transforms.ToTensor()
])

transform_croco = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
checkpoint_path = '/workspace/minkyung/dift/croco/checkpoint/CroCo_V2_ViTLarge_BaseDecoder.pth'

class HPatchDataset(Dataset):
    def __init__(self, imdir, spdir):
        self.imfs = []
        for f in os.listdir(imdir):
            scene_dir = os.path.join(imdir, f)
            self.imfs.extend([os.path.join(scene_dir, '{}.ppm').format(ind) for ind in range(1, 7)])
        self.spdir = spdir

    def __getitem__(self, item):
        imf = self.imfs[item]
        im = io.imread(imf)
        name, idx = imf.split('/')[-2:]
        coord = np.loadtxt(os.path.join(self.spdir, f'{name}-{idx[0]}.kp')).astype(np.float32)
        out = {'coord': coord, 'imf': imf}
        return out

    def __len__(self):
        return len(self.imfs)


def main(args):
    for arg in vars(args):
        value = getattr(args,arg)
        if value is not None:
            print('%s: %s' % (str(arg),str(value)))

    dataset = HPatchDataset(imdir=args.hpatches_path, spdir=args.kpts_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if args.dift_model == 'sd':
        # dift = SDFeaturizer4Eval()
        "hi"
    elif args.dift_model == 'adm':
        # dift = ADMFeaturizer4Eval()
        "hello"
    elif args.dift_model == 'vggt':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        dtype = torch.float16
        dift = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    elif args.dift_model =='dino':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        dift = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
    elif args.dift_model == 'croco':
        dtype = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(checkpoint_path, 'cpu')
        dift = CroCoNet( **ckpt.get('croco_kwargs',{})).to(device)
        dift.eval()
        msg = dift.load_state_dict(ckpt['model'], strict=True)

    with torch.no_grad():
        for data in tqdm(data_loader):
            img_path = data['imf'][0]
            img = Image.open(img_path)
            w, h = img.size
            coord = data['coord'].to('cuda')
            c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord.device).float()
            coord_norm = (coord - c) / c

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    if args.dift_model == 'vggt':
                        # Predict attributes including cameras, depth maps, and point maps.
                        img = transform(img).unsqueeze(0).to(device) # Shape: [1, 3, H, W]
                        feat = dift(images=img, layer=args.layer) # dict each [1, 2048, 37, 37]
                    elif args.dift_model == 'dino':
                        img = transform(img).unsqueeze(0).to(device)
                        out = dift.forward_features(img)['x_norm_patchtokens'] # [1, 1369, 768]
                        feat = out.permute(0, 2, 1).reshape(1, 768, 37, 37)
                    elif args.dift_model == 'croco':
                        img = transform_croco(img).unsqueeze(0).to(device)
                        out, _, _ = dift._encode_image(img) # [1, 196, 1024]
                        feat = out.permute(0, 2, 1).reshape(1, -1, 14, 14) # dict each [1, 1024, 14, 14]
                    else: 
                        feat = dift.forward(img,
                                            img_size=args.img_size,
                                            t=args.t,
                                            up_ft_index=args.up_ft_index,
                                            ensemble_size=args.ensemble_size) # [1, 640, 96, 96]

            feat = F.grid_sample(feat, coord_norm.unsqueeze(2)).squeeze(-1)
            feat = feat.transpose(1, 2)

            desc = feat.squeeze(0).detach().cpu().numpy()
            kpt = coord.cpu().numpy().squeeze(0)

            out_dir = os.path.join(args.save_path, os.path.basename(os.path.dirname(img_path)))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'{os.path.basename(img_path)}.dift_{args.dift_model}'), 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=kpt,
                    scores=[],
                    descriptors=desc
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--hpatches_path', type=str, default='/scratch/dift_release/d2-net/hpatches_sequences/hpatches-sequences-release', help='path to hpatches dataset')
    parser.add_argument('--kpts_path', type=str, default='./superpoint-1k', help='path to 1k superpoint keypoints')
    parser.add_argument('--save_path', type=str, default='./hpatches_results', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm', 'vggt', 'dino', 'croco'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--layer', default=23, type=int)
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    main(args)