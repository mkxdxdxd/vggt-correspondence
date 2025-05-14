import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
# from src.models.dift_sd import SDFeaturizer4Eval
# from src.models.dift_adm import ADMFeaturizer4Eval
from vggt.models.vggt import VGGT
import os
import json
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from croco.models.croco import CroCoNet

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor()
])

transform_croco = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

checkpoint_path = '/workspace/minkyung/dift/croco/checkpoint/CroCo_V2_ViTLarge_BaseDecoder.pth'

def main(args):
    for arg in vars(args):
        value = getattr(args,arg)
        if value is not None:
            print('%s: %s' % (str(arg),str(value)))

    torch.cuda.set_device(0)

    dataset_path = args.dataset_path
    test_path = 'PairAnnotation/test'
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    cat2json = {}

    for cat in all_cats:
        cat_list = []
        for i in json_list:
            if cat in i:
                cat_list.append(i)
        cat2json[cat] = cat_list

    # get test image path for all cats
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        cat_list = cat2json[cat]
        for json_path in cat_list:
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
                temp_f.close()
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            if src_imname not in cat2img[cat]:
                cat2img[cat].append(src_imname)
            if trg_imname not in cat2img[cat]:
                cat2img[cat].append(trg_imname)

    if args.dift_model == 'sd':
        # dift = SDFeaturizer4Eval(cat_list=all_cats)
        "hello"
    elif args.dift_model == 'adm':
        # dift = ADMFeaturizer4Eval()
        "hi"
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

    print("saving all test images' features...")
    os.makedirs(args.save_path, exist_ok=True)
    for cat in tqdm(all_cats):
        output_dict = {}
        image_list = cat2img[cat]
        for image_path in image_list:
            img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path)).convert('RGB')
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    if args.dift_model == 'vggt':
                        # Predict attributes including cameras, depth maps, and point maps.
                        img = transform(img).unsqueeze(0).to(device) # Shape: [1, 3, H, W]
                        output_dict[image_path] = dift(images=img, layer=args.layer) # dict each [1, 2048, 37, 37]
                    elif args.dift_model == 'dino':
                        img = transform(img).unsqueeze(0).to(device)
                        out = dift.forward_features(img)['x_norm_patchtokens'] # [1, 1369, 768]
                        out = out.permute(0, 2, 1).reshape(1, 768, 37, 37)
                        output_dict[image_path] = out
                    elif args.dift_model == 'croco':
                        img = transform_croco(img).unsqueeze(0).to(device)
                        out, _, _ = dift._encode_image(img) # [1, 196, 1024]
                        out = out.permute(0, 2, 1).reshape(1, -1, 14, 14) # dict each [1, 1024, 14, 14]
                        output_dict[image_path] = out
                    else: 
                        output_dict[image_path] = dift.forward(img,
                                                category=cat,
                                                img_size=args.img_size,
                                                t=args.t,
                                                up_ft_index=args.up_ft_index,
                                                ensemble_size=args.ensemble_size)
        torch.save(output_dict, os.path.join(args.save_path, f'{cat}.pth'))

    total_pck = []
    all_correct = 0
    all_total = 0

    for cat in all_cats:
        cat_list = cat2json[cat]
        output_dict = torch.load(os.path.join(args.save_path, f'{cat}.pth')) # dict each [1, 1280, 48, 48]

        cat_pck = []
        cat_correct = 0
        cat_total = 0

        for json_path in tqdm(cat_list):

            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)

            src_img_size = data['src_imsize'][:2][::-1] # [333, 500]
            trg_img_size = data['trg_imsize'][:2][::-1] # [333, 500]

            src_ft = output_dict[data['src_imname']] # [1, 1280, 48, 48]
            trg_ft = output_dict[data['trg_imname']] # [1, 1280, 48, 48]

            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft) #torch.Size([1, 1280, 333, 500])
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            h = trg_ft.shape[-2] # 333
            w = trg_ft.shape[-1] # 500
            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0]) # 207

            total = 0
            correct = 0

            for idx in range(len(data['src_kps'])):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx] # [164, 204]
                trg_point = data['trg_kps'][idx] # [398, 242]

                num_channel = src_ft.size(1)
                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                trg_vec = F.normalize(trg_vec) # HW, c
                cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

            cat_pck.append(correct / total)
        total_pck.extend(cat_pck)

        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--dataset_path', type=str, default='./SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/scratch/lt453/spair_ft/', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm', 'vggt', 'dino', 'croco'], default='sd', help="which dift version to use")
    parser.add_argument('--layer', type=int, default=4)
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    main(args)