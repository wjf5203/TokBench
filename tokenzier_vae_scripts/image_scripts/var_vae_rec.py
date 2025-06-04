import os
import sys
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
import math
from torchvision.transforms import transforms
from resize_rec import smart_padding, restore_original


sys.path.append('VAR')
from models import VQVAE, build_vae_var
from utils.data import pil_loader, normalize_01_into_pm1

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--image_path', type=str, default='/path/to/TokBench/images/text_data/ic13')
    parser.add_argument('--save_path', type=str, default='/path/to/reconstruction/images/text_data/ic13')
    parser.add_argument('--padding_size', type=int, default=256)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def tensor_to_pilimg(img: torch.Tensor ):
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    img = Image.fromarray(img[0])
    return img



def main(args):
    image_set_id = args.chunk_idx
    num_chunks=args.num_chunks
    batch_size = args.batch_size
    padding_size = args.padding_size
    image_save_pth = '{}_{}'.format(args.save_path,str(padding_size))
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth, exist_ok=True)

    V=4096
    Cvae=32
    ch=160
    share_quant_resi=4

    if padding_size == 256:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    elif padding_size == 512:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    elif padding_size == 1024:
        patch_nums = (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64)
    else:
        assert False , 'unsupport size for VAR'
    device = 'cuda'
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)

    vae_local.load_state_dict(torch.load('/data3/jfwu/tokenizer_modelzoo/var/vae_ch160v4096z32.pth', map_location='cpu'), strict=True)
    vae_local.eval() 

    preprocess = transforms.Compose([
        # transforms.Resize(int(unitok_cfg.img_size * unitok_cfg.resize_ratio)),
        # transforms.CenterCrop(unitok_cfg.img_size),
        transforms.ToTensor() 
    ])
    all_datas = os.listdir(args.image_path)
    all_datas.sort()
    
    chunked_filenames = np.array_split(all_datas, num_chunks)
    subset = chunked_filenames[image_set_id].tolist()
    chunk_inputs = split_list(subset, batch_size)
     
    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path,chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        target_size = (padding_size, padding_size)  # 支持任意方形尺寸
        # 3. 智能填充（保持原始比例）
        padded_img, meta = smart_padding(original_img, target_size)
        input_img = preprocess(padded_img).unsqueeze(0).to('cuda')    
        input_img = normalize_01_into_pm1(input_img)
        with torch.no_grad():
            rec_img = vae_local.img_to_reconstructed_img(input_img, patch_nums ,last_one=True)
            rec_img = tensor_to_pilimg(rec_img)
        final_img = restore_original(rec_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
