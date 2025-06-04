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
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--image_path', type=str, default='/path/to/TokBench/images/text_data/ic13')
    parser.add_argument('--save_path', type=str, default='/path/to/reconstruction/images/text_data/ic13')
    parser.add_argument('--model_name', type=str, default='sd3p5', help='sdxl,sd3p5,flux1')
    parser.add_argument('--padding_size', type=int, default=256)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def tensor_to_pilimg(img: torch.Tensor ):
    img = img.mul_(255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(img)
    return img



def main(args):
    image_set_id = args.chunk_idx
    num_chunks=args.num_chunks
    batch_size = args.batch_size
    padding_size = args.padding_size
    image_save_pth = '{}_{}'.format(args.save_path,str(padding_size))
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth, exist_ok=True)


    model_path_dict={
        'sdxl':'/data3/jfwu/tokenizer_modelzoo/sdxl-vae/',
        'sd3p5':'/data3/jfwu/tokenizer_modelzoo/sd3p5-large-vae/vae/',
        'flux1':'/data3/jfwu/tokenizer_modelzoo/flux1-vae/vae/'
    }
    model_path = model_path_dict[args.model_name]
    vae = AutoencoderKL.from_pretrained(model_path)
    vae.to(dtype=torch.float32)  # otherwise it produces NaNs, even madebyollin's VAE
    vae.to(device="cuda")

    all_datas = os.listdir(args.image_path)
    all_datas.sort()
    
    chunked_filenames = np.array_split(all_datas, num_chunks)
    subset = chunked_filenames[image_set_id].tolist()
    chunk_inputs = split_list(subset, batch_size)
     
    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path,chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        target_size = (padding_size, padding_size)  
        padded_img, meta = smart_padding(original_img, target_size)

        img_tensor = pil_to_tensor(padded_img).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(vae.device)
        input_img = img_tensor.to(vae.dtype)

        with torch.no_grad():
            latent = vae.encode(input_img, return_dict=False)[0].sample()
             
            rec_img = vae.decode(latent).sample
            rec_img = rec_img.squeeze(0).cpu().detach()
         
        # rec_img = transforms.ToPILImage()(rec_img)
        rec_img = tensor_to_pilimg(rec_img)
        final_img = restore_original(rec_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
