import os
import sys
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
import math



def smart_padding(img, target_size=(512, 512), background_color=(0, 0, 0)):
    """
    padding to target size, keep the image ori ratio and return the metainfo
    """
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (math.ceil(original_width * ratio), math.ceil(original_height * ratio))
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    padded_img = Image.new("RGB", target_size, background_color)
    
    paste_position = (
        (target_width - new_size[0]) // 2,
        (target_height - new_size[1]) // 2
    )
    
    padded_img.paste(resized_img, paste_position)
    
    meta = {
        "original_size": (original_width, original_height),
        "scaled_size": new_size,
        "paste_position": paste_position,
        "target_size": target_size
    }
    
    return padded_img, meta

def restore_original(padded_img, meta):
    """
    resize and crop to ori size
    """
    scaled_img = padded_img.crop((
        meta["paste_position"][0],
        meta["paste_position"][1],
        meta["paste_position"][0] + meta["scaled_size"][0],
        meta["paste_position"][1] + meta["scaled_size"][1]
    ))
    restored_img = scaled_img.resize(
        meta["original_size"], 
        Image.LANCZOS
    )
    return restored_img


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

        final_img = restore_original(padded_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
