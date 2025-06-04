from torch import nn
import torch
import  json
import os
from tqdm import tqdm
import argparse
from torch.nn import functional as F

from chameleon.inference.image_tokenizer import ImageTokenizer
import  numpy as np

from PIL import Image, ImageOps
import math
from resize_rec import smart_padding, restore_original



    
    
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--image_path', type=str, default='/home/jfwu/projects/RecBench/RecBench/ic15_test')
    parser.add_argument('--save_path', type=str, default='/home/jfwu/projects/RecBench/RecBench/aresults/ic15_test')
    parser.add_argument('--padding_size', type=int, default=256)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def main(args):
    image_set_id = args.chunk_idx
    num_chunks=args.num_chunks
    batch_size = args.batch_size
    padding_size = args.padding_size
    image_save_pth = '{}_{}'.format(args.save_path,str(padding_size))
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth, exist_ok=True)
    
    vqgan_cfg_path = "chameleon/vqgan.yaml"
    vqgan_ckpt_path = "chameleon/vqgan.ckpt"
    image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda",)


    all_datas = os.listdir(args.image_path)
    all_datas.sort()
    
    chunked_filenames = np.array_split(all_datas, num_chunks)
    subset = chunked_filenames[image_set_id].tolist()
    chunk_inputs = split_list(subset, batch_size)
     
    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path,chunk[0])
        
        target_size = (padding_size, padding_size)  # 支持任意方形尺寸
    
        original_img = Image.open(image_path)
        
        # 3. 智能填充（保持原始比例）
        padded_img, meta = smart_padding(original_img, target_size)
        
        # 4. 自定义处理模块（ ）
        vq_code =  image_tokenizer.img_tokens_from_pil(padded_img) 
        feature_size = padding_size//16
        rec_img = image_tokenizer.pil_from_img_toks(vq_code  ,h_latent_dim=feature_size,w_latent_dim=feature_size)
        # 5. 还原原始尺寸
        final_img = restore_original(rec_img, meta)
        # 6. 保存结果
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))
        
        
 
    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)