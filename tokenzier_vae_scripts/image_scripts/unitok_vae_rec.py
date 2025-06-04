import os
import torch
import  json
import os
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
import sys
sys.path.append('UniTok')

from utils.config import Args
from models.unitok import UniTok
from utils.data import normalize_01_into_pm1
from torchvision.transforms import transforms, InterpolationMode

from PIL import Image, ImageOps
import math
from resize_rec import smart_padding, restore_original

 

    
    
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--image_path', type=str, default='/home/jfwu/projects/RecBench/RecBench/ic15_test')
    parser.add_argument('--save_path', type=str, default='/path/to/reconstruction/images/text_data/ic13')
    parser.add_argument('--padding_size', type=int, default=1024)
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
        
    # load model
    ckpt_path = '/data3/jfwu/tokenizer_modelzoo/unitok_tokenizer.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt['trainer']['unitok'])
    unitok.to('cuda')
    unitok.eval()

    preprocess = transforms.Compose([
        # transforms.Resize(int(unitok_cfg.img_size * unitok_cfg.resize_ratio)),
        # transforms.CenterCrop(unitok_cfg.img_size),
        transforms.ToTensor(), normalize_01_into_pm1,
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
        img = preprocess(padded_img).unsqueeze(0).to('cuda')
        with torch.no_grad():
            code_idx = unitok.img_to_idx(img)
            rec_img = unitok.idx_to_img(code_idx)
        
        rec_img = tensor_to_pilimg(rec_img)
         
        final_img = restore_original(rec_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))
        
        
 
    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

