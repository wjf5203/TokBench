import sys
sys.path.append("taming-transformers")
import torch
torch.set_grad_enabled(False)

import torch
from taming.models.vqgan import VQModel, GumbelVQ
import os, sys
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
import yaml

from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
# from dataset.augmentation import center_crop_arr
from resize_rec import smart_padding, restore_original



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


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x


def main(args):
    image_set_id = args.chunk_idx
    num_chunks=args.num_chunks
    batch_size = args.batch_size
    padding_size = args.padding_size
    image_save_pth = '{}_{}'.format(args.save_path,str(padding_size))
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth, exist_ok=True)
    
    device = 'cuda'
    # create and load model
    config16384 = load_config("/data3/jfwu/tokenizer_modelzoo/taming_vqgan_imagenet_f16_16384/model.yaml", display=False)
    model16384 = load_vqgan(config16384, ckpt_path="/data3/jfwu/tokenizer_modelzoo/taming_vqgan_imagenet_f16_16384/last.ckpt").to(device)
    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

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


        input_img = transform(padded_img).unsqueeze(0).to(device)    
        with torch.no_grad():
            z, _, [_, _, indices] = model16384.encode(preprocess_vqgan(input_img))
            xrec = model16384.decode(z)

        rec_img = custom_to_pil(xrec[0])
        final_img = restore_original(rec_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
