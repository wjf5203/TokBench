import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
# from dataset.augmentation import center_crop_arr
from resize_rec import smart_padding, restore_original
import sys
sys.path.append("1d-tokenizer")
from modeling.titok import TiTok
torch.manual_seed(0)


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
    
    device = 'cuda'
    # create and load model
    
    
    titok_tokenizer = TiTok.from_pretrained("/data3/jfwu/SSD/model_zoo/tokenizer_titok_l32_imagenet/")
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
 
    device = "cuda"
    titok_tokenizer = titok_tokenizer.to(device)



    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
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


        
        image = torch.from_numpy(np.array(padded_img).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
            reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
            reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
            reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            rec_img = Image.fromarray(reconstructed_image)

        final_img = restore_original(rec_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)





