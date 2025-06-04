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
# sys.path.append('TokenFlow')
from TokenFlow.tokenflow.tokenizer.vq_model import VQ_models



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


def encode_images(model, images, model_type='vavae'):
    with torch.no_grad():
        posterior = model.encode(images) 
    return posterior.sample().to(torch.float32)

def decode_to_images(model, z):
    with torch.no_grad():
        images = model.decode(z)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images


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
    
    vq_model = VQ_models["TokenFlow"](
        codebook_size=32768,
        codebook_embed_dim=8,
        semantic_code_dim=32,
        teacher="clipb_224",
        enhanced_decoder=True,
        infer_interpolate=True,
        resolution = args.padding_size
    )
    vq_model.to(device)
    vq_model.eval() # important
    checkpoint = torch.load( '/data3/jfwu/model_zoo/TokenFlow/tokenflow_clipb_32k_enhanced.pt' , map_location="cpu")
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    vq_model.load_state_dict(model_weight)
    del checkpoint

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


        input_img = transform(padded_img).unsqueeze(0).to(device)    


        with torch.no_grad():
            latent, _, _ = vq_model.encode(input_img)
            # samples = vq_model.decode_code(indices, latent.shape) # output value is between [-1, 1]
            samples = vq_model.decode(latent) # output value is between [-1, 1]
            if isinstance(samples, tuple):
                samples = samples[1]

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        rec_img = Image.fromarray(samples[0])
        final_img = restore_original(rec_img, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
