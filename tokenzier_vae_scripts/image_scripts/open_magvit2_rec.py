import os
import sys
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
from torchvision.transforms import transforms
from resize_rec import smart_padding, restore_original
import argparse
import torchvision.transforms as T
from omegaconf import OmegaConf
import importlib


sys.path.append('SEED-Voken')
from src.Open_MAGVIT2.models.lfqgan import VQModel
from src.Open_MAGVIT2.models.lfqgan_pretrain import VQModel as VQModel_pretrain
from src.IBQ.models.ibqgan import IBQ


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = {
    "Open-MAGVIT2-pretrain": VQModel_pretrain,
    "Open-MAGVIT2": VQModel,
    "IBQ": IBQ
}

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
	model = MODEL_TYPE[model_type](**config.model.init_args)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

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


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--image_path', type=str, default='/path/to/TokBench/images/text_data/ic13')
    parser.add_argument('--save_path', type=str, default='/path/to/reconstruction/images/text_data/ic13')
    parser.add_argument('--model_name', type=str, default='magvit2_imgnet_f16', help='magvit2_imgnet_f16,magvit2_imgnet_f8, magvit2_pretrain_f16_16384, magvit2_pretrain_f16_262144')
    parser.add_argument('--padding_size', type=int, default=256)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]





def main(args):

    config_dict = {
        'magvit2_imgnet_f16':'SEED-Voken/configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml',
        'magvit2_imgnet_f8':'SEED-Voken/configs/Open-MAGVIT2/gpu/imagenet_lfqgan_128_L.yaml', 
        'magvit2_pretrain_f16_16384':'SEED-Voken/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_16384.yaml', 
        'magvit2_pretrain_f16_262144':'SEED-Voken/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml'
    }

    ckpt_dict = {
        'magvit2_imgnet_f16':'/data3/jfwu/tokenizer_modelzoo/Open_MAGVIT2/imagenet_256_L/imagenet_256_L.ckpt',
        'magvit2_imgnet_f8':'/data3/jfwu/tokenizer_modelzoo/Open_MAGVIT2/imagenet_128_L/imagenet_128_L.ckpt',
        'magvit2_pretrain_f16_16384':'/data3/jfwu/tokenizer_modelzoo/Open_MAGVIT2/pretrain256_16384/pretrain256_16384.ckpt', 
        'magvit2_pretrain_f16_262144':'/data3/jfwu/tokenizer_modelzoo/Open_MAGVIT2/pretrain256_262144/pretrain256_262144.ckpt'
    }

    config_file = config_dict[args.model_name]
    configs = OmegaConf.load(config_file)
    model_type = "Open-MAGVIT2" if 'magvit' in args.model_name else "IBQ"
    if 'pretrain' in args.model_name:
         model_type = "Open-MAGVIT2-pretrain"
    model = load_vqgan_new(configs, model_type , ckpt_dict[args.model_name]).to(DEVICE)

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
        # 3. center padding（keep original ratio)
        padded_img, meta = smart_padding(original_img, target_size)
        
        padded_img = np.array(padded_img)
        padded_img = padded_img / 127.5 - 1.0
        input_images = T.ToTensor()(padded_img).unsqueeze(0).to(device=DEVICE,  dtype=torch.float)
        
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    if model_type in ["Open-MAGVIT2", "Open-MAGVIT2-pretrain"]:
                        quant, diff, indices, _ = model.encode(input_images)
                    elif model_type == "IBQ":
                        quant, qloss, (_, _, indices) = model.encode(input_images)
                    reconstructed_images = model.decode(quant)
            else:
                if model_type == ["Open-MAGVIT2", "Open-MAGVIT2-pretrain"]:
                    quant, diff, indices, _ = model.encode(input_images)
                elif model_type == "IBQ":
                    quant, qloss, (_, _, indices) = model.encode(input_images)
                reconstructed_images = model.decode(quant)
        
        reconstructed_image = reconstructed_images[0]
        reconstructed_image = custom_to_pil(reconstructed_image)

        final_img = restore_original(reconstructed_image, meta)
        final_img.save('{}/{}'.format(image_save_pth,chunk[0]))


    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
