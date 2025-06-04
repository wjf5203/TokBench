import argparse
import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import numpy as np

import os
from tqdm import tqdm
import argparse
from PIL import Image
import math

from resize_rec import resize_padding_images,restore_images

 


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--video_path', type=str, default='/data3/jfwu/RecBench/videos/ori/text/DS/')
    parser.add_argument('--save_path', type=str, default='/data3/jfwu/RecBench/video_results/debug')
    parser.add_argument('--short_size', type=int, default=480)
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
    short_size = args.short_size
    image_save_pth = '{}_{}'.format(args.save_path,str(short_size))
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth, exist_ok=True)

    
    all_datas = os.listdir(args.video_path)
    all_datas.sort()
    
    chunked_filenames = np.array_split(all_datas, num_chunks)
    subset = chunked_filenames[image_set_id].tolist()
    chunk_inputs = split_list(subset, batch_size)
     

    device = torch.device('cuda')
    dtype = torch.bfloat16

    model = AutoencoderKLCogVideoX.from_pretrained('/data3/jfwu/SSD/video_model_zoo/CogVideoX1.5-5B/vae/', torch_dtype=dtype).to(device)
    model.enable_slicing()
    model.enable_tiling()


    for chunk in tqdm(chunk_inputs):
        video_path = os.path.join(args.video_path,chunk[0])
        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        print('fps:',video_fps)
        frames = [ frame for frame in video_reader]
        video_reader.close()
        # frames = frames[:113] # 8 9 16 17 24 25

        ori_length = len(frames)
        padding_length = math.ceil((ori_length-1)/8) * 8 +1
        num_pad_frame = padding_length-ori_length
        padded_frames = [frames[-1]]*num_pad_frame
         


        frames.extend(padded_frames)
        input_frames, operation_metas = resize_padding_images(frames, short_size=args.short_size)

        frames = [transforms.ToTensor()(frame) for frame in input_frames]
        frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
         
        # print(frames_tensor.shape)
        with torch.no_grad():
            encoded_frames = model.encode(frames_tensor)[0].sample()
            decoded_frames = model.decode(encoded_frames).sample
        # print(decoded_frames.shape)
         
        decoded_frames = decoded_frames.to(dtype=torch.float32)
        decoded_frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        decoded_frames = np.clip(decoded_frames, 0, 1) * 255
        decoded_frames = decoded_frames.astype(np.uint8)
        output_frames = [frame for frame in decoded_frames]
        output_frames = restore_images(output_frames,operation_metas)[:ori_length] # crop the padded frames
         
        frames = np.stack(output_frames,axis=0)
        writer = imageio.get_writer('{}/{}'.format(image_save_pth,chunk[0]), fps=video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



