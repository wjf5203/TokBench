import os
import sys
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
import imageio
import math
from resize_rec import resize_padding_images,restore_images

 
sys.path.append('Cosmos-Tokenizer')

import importlib
import cosmos_tokenizer.video_lib
import mediapy as media

importlib.reload(cosmos_tokenizer.video_lib)
from cosmos_tokenizer.video_lib import CausalVideoTokenizer


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--video_path', type=str, default='/data3/jfwu/RecBench/videos/ori/text/DS/')
    parser.add_argument('--save_path', type=str, default='/data3/jfwu/RecBench/video_results/debug')
    parser.add_argument('--model', type=str, choices=['CosmosCV4x8x8','CosmosCV8x16x16','CosmosDV4x8x8','CosmosDV8x16x16'], default='CosmosCV4x8x8')
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
     


    model_name_dict = {
        'CosmosCV4x8x8':'Cosmos-0.1-Tokenizer-CV4x8x8',
        'CosmosCV8x16x16':'Cosmos-0.1-Tokenizer-CV8x16x16',
        'CosmosDV4x8x8':'Cosmos-0.1-Tokenizer-DV4x8x8',
        'CosmosDV8x16x16':'Cosmos-0.1-Tokenizer-DV8x16x16',
    }
    model_name = model_name_dict[args.model]
    temporal_window = 49 # @param {type:"slider", min:1, max:121, step:8}

    encoder_ckpt = f"/data3/jfwu/SSD/video_model_zoo/{model_name}/encoder.jit"
    decoder_ckpt = f"/data3/jfwu/SSD/video_model_zoo/{model_name}/decoder.jit"
    tokenizer = CausalVideoTokenizer(
            checkpoint_enc=encoder_ckpt,
            checkpoint_dec=decoder_ckpt,
            device="cuda",
            dtype="bfloat16",
        )

    for chunk in tqdm(chunk_inputs):
        video_path = os.path.join(args.video_path,chunk[0])
        
        

        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        frames = [frame[:,:,::-1] for frame in video_reader]  # cosmos need BGR format
        video_reader.close()
        input_frames, operation_metas = resize_padding_images(frames, short_size=args.short_size)

        input_video = np.stack(input_frames,axis=0)
        batched_input_video = np.expand_dims(input_video, axis=0) # input  B x Tx H x W x C
        # 5) Create the CausalVideoTokenizer instance with the encoder & decoder.
        #    - device="cuda" uses the GPU
        #    - dtype="bfloat16" expects Ampere or newer GPU (A100, RTX 30xx, etc.)


        # 6) Use the tokenizer to autoencode (encode & decode) the video.
        #    The output is a NumPy array with shape = B x T x H x W x C, range [0..255].
        batched_output_video = tokenizer(batched_input_video,
                                        temporal_window=temporal_window)
        # 7) Extract the single video from the batch (index 0).
        output_video = batched_output_video[0]


        output_frames = [frame[:,:,::-1] for frame in output_video] # save in RGB format

        output_frames = restore_images(output_frames,operation_metas)

        frames = np.stack(output_frames,axis=0)
        frames = frames.astype(np.uint8) # [numframe,h,w,3]
        writer = imageio.get_writer('{}/{}'.format(image_save_pth,chunk[0]), fps=video_fps)
        for frame in frames:
            writer.append_data(frame ) 
        writer.close()

    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
