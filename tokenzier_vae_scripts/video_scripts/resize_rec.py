import os
import sys
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import  numpy as np
import imageio
import math



def resize_padding_images(images,short_size):
    if not images:
        return [], {}
    
    H0, W0 = images[0].shape[:2]
    
    if H0 <= W0:
        short_side = 'height'
        new_h = short_size
        new_w = int(round(W0 * (short_size / H0)))
        l_new = new_w
    else:
        short_side = 'width'
        new_w = short_size
        new_h = int(round(H0 * (short_size / W0)))
        l_new = new_h
    
    target_l = math.ceil(l_new / 32) * 32
    pad_total = target_l - l_new
    
    if short_side == 'height':
        target_h, target_w = new_h, target_l
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_bottom = 0
    else:
        target_h, target_w = target_l, new_w
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_right = 0
    
    processed_images = []
    for img in images:
        img_pil = Image.fromarray(img.astype('uint8'))
        # Resize
        resized_img = img_pil.resize((new_w, new_h), Image.BILINEAR)
        # Pad
        padded_img = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        if short_side == 'height':
            padded_img.paste(resized_img, (pad_left, 0))
        else:
            padded_img.paste(resized_img, (0, pad_top))
        processed_images.append(np.array(padded_img))
    
    meta = {
        'original_size': (H0, W0),
        'resized_size': (new_h, new_w),
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'short_side': short_side,
        'target_size': (target_h, target_w)
    }
    
    return processed_images, meta

def restore_images(processed_images, meta):
    if not processed_images or not meta:
        return []
    
    H0, W0 = meta['original_size']
    new_h, new_w = meta['resized_size']
    pad_top = meta['pad_top']
    pad_bottom = meta['pad_bottom']
    pad_left = meta['pad_left']
    pad_right = meta['pad_right']
    target_h, target_w = meta['target_size']
    short_side = meta['short_side']
    
    restored_images = []
    for img in processed_images:
        if short_side == 'height':
            cropped = img[:, pad_left:target_w - pad_right, :]
        else:
            cropped = img[pad_top:target_h - pad_bottom, :, :]
        img_pil = Image.fromarray(cropped.astype('uint8'))
        restored_pil = img_pil.resize((W0, H0), Image.BILINEAR)
        restored_images.append(np.array(restored_pil))
    
    return restored_images




def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--video_path', type=str, default='/data3/jfwu/RecBench/videos/ori/text/DS/')
    parser.add_argument('--save_path', type=str, default='/data3/jfwu/RecBench/video_results/resize/debug')
    parser.add_argument('--short_size', type=int, default=256)
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
     
    for chunk in tqdm(chunk_inputs):
        video_path = os.path.join(args.video_path,chunk[0])
        
        

        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        frames = [ frame for frame in video_reader]
        video_reader.close()
        input_frames, operation_metas = resize_padding_images(frames, short_size=args.short_size)

        output_frames = restore_images(input_frames,operation_metas)

        frames = np.stack(output_frames,axis=0)
        frames = frames.astype(np.uint8) # [numframe,h,w,3]
        writer = imageio.get_writer('{}/{}'.format(image_save_pth,chunk[0]), fps=video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(image_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
