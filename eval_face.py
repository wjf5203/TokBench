#!/usr/bin/env python3
# Copyright (c) 2024 Junfeng Wu, Dongliang Luo. All Rights Reserved.
"""
TokBench Evaluation Script.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import json
import os
import argparse
from tqdm import tqdm



def get_args_parser():
    parser = argparse.ArgumentParser('Set ', add_help=False)
    parser.add_argument('--original_image_path', type=str, default='/data3/jfwu/SSD/face_data/WFLW_images_all')
    parser.add_argument('--reconstruction_image_path', type=str, default='/data3/jfwu/RecBench/results_face/chameleon/face_256/')
    parser.add_argument('--tokenizer', type=str, default='chameleon')
    parser.add_argument('--setting', type=str, choices=["256","512","1024"], default='256')
    parser.add_argument('--meta_path', type=str, default='face_meta.json')
    parser.add_argument('--save_dir', type=str, default='output')
    return parser

def main(args):
     
    app = FaceAnalysis( name="antelopev2", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    all_images = json.load(open(args.meta_path,'r'))

    eval_results = {}
    all_similarity = []
    for image_meta in tqdm(all_images):
        image_path = image_meta['image_name']
        img_ori = cv2.imread(os.path.join(args.original_image_path,image_path))
        img_rec = cv2.imread(os.path.join(args.reconstruction_image_path,image_path))
        eval_results[image_path]={
            "file_name": image_path,
            "height": image_meta['img_height'],
            "width": image_meta['img_width'],
            "avg_acc": None,
            "avg_ned": None,
            "results": []
            }
        for face in image_meta['faces']:
            face_input = insightface.app.common.Face(kps=np.array(face['kps']))
            embed1 = app.models['recognition'].get(img_ori, face_input)
            embed2 = app.models['recognition'].get(img_rec, face_input)
            similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
            all_similarity.append(similarity)
            eval_results[image_path]["results"].append( {"ratio":face["ratio"], 
                                                         "similarity":float(similarity),
                                                         "box":face["box"],
                                                         "gt":face["gt"],
                                                           } )
    mean_similarity = np.mean(all_similarity)
    mean_similarity = float(mean_similarity)
    
    
    result_log_path = os.path.join(args.save_dir, "face.json" )
    if os.path.exists(result_log_path):
        with open(result_log_path, 'r') as fp:
            log_results = json.load(fp)
    else:
        log_results = dict()

    method_name = args.tokenizer
    method_setting = args.setting
    if method_name not in log_results:
        log_results[method_name] = dict()
    log_results[method_name][method_setting] = dict(
        results=dict(
            mean_similarity = mean_similarity
        ),
        details= eval_results
    )
    
    with open(result_log_path, 'w') as fp:
        json.dump(log_results, fp, indent=2)
    
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
