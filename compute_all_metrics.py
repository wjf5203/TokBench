#!/usr/bin/env python3
# Copyright (c) 2024 Junfeng Wu, Dongliang Luo. All Rights Reserved.
"""
TokBench Evaluation Script.
"""


import os
import json
import numpy as np
from prettytable import PrettyTable
import argparse



def reset_ans(path):
    flatten_ans = dict()
    
    with open(path, 'r') as fp:
        ans = json.load(fp)
    
    for method in ans:
        flatten_ans[method] = dict()
        
    for method in ans:
        temp = ans[method]
        
        for setting in temp:
            flatten_ans[method][setting.split('_')[-1]] = []

    return flatten_ans
            

def extract_ans(flatten_ans, path):
    with open(path, 'r') as fp:
        ans = json.load(fp)
    print(ans.keys())
    
    for method in ans:

        if method in flatten_ans:
            temp = ans[method]
            for setting in temp:
                detail_ans = ans[method][setting]["details"]
          
                res = []
                for k, v in detail_ans.items():
                    # print(v["results"])
                    res.extend(v["results"])
                    
                flatten_ans[method][setting.split('_')[-1]].extend(res)


def find_group(x, range_list):
    for i in range(len(range_list)):
        x_min, x_max = range_list[i]
        if x_min <= x < x_max:
            return i

    return None







def get_args_parser():
    parser = argparse.ArgumentParser(' ', add_help=False)
    parser.add_argument('--tokenizer', type=str, default='chameleon')
    parser.add_argument('--setting', type=str, choices=["256","512","1024"], default='256')
    return parser


def summarize_text(args):
    ans_paths = [
        'ic13.json',
        'ic15.json', 
        'tt.json', 
        'textocr.json',
        'sroie.json',
        'cord.json',
        'infograph.json',
        'docvqa.json'
    ]
    ans_paths = ['output/'+ans for ans in ans_paths]



    flatten_ans = reset_ans('output/ic13.json')
    print(flatten_ans)
    for path in ans_paths:
        extract_ans(flatten_ans, path)

    for method in flatten_ans:
        temp = flatten_ans[method]
        for setting in temp:
            flatten_ans[method][setting] = sorted(flatten_ans[method][setting], key=lambda x: x["ratio"])
            print(f"{method} | {setting} >>> (Accuracy: {np.mean([x['accuracy'] for x in flatten_ans[method][setting]]):.2%} | 1-NED: {np.mean([x['ned'] for x in flatten_ans[method][setting]]):.2%})")


    setting = args.setting

    ratio_ranges = {
        '256':[0.02, 0.03, 0.04, 1],
        '512':[0.01, 0.02, 0.03, 1],
        '1024':[0.005, 0.01, 0.02, 1],
    }
    ratio_range = ratio_ranges[setting]

    len_range = len(ratio_range) - 1

    range_lists = []
    for i in range(len_range):
        range_lists.append( (ratio_range[i], ratio_range[i+1]) )

    # print(range_lists)

    tb_split = ['Method', 'Setting']
    tb_split.extend(['a' + repr(v).replace(" ", "") for v in range_lists])
    tb_split.append('mean_acc')
    tb_split.extend(['n' + repr(v).replace(" ", "") for v in range_lists])
    tb_split.append('mean_ned')
    tb = PrettyTable(tb_split)

    for method in flatten_ans:
        if method == 'ori':
            continue
        if method == 'titok' and setting in ['512','1024']:
            continue
        temp = flatten_ans[method]
        # for setting in temp:
        curve_dict = dict(acc=[[] for _ in range_lists], ned=[[] for _ in range_lists], ratio=range_lists)

        for v in flatten_ans[method][setting]:
            group_id = find_group(v['ratio'], range_lists)
            if group_id is not None:
                curve_dict['acc'][group_id].append(v['accuracy'])
                curve_dict['ned'][group_id].append(v['ned'])

        # print(f"{method} | {setting} >>> ".endswith(''))
        show_row = [method, setting]
        acc_res = []
        ned_res = []
        # for r in range(curve_dict['ratio']):
        
        acc_res = [np.mean(x) for x in curve_dict['acc']]
        ned_res = [np.mean(x) for x in curve_dict['ned']]
        acc_res.append(np.mean(acc_res))
        ned_res.append(np.mean(ned_res))

        acc_res = [f"{x:.2%}" for x in acc_res]
        ned_res = [f"{x:.2%}" for x in ned_res]

        show_row.extend(acc_res)
        show_row.extend(ned_res)

        tb.add_row(show_row)
            
    print(tb)   
            
def summarize_face(args):
    ans_paths = [
    'output/face.json']
    flatten_ans = reset_ans(ans_paths[0])
    for path in ans_paths:
        extract_ans(flatten_ans, path)
    for method in flatten_ans:
        temp = flatten_ans[method]
        for setting in temp:
            flatten_ans[method][setting] = sorted(flatten_ans[method][setting], key=lambda x: x["ratio"])
            print(f"{method} | {setting} >>> (similarity: {np.mean([x['similarity'] for x in flatten_ans[method][setting]]):.2%}  ")


    setting = args.setting

    ratio_ranges = {
        '256':[0.00, 0.1, 0.2, 1],
        '512':[0.00, 0.05, 0.2, 1],
        '1024':[0.00, 0.05, 0.1, 1],
    }
    ratio_range = ratio_ranges[setting]

    len_range = len(ratio_range) - 1

    range_lists = []
    for i in range(len_range):
        range_lists.append( (ratio_range[i], ratio_range[i+1]) )

    # print(range_lists)

    tb_split = ['Method', 'Setting']
    tb_split.extend(['F_sim' + repr(v).replace(" ", "") for v in range_lists])
    tb_split.append('mean_similarity')
    
    tb = PrettyTable(tb_split)

    for method in flatten_ans:
        if method == 'ori':
            continue
        temp = flatten_ans[method]
        # for setting in temp:
        curve_dict = dict(similarity=[[] for _ in range_lists], ned=[[] for _ in range_lists], ratio=range_lists)

        for v in flatten_ans[method][setting]:
            group_id = find_group(v['ratio'], range_lists)
            if group_id is not None:
                curve_dict['similarity'][group_id].append(v['similarity'])
    
        # print(f"{method} | {setting} >>> ".endswith(''))
        show_row = [method, setting]
        similarity_res = []
        # for r in range(curve_dict['ratio']):
        
        similarity_res = [np.mean(x) for x in curve_dict['similarity']]
        similarity_res.append(np.mean(similarity_res))

        similarity_res = [   f"{x:.2%}"  for x in similarity_res]

        show_row.extend(similarity_res)

        tb.add_row(show_row)
            
    print(tb)   
            



def main(args):
   summarize_text(args)
   summarize_face(args)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

