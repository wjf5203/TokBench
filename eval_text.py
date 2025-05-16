#!/usr/bin/env python3
# Copyright (c) 2024 Junfeng Wu, Dongliang Luo. All Rights Reserved.
"""
TokBench Evaluation Script.
"""

import os
import multiprocessing as mp
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import Normalize

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import datasets
from doctr import transforms as T
from doctr.datasets import VOCABS, OCRJSONForTokBench
from doctr.models import recognition
from doctr.utils.metrics import TextMatch, RecMetric, RecMetricWithDetails

from glob import glob
import json
from pathlib import Path


@torch.inference_mode()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Assessment Level
    ratios = []
    img_metas = []
    gt_annos = []
    # Validation loop
    val_loss, batch_cnt = 0, 0
    pbar = tqdm(val_loader)
    for images, listed_targets in pbar:
        # print(listed_targets)
        targets = [t[0] for t in listed_targets]
        # ratios = [t[1][0] for t in listed_targets]
        # print(targets, ratios)
        try:
            if torch.cuda.is_available():
                images = images.cuda()
            images = batch_transforms(images)
            if amp:
                with torch.cuda.amp.autocast():
                    out = model(images, targets, return_preds=True)
            else:
                out = model(images, targets, return_preds=True)
                # Compute metric
            if len(out["preds"]):
                words, _ = zip(*out["preds"])
            else:
                words = []

            val_metric.update(targets, words)
            ratios.extend([t[1][0] for t in listed_targets])
            gt_annos.extend([t[1][1] for t in listed_targets])
            img_metas.extend([t[1][2] for t in listed_targets])
            # print(targets, words)

            val_loss += out["loss"].item()
            batch_cnt += 1
            # instance_cnt
        except ValueError:
            pbar.write(f"unexpected symbol/s in targets:\n{targets} \n--> skip batch")
            continue

    val_loss /= batch_cnt
    # result = val_metric.summary()
    result = val_metric.detail()
    # print(ratios)
    result.update(val_loss=val_loss, ratios=ratios, gt_annos=gt_annos, img_metas=img_metas)
    return result


def check_img_filename(img_folder):
    print('Checking image filenames')
    filenames = list(glob(os.path.join(img_folder, '*.jpg')))
    for filename in filenames:
        if '.jpg.jpg' in filename:

            new_filename = filename.replace('.jpg.jpg', '.jpg')

            os.rename(filename, new_filename)
            print(f"Renamed: {filename} -> {new_filename}")


def main(args):
    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")

    pbar = tqdm(disable=False if slack_token and slack_channel else True)
    if slack_token and slack_channel:
        # Monkey patch tqdm write method to send messages directly to Slack
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))

    torch.backends.cudnn.benchmark = True

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    # Load doctr model
    model = recognition.__dict__[args.arch](
        pretrained=True if args.resume is None else False,
        input_shape=(3, args.input_size, 4 * args.input_size),
        vocab=VOCABS[args.vocab],
    ).eval()

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    st = time.time()
    

    # ds = datasets.OCRJSON(
    ds=OCRJSONForTokBench(
        img_folder=args.img_folder,  # "/path/to/dataset/ocr/spotting/ic13/test_images",
        label_path=args.gt_path,
        dataset_name=args.dataset,
        # train=True,
        # download=True,
        recognition_task=True,
        detection_task=False,
        # use_polygons=args.regular,
        # gt_prefix=args.gt_prefix,
        img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(ds),
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.collate_fn,
    )
    pbar.write(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in {len(test_loader)} batches)")

    mean, std = model.cfg["mean"], model.cfg["std"]
    batch_transforms = Normalize(mean=mean, std=std)

    # Metrics
    # val_metric = TextMatch()
    val_metric = RecMetricWithDetails()

    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        pbar.write("No accessible GPU, target device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    pbar.write("Running evaluation")
    eval_result = evaluate(model, test_loader, batch_transforms, val_metric, amp=args.amp)
    # val_loss, exact_match, caseless_match, partial_match = eval_result["val_loss"], eval_result["raw"], eval_result["caseless"], eval_result["unicase"]
    # exact_ned, caseless_ned, partial_ned = eval_result["raw_ned"], eval_result["caseless_ned"], eval_result["unicase_ned"]
    # pbar.write(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Caseless: {exact_match:.2%} | Partial: {partial_match:.2%})")
    # pbar.write(
    #     f"[1-NED] (Exact: {exact_ned:.2%} | Caseless: {caseless_ned:.2%}) | Partial: {partial_ned:.2%}")
    exact_match = eval_result["raw"]
    exact_ned = eval_result["raw_ned"]
    exact_ned = [1-x for x in exact_ned]
    # batch_cnt = eval_result["batch_cnt"]

    assert len(exact_match) == len(exact_ned)
    instance_cnt = len(exact_match)

    avg_exact_match = sum(exact_match) / instance_cnt
    avg_exact_ned = sum(exact_ned) / instance_cnt

    # assert len(eval_result['ratio']) == batch_cnt

    pbar.write(f"Num Instance: {instance_cnt}")
    pbar.write(f"Exact Match (Accuracy: {avg_exact_match:.2%} | 1-NED: {avg_exact_ned:.2%})")

    # save result
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    result_log_path = os.path.join(args.save_dir, args.dataset + '.json')
    # overall_result_log_path = os.path.join(args.save_dir, args.dataset + '_overall.json')

    if os.path.exists(result_log_path):
        with open(result_log_path, 'r') as fp:
            log_results = json.load(fp)
    else:
        log_results = dict()

    # method_name = os.path.split(args.img_folder)[-1]
    method_name = args.method_name
    method_setting = args.setting
    if method_name not in log_results:
        log_results[method_name] = dict()
    log_results[method_name][method_setting] = dict(
        results=dict(
            accuracy = f"{avg_exact_match:.2%}",
            ned = f"{avg_exact_ned:.2%}"
        ),
        details=dict()
            # accuracy = exact_match,
            # ned = exact_ned,
            # ratio = eval_result['ratios']
    )

    # save_anno_in_log = True


    for img_meta, gt_anno, ratio, exact_match_per, exact_ned_per, r in zip(
            eval_result["img_metas"], eval_result["gt_annos"], eval_result['ratios'], exact_match, exact_ned, eval_result['results']):

        if img_meta["file_name"] not in log_results[method_name][method_setting]["details"]:
            log_results[method_name][method_setting]["details"][img_meta["file_name"]] = dict(
                file_name=img_meta["file_name"],
                height=img_meta["height"],
                width=img_meta["width"],
                avg_acc=0,
                avg_ned=0,
                results=[]
            )
        if args.save_anno:
            log_results[method_name][method_setting]["details"][img_meta["file_name"]]["results"].append(
                dict(
                    gt=r['gt'],
                    pred=r['pred'],
                    ratio=ratio,
                    accuracy=exact_match_per,
                    ned=exact_ned_per,
                    anno=gt_anno,
                )
            )
        else:
            log_results[method_name][method_setting]["details"][img_meta["file_name"]]["results"].append(
                dict(
                    gt=r['gt'],
                    pred=r['pred'],
                    ratio=ratio,
                    accuracy=exact_match_per,
                    ned=exact_ned_per
                )
            )

 

    pred_cnt = 0
    # save_anno_in_log = True



    for k in log_results[method_name][method_setting]["details"]:
        # if len(log_results[method_name][method_setting]["details"][k]["results"]) > 0:
        avg_acc = sum([x["accuracy"] for x in log_results[method_name][method_setting]["details"][k]["results"]]) / len(log_results[method_name][method_setting]["details"][k]["results"])
        avg_ned = sum([x["ned"] for x in log_results[method_name][method_setting]["details"][k]["results"]]) / len(log_results[method_name][method_setting]["details"][k]["results"])

        log_results[method_name][method_setting]["details"][k]['avg_acc'] = avg_acc
        log_results[method_name][method_setting]["details"][k]['avg_ned'] = avg_ned

        pred_cnt += len(log_results[method_name][method_setting]["details"][k]["results"])

    print(f"INFERENCE ON {pred_cnt} instances")

    with open(result_log_path, 'w') as fp:
        json.dump(log_results, fp, indent=2)


    if args.refine:
        print("refining GT")
        temp_ans = {}
        for img_meta, gt_anno, ratio, exact_match_per, exact_ned_per, r in zip(
                eval_result["img_metas"], eval_result["gt_annos"], eval_result['ratios'], exact_match, exact_ned,
                eval_result['results']):

            if img_meta["file_name"] not in temp_ans:
                temp_ans[img_meta["file_name"]] = dict(
                    file_name=img_meta["file_name"],
                    height=img_meta["height"],
                    width=img_meta["width"],
                    annotations=[]
                )
            if exact_match_per == 1:
                temp_ans[img_meta["file_name"]]["annotations"].append(gt_anno)
            # else:
            #     temp_ans[img_meta["file_name"]]["annotations"].append(gt_anno)


        # format new annotation json
        refined_anno = []
        for k in temp_ans:
            img_anno = temp_ans[k]
            if len(img_anno["annotations"]) > 0:
                refined_anno.append(img_anno)

        ins_cnt = []
        for img in refined_anno:
            temp = [x['ratio'] for x in img['annotations']]
            ins_cnt.extend(temp)
        print('Refined instance num: ', len(ins_cnt))

        with open(args.gt_path, 'w') as fp:
            json.dump(refined_anno, fp, indent=2)
    # img_metas


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--arch", type=str, default='parseq', help="text-recognition model to evaluate")
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for evaluation")
    parser.add_argument("--dataset", type=str, default="IC13", help="Dataset to evaluate on")
    parser.add_argument("--save_dir", type=str, default="./output", help="The folder to save logs")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("--img_folder", type=str, default="/path/to/dataset/ocr/spotting/ic13/test_images", help="The folder of test images")
    parser.add_argument("--gt_path", type=str, default="/path/to/dataset/ocr/bench/gt/Challenge2_Test_Task1_GT",
                        help="The GT folder of test images")
    parser.add_argument("--method_name", type=str, default="tokenizer1", help="The reconstruction method name")
    parser.add_argument("--setting", type=str, default="256", choices=["256","512","1024"], help="The evaluation setting [256,512,1024]")
    parser.add_argument("--gt_prefix", type=bool, default=False, help="Whether add prefix to GT's filenames")
    parser.add_argument("--replace", type=bool, default=False, help="Replace .jpg.jpg in filenames")
    parser.add_argument("--refine", type=bool, default=False, help="Removing false recognitions from gt?")
    parser.add_argument("--save_anno", type=bool, default=False, help="Save annotation in log?")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument(
        "--only_regular", dest="regular", action="store_true", help="test set contains only regular text"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    check_img_filename(args.img_folder)
    main(args)
