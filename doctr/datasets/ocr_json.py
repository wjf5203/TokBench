# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import json
import imageio

from .datasets import AbstractDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image, crop_bboxes_from_frame

__all__ = ["OCRJSON", "OCRJSONForTokBench", "VideoOCRJSONForTokBench"]


class OCRJSON(AbstractDataset):
    """IC13 dataset from `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/ic13-grid.png&src=0
        :align: center

    >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
    >>> from doctr.datasets import OCRJSON
    >>> test_set = OCRJSON(img_folder="/path/to/JPEG_Images",
    >>>                 label_path="/path/to/test.json")
    Args:
        img_folder: folder with all the images of the dataset
        label_path: json file with all annotations for the images
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        detection_task: whether the dataset should be used for detection task
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        # use_polygons: bool = False,
        recognition_task: bool = False,
        detection_task: bool = False,
        dataset_name: str = 'IC13',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder, pre_transforms=convert_target_to_relative if not recognition_task else None, **kwargs
        )
        if recognition_task and detection_task:
            raise ValueError(
                "`recognition_task` and `detection_task` cannot be set to True simultaneously. "
                + "To get the whole dataset with boxes and labels leave both parameters to False."
            )

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}"
            )

        self.data: list[tuple[Path | np.ndarray, str | dict[str, Any] | np.ndarray]] = []
        np_dtype = np.float32

        # img_names = os.listdir(img_folder)
        # load annotations from json file
        # with open(label_path, 'r') as fp:
        #     annos = json.load(fp)

        # for anno in tqdm(iterable=annos, desc=f"Preparing and Loading {dataset_name}", total=len(annos)):
        #     img_path = Path(img_folder, anno['file_name'])
        #
        #     words = anno['annotations']
        #     labels = [word['rec'] for word in words]
        #
        #     box_targets = np.array([word["bbox"] for word in words], dtype=np_dtype)
        #
        #
        #     if recognition_task:
        #         crops = crop_bboxes_from_image(img_path=img_path, geoms=box_targets)
        #         for crop, label in zip(crops, labels):
        #             self.data.append((crop, label))
        #     elif detection_task:
        #         self.data.append((img_path, box_targets))
        #     else:
        #         self.data.append((img_path, dict(boxes=box_targets, labels=labels)))
        with open(label_path, 'r') as fp:
            annos = json.load(fp)

        for anno in tqdm(iterable=annos, desc=f"Preparing and Loading {dataset_name}", total=len(annos)):
            img_path = Path(img_folder, anno['file_name'])

            words = anno['annotations']
            labels = [word['rec'] for word in words]

            box_targets = np.array([word["bbox"] for word in words], dtype=np_dtype)
            ratios = [word['ratio'] for word in words]

            if recognition_task:
                crops = crop_bboxes_from_image(img_path=img_path, geoms=box_targets)
                crop_shapes = [x.shape for x in crops]

                ider = 0
                for s in crop_shapes:
                    if 0 in s:
                        print(img_path)
                        print(s)
                        print(labels[ider])
                        raise RuntimeError
                    ider +=1
                for crop, label, ratio in zip(crops, labels, ratios):
                    self.data.append((crop, (label, [ratio])))
            elif detection_task:
                self.data.append((img_path, box_targets))
            else:
                self.data.append((img_path, dict(boxes=box_targets, labels=labels)))

class OCRJSONForTokBench(AbstractDataset):
    """IC13 dataset from `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/ic13-grid.png&src=0
        :align: center

    >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
    >>> from doctr.datasets import OCRJSONForTokBench
    >>> test_set = OCRJSONForTokBench(img_folder="/path/to/JPEG_Images",
    >>>                 label_path="/path/to/test.json")
    Args:
        img_folder: folder with all the images of the dataset
        label_path: json file with all annotations for the images
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        detection_task: whether the dataset should be used for detection task
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
            self,
            img_folder: str,
            label_path: str,
            # use_polygons: bool = False,
            recognition_task: bool = False,
            detection_task: bool = False,
            dataset_name: str = 'IC13',
            **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder, pre_transforms=convert_target_to_relative if not recognition_task else None, **kwargs
        )
        if recognition_task and detection_task:
            raise ValueError(
                "`recognition_task` and `detection_task` cannot be set to True simultaneously. "
                + "To get the whole dataset with boxes and labels leave both parameters to False."
            )

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}"
            )

        self.data: list[tuple[Path | np.ndarray, str | dict[str, Any] | np.ndarray]] = []
        np_dtype = np.float32

        # img_names = os.listdir(img_folder)
        # load annotations from json file
        # with open(label_path, 'r') as fp:
        #     annos = json.load(fp)

        # for anno in tqdm(iterable=annos, desc=f"Preparing and Loading {dataset_name}", total=len(annos)):
        #     img_path = Path(img_folder, anno['file_name'])
        #
        #     words = anno['annotations']
        #     labels = [word['rec'] for word in words]
        #
        #     box_targets = np.array([word["bbox"] for word in words], dtype=np_dtype)
        #
        #
        #     if recognition_task:
        #         crops = crop_bboxes_from_image(img_path=img_path, geoms=box_targets)
        #         for crop, label in zip(crops, labels):
        #             self.data.append((crop, label))
        #     elif detection_task:
        #         self.data.append((img_path, box_targets))
        #     else:
        #         self.data.append((img_path, dict(boxes=box_targets, labels=labels)))
        with open(label_path, 'r') as fp:
            annos = json.load(fp)

        for anno in tqdm(iterable=annos, desc=f"Preparing and Loading {dataset_name}", total=len(annos)):
            img_path = Path(img_folder, anno['file_name'])

            img_meta = dict(
                file_name=anno["file_name"],
                height=anno["height"],
                width=anno["width"]
            )

            words = anno['annotations']
            labels = [word['rec'] for word in words]

            box_targets = np.array([word["bbox"] for word in words], dtype=np_dtype)
            ratios = [word['ratio'] for word in words]

            if recognition_task:
                crops = crop_bboxes_from_image(img_path=img_path, geoms=box_targets)
                for crop, label, ratio, word in zip(crops, labels, ratios, words):
                    self.data.append((crop, (label, [ratio, word, img_meta])))
            elif detection_task:
                self.data.append((img_path, box_targets))
            else:
                self.data.append((img_path, dict(boxes=box_targets, labels=labels)))



        # for img_name in tqdm(iterable=img_names, desc="Preparing and Loading {}", total=len(img_names)):
        #     img_path = Path(img_folder, img_name)
        #     label_path = Path(label_folder, "gt_" + Path(img_name).stem + ".txt")
        #
        #     with open(label_path, newline="\n") as f:
        #         _lines = [
        #             [val[:-1] if val.endswith(",") else val for val in row]
        #             for row in csv.reader(f, delimiter=" ", quotechar="'")
        #         ]
        #     labels = [line[-1].replace('"', "") for line in _lines]
        #     # xmin, ymin, xmax, ymax
        #     box_targets: np.ndarray = np.array([list(map(int, line[:4])) for line in _lines], dtype=np_dtype)
        #
        #
        #     if recognition_task:
        #         crops = crop_bboxes_from_image(img_path=img_path, geoms=box_targets)
        #         for crop, label in zip(crops, labels):
        #             self.data.append((crop, label))
        #     elif detection_task:
        #         self.data.append((img_path, box_targets))
        #     else:
        #         self.data.append((img_path, dict(boxes=box_targets, labels=labels)))


class VideoOCRJSONForTokBench(AbstractDataset):
    """IC13 dataset from `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/ic13-grid.png&src=0
        :align: center

    >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
    >>> from doctr.datasets import OCRJSONForTokBench
    >>> test_set = VideoOCRJSONForTokBench(img_folder="/path/to/JPEG_Images",
    >>>                 label_path="/path/to/test.json")
    Args:
        img_folder: folder with all the images of the dataset
        label_path: json file with all annotations for the images
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        detection_task: whether the dataset should be used for detection task
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
            self,
            img_folder: str,
            label_path: str,
            # use_polygons: bool = False,
            recognition_task: bool = False,
            detection_task: bool = False,
            dataset_name: str = 'ICDAR',
            **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder, pre_transforms=convert_target_to_relative if not recognition_task else None, **kwargs
        )
        if recognition_task and detection_task:
            raise ValueError(
                "`recognition_task` and `detection_task` cannot be set to True simultaneously. "
                + "To get the whole dataset with boxes and labels leave both parameters to False."
            )

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}"
            )

        self.data: list[tuple[Path | np.ndarray, str | dict[str, Any] | np.ndarray]] = []
        np_dtype = np.float32


        with open(label_path, 'r') as fp:
            annos = json.load(fp)


        last_vedio = '###'
        for anno in tqdm(iterable=annos, desc=f"Preparing and Loading {dataset_name}", total=len(annos)):
            img_path = os.path.join(img_folder, anno['file_name'])
            video_path = os.path.join(img_folder, anno['video_name'] + '.mp4')

            if anno['video_name'] != last_vedio:
                video_reader = imageio.get_reader(video_path, "ffmpeg")
                frames = [frame for frame in video_reader]
                last_vedio = anno['video_name']

            words = anno['annotations']
            if len(words) == 0:
                continue

            frame_id = os.path.splitext(anno['file_name'])[0]
            frame_id = int(frame_id.split("-frame")[-1])
            img_meta = dict(
                file_name=anno["file_name"],
                height=anno["height"],
                width=anno["width"],
                video_name=anno["video_name"],
                frame_id=frame_id
            )

            img = frames[frame_id-1]

            # words = anno['annotations']
            # if len(words) == 0:
            #     continue
            labels = [word['rec'] for word in words]

            box_targets = np.array([word["bbox"] for word in words], dtype=np_dtype)
            ratios = [word['ratio'] for word in words]

            if recognition_task:
                crops = crop_bboxes_from_frame(img, geoms=box_targets, path=img_path)
                for crop, label, ratio, word in zip(crops, labels, ratios, words):
                    if min(crop.shape[0], crop.shape[1]) <= 1:
                        print(anno["file_name"], crop.shape)
                        continue
                    self.data.append((crop, (label, [ratio, word, img_meta])))
            elif detection_task:
                self.data.append((img_path, box_targets))
            else:
                self.data.append((img_path, dict(boxes=box_targets, labels=labels)))


