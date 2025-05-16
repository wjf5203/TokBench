# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .datasets import AbstractDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["ArbitraryText"]


class ArbitraryText(AbstractDataset):
    """IC13 dataset from `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/ic13-grid.png&src=0
        :align: center

    # >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
    # >>> from doctr.datasets import IC13
    # >>> train_set = IC13(img_folder="/path/to/Challenge2_Training_Task12_Images",
    # >>>                  label_folder="/path/to/Challenge2_Training_Task1_GT")
    # >>> img, target = train_set[0]
    # >>> test_set = IC13(img_folder="/path/to/Challenge2_Test_Task12_Images",
    # >>>                 label_folder="/path/to/Challenge2_Test_Task1_GT")
    # >>> img, target = test_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_folder: folder with all annotation files for the images
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        detection_task: whether the dataset should be used for detection task
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_folder: str,
        use_polygons: bool = False,
        recognition_task: bool = False,
        detection_task: bool = False,
        gt_prefix: bool = True,
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
        if not os.path.exists(label_folder) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_folder if not os.path.exists(label_folder) else img_folder}"
            )

        self.data: list[tuple[Path | np.ndarray, str | dict[str, Any] | np.ndarray]] = []
        np_dtype = np.float32

        img_names = os.listdir(img_folder)

        for img_name in tqdm(iterable=img_names, desc="Preparing and Loading IC13", total=len(img_names)):
            img_path = Path(img_folder, img_name)

            if gt_prefix:
                label_path = Path(label_folder, "gt_" + Path(img_name).stem + ".txt")
            else:
                label_path = Path(label_folder, Path(img_name).stem + ".txt")

            with open(label_path, newline="\n") as f:
                _lines = [
                    [val[:-1] if val.endswith(",") else val for val in row]
                    for row in csv.reader(f)
                ]

            valid_lines = []
            for x in _lines:
                if x[-1] == '#######':
                    pass
                else:
                    x[-1] = x[-1].replace("####", "")
                    valid_lines.append(x)

            labels = [line[-1] for line in valid_lines]

            # xmin, ymin, xmax, ymax
            box_targets = [list(map(int, line[:-1])) for line in valid_lines]
            if img_name == '0000093.jpg':
                print(box_targets)
                print(labels)
            if use_polygons:
                # (x, y) coordinates (x1, y1, x2, y2, ...)
                _box_targets = []
                for x in box_targets:
                    len_x = len(x)
                    try:
                        _box_targets.append(
                            [
                                max(0, min(x[0:len_x:2])),  # xmin
                                max(0, min(x[1:len_x:2])),  # ymin
                                max(x[0:len_x:2]),  # xmax
                                max(x[1:len_x:2]),  # xmax
                            ],
                        )
                    except:
                        print(img_name)
                box_targets = _box_targets
                # box_targets = np.array(
                #     [
                #         min(box_targets[0:len_coords:2]),   # xmin
                #         min(box_targets[1:len_coords:2]),   # ymin
                #         max(box_targets[0:len_coords:2]),   # xmax
                #         max(box_targets[1:len_coords:2]),   # xmax
                #     ],
                #     dtype=np_dtype,
                # )
            # else:
            box_targets = np.array(box_targets)
            # remove invalid


            if recognition_task:
                try:
                    crops = crop_bboxes_from_image(img_path=img_path, geoms=box_targets)
                except:
                    print(img_name)
                for crop, label in zip(crops, labels):
                    self.data.append((crop, label))
            elif detection_task:
                self.data.append((img_path, box_targets))
            else:
                self.data.append((img_path, dict(boxes=box_targets, labels=labels)))
