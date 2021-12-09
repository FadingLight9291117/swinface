import os
import numbers
import json
import os.path
from pathlib import Path
import sys
import itertools

import torch
import torch.utils.data as data
import cv2
import numpy as np


class WiderFaceDataset(data.Dataset):
    def __init__(self, data_dir, preproc=None, dataset_type='train'):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        elif isinstance(data_dir, Path):
            ...
        else:
            msg = f'{data_dir.__class__} must be str or Path'
            raise TypeError(msg)
        self.preproc = preproc
        self.dataset_type = dataset_type
        self.img_dir = data_dir / 'images'
        self.label_path = data_dir / 'label.json'
        self.labels = json.load(self.label_path.open(encoding='utf-8'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.img_dir / label['filename']

        img = cv2.imread(img_path.__str__())
        height, width = img.shape[:2]
        bboxes = label['bboxes']
        if self.dataset_type == 'train':
            landmss = label['landms']
        elif self.dataset_type == 'eval':
            landmss = [[-1] * 10] * len(bboxes)
        annos = np.zeros((0, 15))
        if len(bboxes) == 0:
            return annos
        for bbox, landms in zip(bboxes, landmss):
            has_landms = -1 if landms[0] < 0 else 1
            anno = bbox + landms + [has_landms]
            anno = np.array([anno], dtype=np.float64)
            annos = np.append(annos, anno, axis=0)
        target = np.array(annos)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


if __name__ == '__main__':
    data_dir = './widerface/train'
    dataset = WiderFaceDataset(data_dir)
