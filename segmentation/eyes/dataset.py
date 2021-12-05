import os.path
from pathlib import Path
from typing import Union, Tuple

import numpy
import numpy as np
import torch.utils.data
from PIL import Image
from pycocotools import coco


class CocoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 annotation_path: Union[str, Path],
                 images_root: [str, Path],
                 device='cpu',
                 transform=None):
        """
        :param annotation_path: Path to the annotation json annotation file
        :param images_root:     Path to the images folder
        :param device           Torch device
        :param transform:       Image torch transformation function
        """
        self.coco = coco.COCO(annotation_path)
        self.root = images_root
        self._ids = dict(enumerate(self.coco.imgs.keys()))
        self.device = device
        self.transform = transform

    def _get_annotations(self, index):
        image_index = self._ids[index]
        annotations = self.coco.getAnnIds(imgIds=image_index)
        annotations = self.coco.loadAnns(annotations)
        return annotations

    def _get_image_json(self, index):
        image_index = self._ids[index]
        [image_json] = self.coco.loadImgs(image_index)
        return image_json

    def _build_mask(self, image_shape: (int, int), annotations: list) -> torch.Tensor:
        mask = np.zeros(image_shape)
        for annotation in annotations:
            class_mask = self.coco.annToMask(annotation)
            class_mask[class_mask != 0] = annotation['category_id']
            mask += class_mask
        return torch.tensor(mask).long()

    def _read_image(self, file_path) -> torch.Tensor:
        image = Image.open(os.path.join(self.root, file_path))
        image = numpy.asarray(image).transpose((2, 0, 1))
        image = torch.tensor(image).float()
        return image if self.transform is None else self.transform(image)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param index: index of the item in dataset

        :return: (image: W x H x C, mask: W x H)
        """
        annotations = self._get_annotations(index)
        image_json = self._get_image_json(index)

        w, h, file_name = image_json['width'], image_json['height'], image_json['file_name']
        mask = self._build_mask((w, h), annotations)
        image = self._read_image(image_json['file_name'])
        return image.to(self.device), mask.to(self.device)

    def __len__(self):
        return len(self._ids)
