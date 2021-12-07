import os.path
from pathlib import Path
from typing import Union, Tuple

import albumentations as A
import numpy
import numpy as np
import torch.utils.data
from PIL import Image
from pycocotools import coco


class CocoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            annotation_path: Union[str, Path],
            images_root: [str, Path],
            device='cpu'
    ):
        """
        :param annotation_path: Path to the annotation json annotation file
        :param images_root:     Path to the images folder
        :param device           Torch device
        """
        self.coco = coco.COCO(annotation_path)
        self.root = images_root
        self._ids = dict(enumerate(self.coco.imgs.keys()))
        self.device = device

    def _get_annotations(self, index):
        image_index = self._ids[index]
        annotations = self.coco.getAnnIds(imgIds=image_index)
        annotations = self.coco.loadAnns(annotations)
        return annotations

    @property
    def _aug_transform(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(15, p=0.5)
        ])

    def _get_image_json(self, index):
        image_index = self._ids[index]
        [image_json] = self.coco.loadImgs(image_index)
        return image_json

    def build_mask(self, image_shape: (int, int), annotations: list) -> numpy.ndarray:
        mask = np.zeros(image_shape)
        for annotation in annotations:
            class_mask = self.coco.annToMask(annotation)
            class_mask[class_mask != 0] = annotation['category_id']
            mask += class_mask
        return mask

    def read_image(self, file_path) -> numpy.ndarray:
        image = Image.open(os.path.join(self.root, file_path))
        image = numpy.asarray(image).transpose((2, 0, 1))
        return image

    def augmentation(self, image, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self._aug_transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param index: index of the item in dataset

        :return: (image: W x H x C, mask: W x H)
        """
        annotations = self._get_annotations(index)
        image_json = self._get_image_json(index)

        w, h, file_name = image_json['width'], image_json['height'], image_json['file_name']
        mask = self.build_mask((w, h), annotations)
        image = self.read_image(image_json['file_name'])
        image, mask = self.augmentation(image, mask)
        return torch.tensor(image, device=self.device, dtype=torch.float), \
               torch.tensor(mask, device=self.device, dtype=torch.long)

    def get_numpy(self, index) -> Tuple[np.ndarray, np.ndarray]:
        image, mask_ = self[index]
        image = image.cpu().numpy()
        mask_ = mask_.cpu().numpy()
        return image, mask_

    def train_val_split(self, val_size=0.0):
        train_size = int(len(self) * (1 - val_size))
        return torch.utils.data.random_split(self, lengths=[train_size, len(self) - train_size])

    def __len__(self):
        return len(self._ids)
