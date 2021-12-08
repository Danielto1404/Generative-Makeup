import os.path
from pathlib import Path
from typing import Union, Tuple

import albumentations as A
import numpy
import numpy as np
import torch.utils.data
import torchvision.transforms
from PIL import Image
from pycocotools import coco


class CocoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            annotation_path: Union[str, Path],
            images_root: [str, Path],
    ):
        """
        :param annotation_path: Path to the annotation json annotation file
        :param images_root:     Path to the images folder
        """
        self.coco = coco.COCO(annotation_path)
        self.root = images_root
        self._ids = dict(enumerate(self.coco.imgs.keys()))

    def _get_annotations(self, index):
        image_index = self._ids[index]
        annotations = self.coco.getAnnIds(imgIds=image_index)
        annotations = self.coco.loadAnns(annotations)
        return annotations

    def _get_image_json(self, index):
        image_index = self._ids[index]
        [image_json] = self.coco.loadImgs(image_index)
        return image_json

    def build_mask(self, image_shape: (int, int), annotations: list) -> numpy.ndarray:
        """

        :param image_shape:
        :param annotations:
        :return: (W x H)
        """
        mask = np.zeros(image_shape)
        for annotation in annotations:
            class_mask = self.coco.annToMask(annotation)
            # class_mask[class_mask != 0] = annotation['category_id']
            class_mask[class_mask != 0] = 1
            mask += class_mask
        return mask

    def read_image(self, file_name) -> numpy.ndarray:
        """
        :param file_name: file name in root images root folder
        :return: C x W x H numpy image
        """
        image = Image.open(os.path.join(self.root, file_name))
        image = numpy.asarray(image)
        return image

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param index: index of the item in dataset
        :return: (image: C x W x H, mask: W x H)
        """
        annotations = self._get_annotations(index)
        image_json = self._get_image_json(index)

        w, h, file_name = image_json['width'], image_json['height'], image_json['file_name']
        mask_ = self.build_mask((w, h), annotations)
        image = self.read_image(image_json['file_name'])
        return image, mask_

    def train_val_split(self, val_size=0.0):
        assert 0 <= val_size <= 1, 'Val size must be in range [0, 1]'
        train_size = int(len(self) * (1 - val_size))
        train, val = torch.utils.data.random_split(self, lengths=[train_size, len(self) - train_size])
        return CocoSegmentationSubset(self, train.indices, mode='train'), \
               CocoSegmentationSubset(self, val.indices, mode='val')

    def __len__(self):
        return len(self._ids)


class CocoSegmentationSubset(torch.utils.data.Subset):
    def __init__(self, dataset: CocoSegmentationDataset, indices, mode='train'):
        assert isinstance(dataset, CocoSegmentationDataset), 'Dataset must be an instance of CocoSegmentationDataset'
        assert mode in ['train', 'val'], 'The mode must be either "train" or "val"'
        super().__init__(dataset, indices)
        self.mode = mode

    @property
    def to_tensor(self):
        return torchvision.transforms.ToTensor()

    @property
    def normalize(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def transform(self):
        if self.mode == 'train':
            return A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.01,
                    scale_limit=0,
                    rotate_limit=20,
                    p=0.5)
            ])
        elif self.mode == 'val':
            return A.NoOp()
        else:
            raise NotImplementedError(f'Unsupported mode type: {self.mode}')

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[self.indices[index]]
        # print(mask.shape)
        transformed = self.transform(image=image, mask=mask)
        # print(transformed['mask'].shape, type(transformed['mask'])),
        image, mask = transformed['image'], transformed['mask']
        return self.normalize(image).float(), torch.tensor(mask).long()

#
# dataset = CocoSegmentationDataset('../../data/ann2.json', '../../data/images/')
#
# t, v = dataset.train_val_split(0.3)
#
# print(t[0][1].shape)
