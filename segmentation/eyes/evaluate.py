import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CocoSegmentationDataset


def evaluate(model, loader: DataLoader):
    val_loss = 0
    for (i, (images, masks)) in tqdm(loader):
        with torch.no_grad():
            outputs = model(images, masks)
            val_loss += outputs['loss'].item()

    print(f'Validation loss: {val_loss}')


def evaluate_from_args(
        annotation_path: str,
        images_root: str,
        load_path: str,
        batch_size: int,
        device: str
):
    dataset = CocoSegmentationDataset(
        annotation_path=annotation_path,
        images_root=images_root
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
    model = torch.load(load_path, map_location=device)
    evaluate(model, loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', help='path to the json COCO validation annotations file')
    parser.add_argument('--images_root', help='path to images folder')
    parser.add_argument('--load_path', help='model checkpoint path')
    parser.add_argument('--batch_size', help='batch size')
    parser.add_argument('--device', help='torch device')
    args = parser.parse_args()
    evaluate_from_args(
        annotation_path=args.annotation_path,
        images_root=args.images_root,
        load_path=args.load_path,
        batch_size=args.batch_size,
        device=args.device
    )
