import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import CocoSegmentationDataset
from model import SegmentationModel


def train(model, train_loader, optimizer, epochs, val_loader=None, verbose=True):
    progress = trange(epochs, desc="Epochs") if verbose else range(epochs)
    for _ in progress:
        model.train()
        train_loss = 0
        for batch_index, (images, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images, masks)
            loss = outputs['loss']
            loss.backward()
            error = loss.item()
            optimizer.step()
            train_loss += error
            if verbose:
                progress.set_postfix_str(f'batch: {batch_index + 1} / {len(train_loader)} | loss: {error}')

        print()
        print(f'Train loss {train_loss / len(train_loader)}')
        if val_loader is None:
            continue

        model.eval()
        val_loss = 0
        for (images, masks) in val_loader:
            with torch.no_grad():
                outputs = model(images, masks)
                val_loss += outputs['loss'].item()

        print(f'Validation loss: {val_loss} / {len(val_loader)}')
        print('~' * 80)


def train_from_args(
        annotation_path: str,
        images_root: str,
        model_path: str,
        num_classes: int,
        batch_size: int,
        num_epochs: int,
        lr: float,
        weight_decay: float,
        device: str = 'cpu'
):
    model = SegmentationModel(num_classes=num_classes).to(device)
    dataset = CocoSegmentationDataset(
        annotation_path=annotation_path,
        images_root=images_root,
        device=device,
    )
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=num_epochs,
        verbose=True
    )

    torch.save(model, model_path)
    print(f"Model saved to {model_path} successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', help='path to the json COCO annotations file')
    parser.add_argument('--images_root', help='path to images folder')
    parser.add_argument('--model_path', help='path to which the model will be saved')
    parser.add_argument('--num_classes', type=int, help='num classes for segmentation')
    parser.add_argument('--num_epochs', type=int, help='amount of training epochs')
    parser.add_argument('--batch_size', type=int, help='batch size in train loader')
    parser.add_argument('--lr', type=float, help='learning rate in Adam optimizer')
    parser.add_argument('--l2', type=float, help='weight_decay in Adam optimizer')
    parser.add_argument('--device', help='torch device to use')

    args = parser.parse_args()

    train_from_args(
        annotation_path=args.annotation_path,
        images_root=args.images_root,
        model_path=args.model_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.l2,
        device=args.device
    )
