import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import CocoSegmentationDataset
from model import SegmentationModel


def train(model, train_loader, optimizer, epochs=5, val_loader=None, device='cpu', verbose=False):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    progress = trange(epochs, desc="Epochs") if verbose else range(epochs)
    for i in progress:
        print('~' * 80)
        print(f'---> epoch: {i + 1} / {len(progress)}')
        model.train()
        train_loss = 0
        for batch_index, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device).float(), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images, masks)
            loss = outputs['loss']
            loss.backward()
            error = loss.item()
            optimizer.step()
            train_loss += error
            if verbose:
                progress.set_postfix_str(f'batch: {batch_index + 1} / {len(train_loader)} | loss: {error}')

        print(f'Train loss {train_loss / len(train_loader)}')
        if val_loader is None:
            continue

        model.eval()
        val_loss = 0
        for (images, masks) in val_loader:
            with torch.no_grad():
                outputs = model(images, masks)
                val_loss += outputs['loss'].item()

        print(f'Validation loss: {val_loss / len(val_loader)}')
        # print(f'Learning rate: {scheduler.get_last_lr()[0]}')

        # scheduler.step()


def train_from_args(
        annotation_path: str,
        images_root: str,
        model_path: str,
        model_name: str,
        val_size: float,
        num_classes: int,
        batch_size: int,
        num_epochs: int,
        lr: float,
        weight_decay: float,
        device: str = 'cpu',
        verbose: bool = False
):
    model = SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    train_dataset, val_dataset = CocoSegmentationDataset(
        annotation_path=annotation_path,
        images_root=images_root,
    ).train_val_split(val_size=val_size)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=num_epochs,
        device=device,
        verbose=verbose
    )

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path} successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', help='path to the json COCO annotations file')
    parser.add_argument('--images_root', help='path to images folder')
    parser.add_argument('--model_path', help='path to which the model will be saved')
    parser.add_argument('--model_name', help='model to train')
    parser.add_argument('--val_size', type=float, default=0.0, help='validation data %')
    parser.add_argument('--num_classes', type=int, help='num classes for segmentation')
    parser.add_argument('--num_epochs', type=int, default=5, help='amount of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in train loader')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate in Adam optimizer')
    parser.add_argument('--l2', type=float, default=0.0, help='weight_decay in Adam optimizer')
    parser.add_argument('--device', default='cpu', help='torch device to use')
    parser.add_argument('--verbose', type=bool, default=False, help='if true than shows the progress')

    args = parser.parse_args()

    train_from_args(
        annotation_path=args.annotation_path,
        images_root=args.images_root,
        model_path=args.model_path,
        model_name=args.model_name,
        val_size=args.val_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.l2,
        device=args.device
    )
