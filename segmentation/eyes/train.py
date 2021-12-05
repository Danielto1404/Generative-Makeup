import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
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


def run_from_args(
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
    model = SegmentationModel(num_classes).to(device)
    dataset = CocoSegmentationDataset(
        annotation_path=annotation_path,
        images_root=images_root,
        transform=transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
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
