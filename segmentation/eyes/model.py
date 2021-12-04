import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import trange


class SegmentationModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        # Cached in Users/a19378208/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
        self.model = deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)

    def forward(self, image, mask=None) -> dict:
        outputs = self.model(image)
        if mask is not None:
            criterion = nn.CrossEntropyLoss()
            tensors = outputs['out']
            outputs['loss'] = criterion(tensors, mask)

        return outputs


def train(model, train_loader, optimizer, epochs, val_loader=None, verbose=True):
    progress = trange(epochs, desc="Epochs") if verbose else range(epochs)
    for i in range(epochs):
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

#
# model = SegmentationModel(3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# dataloader = DataLoader(data, shuffle=True, batch_size=2)
#
# train(model, dataloader, optimizer, epochs=2)
