import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


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
