import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from utils import torch_normalize_image

arh_models = {
    'deeplabv3_resnet50': deeplabv3_resnet50,
    'deeplabv3_resnet101': deeplabv3_resnet101,
    'unet': []
}


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'deeplabv3_resnet50'):
        super().__init__()
        self.num_classes = num_classes
        constructor = arh_models.get(model_name)
        if constructor is None:
            raise NotImplemented(f'Model {model_name} not found, all models: {arh_models.keys()}')

        # Cached in Users/a19378208/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
        self.model = constructor(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.45, 0.45]))

    def forward(self, image, mask=None) -> dict:
        image = torch_normalize_image(image)
        outputs = self.model(image)
        if mask is not None:
            tensors = outputs['out']
            outputs['loss'] = self.criterion(tensors, mask)

        return outputs
