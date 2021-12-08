import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, deeplabv3_resnet101

arh_models = {
    'deeplabv3_resnet50': deeplabv3_resnet50,
    'deeplabv3_resnet101': deeplabv3_resnet101,
    'fcn_resnet50': fcn_resnet50,
}


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str):
        super().__init__()
        self.num_classes = num_classes
        constructor = arh_models.get(model_name)
        if constructor is None:
            raise NotImplemented(f'Model {model_name} not found, all models: {arh_models.keys()}')

        # Cached in Users/a19378208/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
        self.model = constructor(num_classes=num_classes)

    def forward(self, image, mask=None) -> dict:
        outputs = self.model(image)
        if mask is not None:
            criterion = nn.CrossEntropyLoss()
            tensors = outputs['out']
            outputs['loss'] = criterion(tensors, mask)

        return outputs
