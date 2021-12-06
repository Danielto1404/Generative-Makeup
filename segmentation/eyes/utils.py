from torchvision.transforms import transforms

torch_normalize_image = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def visualize_mask(source_image, segmentation_mask):
    pass