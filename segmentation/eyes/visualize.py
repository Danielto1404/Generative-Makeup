import argparse
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from PIL import Image

from utils import torch_normalize_image


def show_image_segmentation(model, image: torch.Tensor):
    np_img = image.cpu().numpy().transpose((1, 2, 0)) / 255

    with torch.no_grad():
        image = torch_normalize_image(image)
        mask = model(image.unsqueeze(0))['out'][0].argmax(0).cpu().numpy()

    plt.figure(figsize=(10, 20))
    f, axes = plt.subplots(2)
    axes[0].imshow(np_img)
    axes[1].imshow(mask)
    plt.show()


def show_random_image(root: str, model, device):
    file_path = np.random.choice(os.listdir(root))
    image = Image.open(os.path.join(root, file_path))
    image = numpy.asarray(image).transpose((2, 0, 1))
    image = torch.tensor(image).float().to(device)
    show_image_segmentation(model, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', help='path to images folder')
    parser.add_argument('--load_path', help='model checkpoint path')
    parser.add_argument('--device', help='torch device')
    args = parser.parse_args()

    segm_model = torch.load(args.load_path, map_location=args.device).eval()
    show_random_image(root=args.images_root, model=segm_model, device=args.device)
