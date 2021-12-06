import argparse
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from PIL import Image


def show_image_segmentation(model, image: torch.Tensor):
    np_img = image.numpy().transpose((1, 2, 0))
    plt.figure()

    with torch.no_grad():
        mask = model(image.unsqueeze(0))['out'][0].argmax(0)

    # subplot(r,c) provide the no. of rows and columns
    f, axes = plt.subplots(2)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axes[0].imshow(np_img)
    axes[1].imshow(mask)


def show_random_image(root: str, model):
    file_path = np.random.choice(os.listdir(root))
    image = Image.open(os.path.join(root, file_path))
    image = numpy.asarray(image).transpose((2, 0, 1))
    image = torch.tensor(image).float().to(model.device)
    show_image_segmentation(model, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', help='path to images folder')
    parser.add_argument('--load_path', help='model checkpoint path')
    parser.add_argument('--device', help='torch device')
    args = parser.parse_args()

    model = torch.load(args.load_path, map_location=args.device)
    show_random_image(args.images_root, model)
