import numpy
import torch
from PIL import Image
from torchvision.transforms import transforms

tt = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def eval(image_path):
    model = torch.load('checkpoints/model.pth').eval()
    image = Image.open(image_path)
    image = numpy.asarray(image).transpose((2, 0, 1))
    image = torch.tensor(image).unsqueeze(0).float()
    image = tt(image)
    with torch.no_grad():
        outputs = model(image)
        tensors = outputs['out']
        result_ = tensors[0].argmax(0)

#     plt.imshow(result_)
#     plt.imshow(image[0].transpose(0, 1).transpose(1, 2))
#     plt.show()
#
#
# eval('/Users/a19378208/Documents/GitHub/Generative-Makeup/data/images/0d384dbbcc121ca5049c423f81c26e6a.png')
