import torch
from torch import nn


def WeightedCE(mask: torch.Tensor):
    # _, counts = torch.unique(mask, return_counts=True)
    # n = mask.numel()
    # 3, 5, 6 = n
    #
    return nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.45, 0.45]))
    # return 1 - counts / n
    # return torch.unique(mask, return_count /s=True)
    # return nn.CrossEntropyLoss()


test = torch.tensor([[-1, 1, 2], [0, 2, 3]])

print(WeightedCE(test))
