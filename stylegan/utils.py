import pickle

import torch


def load_gan(pickle_path, device='cpu'):
    """
    :param pickle_path: path to torch stylegan model
    :param device: torch device
    :return: Generator
    """
    with open(pickle_path, 'rb') as fp:
        state = pickle.load(fp)
        generator = state['G_ema']
        print(generator)
        return generator.to(device)


def generate_image(G, latent=None, class_label=None, device='cpu'):
    if latent is None:
        latent = torch.randn([1, G.z_dim]).to(device)
        
    img = G(latent, class_label)
    return img
