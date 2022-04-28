class CLIP:
    def __init__(self, gan_model, clip_model, device='cpu'):
        """
        :param clip_model:  CLIP model instance
        :param device: cpu | cuda
        """
        self.gan_model = gan_model.to(device)
        self.clip_model = clip_model.to(device)

    def manipulate(self):
        pass
