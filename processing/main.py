import cv2
import dlib
import matplotlib.pyplot as plt

from processing.denoising.denoising import denoise_face
from utils import BGR2RGB

if __name__ == '__main__':

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor('../static/models/shape_predictor_68_face_landmarks.dat')

    seed = 2010
    source = cv2.imread(f"../../../Downloads/sg3-samples/seed{seed}.png")
    target = denoise_face(source, detector=detector, predictor=predictor)

    _, axs = plt.subplots(1, 2, figsize=(30, 15))
    for ax in axs:
        ax.axis("off")

    axs[0].imshow(BGR2RGB(source))
    axs[1].imshow(BGR2RGB(target))
    plt.plot()
    plt.savefig("denoise_result")
