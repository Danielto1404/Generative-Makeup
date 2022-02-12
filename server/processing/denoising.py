import enum
from typing import Tuple, List

import cv2
import dlib
import matplotlib.pyplot as plt
from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS

from server.common import Point, BGR2RGB


class FaceSide(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'


def get_makeup_eye_box(
        landmarks: List[Tuple[int, int]],
        side: FaceSide,
        size: int = 512,
        x_padding: Tuple[int, int] = (0, 0),
        y_padding: Tuple[int, int] = (0, 0)
) -> Tuple[Point, Point]:
    eyebrow_min_index, eyebrow_max_index = FACIAL_LANDMARKS_68_IDXS[f'{side.value}_eyebrow']
    eye_min_index, eye_max_index = FACIAL_LANDMARKS_68_IDXS[f'{side.value}_eye']

    padding_left, padding_right = x_padding
    padding_top, padding_bottom = y_padding

    left, top, right, bottom = size, size, 0, 0
    for i, (x, y) in enumerate(landmarks):
        if i in range(eyebrow_min_index, eyebrow_max_index):
            top = max(
                min(top, y - padding_top),
                0
            )
            left = max(
                min(left, x - padding_left),
                0
            )
            right = min(
                max(right, x + padding_right),
                size
            )

        if i in range(eye_min_index, eye_max_index):
            bottom = min(
                max(bottom, y + padding_bottom),
                size
            )

    return Point(x=left, y=top), Point(x=right, y=bottom)


def detect_eyes(
        image,
        detector,
        predictor,
        size: int = 512,
        x_padding: Tuple[int, int] = (0, 0),
        y_padding: Tuple[int, int] = (0, 0)
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)

    left_eye_box = get_makeup_eye_box(
        landmarks=landmarks,
        side=FaceSide.LEFT,
        size=size,
        x_padding=x_padding,
        y_padding=y_padding
    )

    right_eye_box = get_makeup_eye_box(
        landmarks=landmarks,
        side=FaceSide.RIGHT,
        size=size,
        x_padding=(
            x_padding[1],
            x_padding[0]
        ),
        y_padding=y_padding)

    return left_eye_box, right_eye_box


def _denoise_one_eye(
        image,
        eye_box: Tuple[Point, Point]
):
    (left, top), (right, bottom) = eye_box
    eye_fragment = image[top:bottom, left:right]
    retouched_eye = cv2.fastNlMeansDenoisingColored(
        src=eye_fragment,
        dst=None,
        h=7,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=5
    )
    image[top:bottom, left:right] = retouched_eye
    return image


def denoise_eyes(
        image,
        detector,
        predictor,
        size: int = 512,
        x_padding: Tuple[int, int] = (10, 0),
        y_padding: Tuple[int, int] = (-20, 30)
):
    left_eye_box, right_eye_box = detect_eyes(
        image=image,
        detector=detector,
        predictor=predictor,
        size=size,
        x_padding=x_padding,
        y_padding=y_padding)

    image = _denoise_one_eye(image, left_eye_box)
    image = _denoise_one_eye(image, right_eye_box)

    return image


if __name__ == '__main__':
    img = cv2.imread('../../../../Downloads/generated/seed0241.png')
    plt.imsave("before.png", BGR2RGB(img))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../static/models/shape_predictor_68_face_landmarks.dat')
    img = denoise_eyes(image=img, detector=detector, predictor=predictor)
    img = BGR2RGB(img)
    plt.imsave('after.png', img)
