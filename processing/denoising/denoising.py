import enum
from typing import Tuple, List

import cv2
from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS

from processing.utils import Point, sized_box


class FaceSide(enum.Enum):
    """
    Helper enum for defining face size.
    It can be either left or right.
    """
    LEFT = 'left'
    RIGHT = 'right'


def _get_makeup_eye_box(
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

    left, top, right, bottom = sized_box(size)

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
    """
    Detects eyes bounding boxes with given paddings for x and y axes.

    :param image: face image
    :param detector: dlib frontal face detector
    :param predictor: dlib 68 landmarks face predictor
    :param size: image size
    :param x_padding: x-axis padding for eyes
    :param y_padding: y-axis padding for eyes
    :return: tuple of bounding boxes for left and right eye
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)

    left_eye_box = _get_makeup_eye_box(
        landmarks=landmarks,
        side=FaceSide.LEFT,
        size=size,
        x_padding=x_padding,
        y_padding=y_padding
    )

    right_eye_box = _get_makeup_eye_box(
        landmarks=landmarks,
        side=FaceSide.RIGHT,
        size=size,
        x_padding=(
            x_padding[1],
            x_padding[0]
        ),
        y_padding=y_padding
    )

    return left_eye_box, right_eye_box


def _denoise_one_eye(
        source,
        target,
        eye_box: Tuple[Point, Point]
):
    (left, top), (right, bottom) = eye_box
    eye_fragment = source[top:bottom, left:right]

    retouched_eye = cv2.fastNlMeansDenoisingColored(
        src=eye_fragment,
        dst=None,
        h=7,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=5
    )

    target[top:bottom, left:right] = retouched_eye
    return target


def _denoise_face(
        image,
):
    """
    Applies rough denoising for the whole image.

    :param image: source face for denoising
    :return: denoised rough face
    """
    retouched_image = cv2.fastNlMeansDenoisingColored(
        src=image,
        dst=None,
        h=6,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=5
    )

    return retouched_image


def denoise_face(
        image,
        detector,
        predictor,
        size: int = 512,
        eyes_x_padding: Tuple[int, int] = (10, 0),
        eyes_y_padding: Tuple[int, int] = (-20, 30)
):
    """
    Apply denoise operations for artefacts removing after GAN generation.
    Uses fastNlMeansDenoisingColored method from cv2 library with predefined parameters.

    Apply denoising in 3 steps:
        1. Denoise whole face with eyes
        2. Denoise eyes each eye individually
        3. Overlap individually denoised eyes onto denoised face

    :param image: source
    :param detector: dlib face frontal face detector
    :param predictor: dlib 68 face landmarks face predictor
    :param size: image size
    :param eyes_x_padding: eyes padding for x-axis
    :param eyes_y_padding: eyes padding for y-axis

    :return: denoised face
    """
    left_eye_box, right_eye_box = detect_eyes(
        image=image,
        detector=detector,
        predictor=predictor,
        size=size,
        x_padding=eyes_x_padding,
        y_padding=eyes_y_padding
    )

    retouched_image = _denoise_face(image)
    retouched_image = _denoise_one_eye(source=image, target=retouched_image, eye_box=left_eye_box)
    retouched_image = _denoise_one_eye(source=image, target=retouched_image, eye_box=right_eye_box)

    return retouched_image
