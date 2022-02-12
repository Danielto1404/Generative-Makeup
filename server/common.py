from collections import namedtuple

import cv2

Point = namedtuple("Point", ["x", "y"])
Box = namedtuple("Box", ["left", "top", "right", "bottom"])


def BGR2RGB(img):
    """
    Converts image from BGR format to RGB
    :param img: source image in BGR format
    :return: copy of source image in RGB format
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sized_box(size: int):
    """
    Returns box which start at (0, 0) and ends in (size, size)
    :param size: box size
    """
    return Box(left=0, top=0, right=size, bottom=size)
