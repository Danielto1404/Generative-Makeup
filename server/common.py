from collections import namedtuple

import cv2

Point = namedtuple("Point", ['x', 'y'])


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
