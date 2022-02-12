from collections import OrderedDict

FACIAL_LANDMARKS_68_INDICES = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def check_landmark(key, index):
    (start, end) = FACIAL_LANDMARKS_68_INDICES.get(key, (-1, -1))
    return index in range(start, end)
