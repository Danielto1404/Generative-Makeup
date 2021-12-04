class Polygon:
    def __init__(self, vertices: [float], class_label=0):
        pass


def polygon2pix(image_size: (int, int), polygons: [Polygon]):
    """
    :param image_size: image size (width, height)
    :param polygons:   list of `Polygon` objects

    :return: tensor mask (weight, height), where each pixel have it's own class according to Polygon for this region
    """
    pass
