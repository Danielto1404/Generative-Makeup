import os
import shutil
from math import sqrt
from os.path import join

import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

os.environ["PATH"] += os.pathsep + os.getcwd()


def getBwLittleImgs(datasetPath):
    # Find all classes paths in directory and iterate over it

    bwDir = "bwdir"
    if not os.path.exists(bwDir):
        os.makedirs(bwDir)
    else:
        shutil.rmtree(bwDir)
        os.makedirs(bwDir)

    for (i, imgName) in enumerate(os.listdir(datasetPath)):
        # Construct patch to single image
        imgPath = join(datasetPath, imgName)
        # Read image using OpenCV as grayscale
        image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

        # Check if we opened an image.
        if image is not None:
            # Resize opened image
            resized_image = cv2.resize(image, (32, 32))
            resized_image = np.array(resized_image)
            cv2.imwrite(os.path.join(bwDir, imgName), resized_image)
        else:
            print(imgPath)
            os.remove(imgPath)

        return bwDir


def findDelDuplBw(searchedName, bwDir):
    # Join path to orginal image that we are looking duplicates
    searchedImg = join(bwDir, searchedName)

    # Start iterate over all bw images
    for (j, cmpImageName) in enumerate(os.listdir(bwDir)):

        if cmpImageName == searchedName:
            # If name in bwDir is equal to searched image - pass. I don't wan to deletde searched image in bw dir
            pass
        else:
            # If name is different - concatenate path to image
            cmpImageBw = join(bwDir, cmpImageName)

            try:
                # Open image in bwDir - The searched image
                searchedImageBw = np.array(cv2.imread(searchedImg, cv2.IMREAD_GRAYSCALE))
                # Open image to be compared
                cmpImage = np.array(cv2.imread(cmpImageBw, cv2.IMREAD_GRAYSCALE))
                # Count root mean square between both images (RMS)
                rms = sqrt(mean_squared_error(searchedImageBw, cmpImage))
            except Exception as e:
                print(e)
                continue

            print(rms)
            # If RMS is smaller than 3 - this means that images are simmilar or the same
            if rms > 0.00005:
                # Delete compared image in BW dir
                os.remove(cmpImageBw)
                print(searchedImg, cmpImageName, rms)


def main():
    datasetPath = "/Users/daniilkorolev/Downloads/instagram-crop"

    bwDir = getBwLittleImgs(datasetPath)

    for (i, imgName) in tqdm(enumerate(os.listdir(datasetPath))):

        detectedImg = join(datasetPath, imgName)
        findDelDuplBw(detectedImg, bwDir)

    shutil.rmtree(bwDir)


if __name__ == "__main__":
    main()
