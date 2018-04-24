import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def read_images(path):
    images_list = os.listdir(path)
    images = []

    for image in images_list:
        images.append(np.asarray(Image.open(path + image)))

    return images


PATH = '/home/prch7562/Documents/Deep-HDR-Dynamic-Imaging/flow_images/'
images = read_images(PATH)
plt.imshow(images[0])
plt.show()
