import pickle
import os
import numpy as np
import skimage.external.tifffile as tiff
import matplotlib.pyplot as plt
import imageio
import Functions.init_params as init_params
params = init_params.get_params()
plt.interactive(False)


def load_model(path):
    dir_contents = os.listdir(path)

    if not dir_contents:
        return None

    model_file = open(path + '/' + str(dir_contents[0]), 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model


def read_folder(path):
    scene_content = os.listdir(path)[0]
    scene_path = path + '/' + str(scene_content)

    return scene_content, scene_path


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_exposure(path):
    txt_matcher = __get_specific_contents__(path, '.txt')
    if not txt_matcher:
        return None
    exposure_file = open(path + '/' + txt_matcher[0], 'r')
    exposures = [2**float(line) for line in exposure_file]
    exposure_file.close()
    return exposures


def read_images(path):
    tif_matcher = __get_specific_contents__(path, '.tif')
    input_images = []
    for image in tif_matcher:
        im = np.asarray(tiff.imread(path + '/' + image)) / 65535  # TODO: Cross check for 16-bit images
        im = __clamp__(im, 0, 1)
        input_images.append(im)
    images = np.asarray(input_images)

    hdr_matcher = __get_specific_contents__(path, '.hdr')
    if hdr_matcher:
        # TODO
        pass

    # class type is imageio.core.util.Image
    # Already in single precision
    hdr = imageio.imread(path + '/' + hdr_matcher[0], format='HDR-FI')
    hdr = __clamp__(hdr, 0, 1)  # TODO: required for a .hdr image?

    return images, hdr


def __get_specific_contents__(path, extension):
    folder_contents = os.listdir(path)
    return [file for file in folder_contents if extension in file]


def __clamp__(image, a, b):
    out = image
    out[out < a] = a
    out[out > b] = b
    return out


def __tone_map__(hdr_image):
    return np.log(1 + params.mu * hdr_image) / np.log(1 + params.mu)