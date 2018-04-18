import numpy as np
import matplotlib.pyplot as plt

import Functions.init_params as init_params
from Functions.file_io import clamp
params = init_params.get_params()


def prepare_input_features(ldr_images, exposures, label):
    num_im = len(ldr_images)

    # TODO: return the values from optical flow
    compute_optical_flow(ldr_images, exposures)


def compute_optical_flow(ldr_images, expo):
    warped = ldr_images

    exposure_adjusted_1 = adjust_exposure(ldr_images[0:2], expo[0:2])
    exposure_adjusted_2 = adjust_exposure(ldr_images[1:3], expo[1:3])


def adjust_exposure(images, exposures):
    num_images = len(images)
    num_exposures = len(exposures)

    assert num_images == num_exposures, \
        'Number of images for adjusting exposure is not equal to the number of exposures'
    max_exposure = max(exposures)

    adjusted = np.asarray([ldr_to_ldr(images[i], exposures[i], max_exposure) for i in range(num_images)])

    return adjusted


def ldr_to_ldr(image, exposure_one, exposure_two):

    radiance = ldr_to_hdr(image, exposure_one)
    b = hdr_to_ldr(radiance, exposure_two)

    return b


def ldr_to_hdr(input_image, exposure):
    input_image = clamp(input_image, 0, 1)

    output_image = input_image ** params.gamma
    output_image = output_image / exposure

    return output_image


def hdr_to_ldr(input_image, exposure):
    # TODO: cross-check if pixels are in single precision
    input_image = input_image * exposure
    input_image = clamp(input_image, 0, 1)
    output_image = input_image ** (1/params.gamma)

    return output_image
