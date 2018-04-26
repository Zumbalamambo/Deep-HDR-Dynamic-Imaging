import numpy as np
import cv2
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.misc import imread

from Functions.file_io import get_specific_contents
import Functions.init_params as init_params
from Functions.file_io import clamp
params = init_params.get_params()


def prepare_input_features(ldr_images, exposures, label):
    num_im = len(ldr_images)

    # TODO: compute_optical_flow : get flow vectors from Cython module
    # compute_warping_offline(ldr_images, exposures)


def prepare_input_features_offline(ldr_images, exposures, scenePath):
    warped_scenes_list = sorted(get_specific_contents(scenePath, 'warped'))

    # Reading the optical flow adjusted images offline
    input_ldr = np.asarray([imread(scenePath + '/' + warped_scenes_list[0]) / 255.0,
                            ldr_images[1],
                            imread(scenePath + '/' + warped_scenes_list[1]) / 255.0])

    # concatenating inputs in ldr and hdr domain
    inputs = np.concatenate((input_ldr[0],
                             input_ldr[1],
                             input_ldr[2]), axis=2)
    inputs = np.concatenate((inputs,
                             ldr_to_hdr(input_ldr[0], exposures[0]),
                             ldr_to_hdr(input_ldr[1], exposures[1]),
                             ldr_to_hdr(input_ldr[2], exposures[2])), axis=2)

    return inputs


def compute_optical_flow(ldr_images, expo, scenePath):
    warped = []
    exposure_adjusted_1 = adjust_exposure(ldr_images[0:2], expo[0:2])
    exposure_adjusted_2 = adjust_exposure(ldr_images[1:3], expo[1:3])

    # TODO: Find way to call Cython module for get optical flow vectors
    # TODO: For now, saving images and evaluating flow offline
    save_exposure_adjusted(exposure_adjusted_1, exposure_adjusted_2)


def compute_warping_offline(ldr_images, expo, scenePath):
    flowvectors = sorted(get_specific_contents(scenePath, 'npy'))
    flow_1 = np.load(scenePath + '/' + flowvectors[0])
    flow_2 = np.load(scenePath + '/' + flowvectors[1])

    warped1 = warp_flow_cv2(ldr_images[0], flow_1)


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


def find_flow(ref, img):

    flow = cv2.calcOpticalFlowFarneback(ref, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # u, v, im2W = pyflow.coarse2fine_flow(ref, img, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
    # print('Time Taken: %.2f seconds for image of size (%d, %d)' % (
    # 'ref.shape[0], ref.shape[1], ref.shape[2]))
    # flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    print(flow.dtype)

    return flow


def optical_flow(one, two):
    one = one.astype(np.uint8)
    two = two.astype(np.uint8)
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((1000, 1500, 3))
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:, :, 1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow


def warp_flow_cv2(img, flow):
    hsv = np.zeros(img.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb


def warp_flow(img, flow):
    h, w, c = img.shape
    hf, wf, ch = flow.shape
    hd = (h - hf) / 2
    wd = (w - wf) / 2
    warped = np.zeros((hf, wf, c))

    X, Y = np.meshgrid(np.arange(wf), np.arange(hf))
    xi, yi = np.meshgrid(np.arange(wf+wd), np.arange(hf+hd))

    for i in range(c):
        f = interp2d(X, Y, img[:, :, i], kind='cubic')
        curX = X + flow[:, :, 0]
        curY = Y + flow[:, :, 1]

        # warped[:, :, i] = interp2d(curX, curY, img[:, :, i], kind='cubic')
        warped[:, :, i] = f(curX, curY)

    return warped


def save_exposure_adjusted(expo1, expo2, path):
    from scipy.misc import imsave

    imsave('Man-Standing-1.jpg', expo1[0])
    imsave('Man-Standing-2.1.jpg', expo1[1])
    imsave('Man-Standing-2.2.jpg', expo2[0])
    imsave('Man-Standing-3.jpg', expo2[1])