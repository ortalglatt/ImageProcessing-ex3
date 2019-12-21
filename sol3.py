import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from skimage.util import img_as_bool
from scipy.ndimage.filters import convolve
from scipy import signal
import matplotlib.pyplot as plt
import os


RGB_SHAPE = 3
GRAY_REP = 1
NORM_FACTOR = 255
FILTER = np.array([[1, 1]])
MIN_SIZE = 16
EXTERNALS_DIR = "externals"
EX1_IN1 = "coffee.jpg"
EX1_IN2 = "beach.jpg"
EX1_MASK = "mask1.jpg"
EX2_IN1 = "stars.jpg"
EX2_IN2 = "train.jpg"
EX2_MASK = "mask2.jpg"


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation.
    :param filename: The image filename.
    :param representation: Representation code - (1) gray scale image (2) RGB image.
    :return: The converted image normalized to the range [0, 1].
    """
    im = imread(filename)
    if len(im.shape) == RGB_SHAPE and representation == GRAY_REP:
        im = rgb2gray(im)
        return im.astype(np.float64)
    im_float = im.astype(np.float64)
    im_float /= NORM_FACTOR
    return im_float


def filter_vector(filter_size):
    """
    Creates the filter vector.
    :param filter_size: The size of the filter vector to create.
    :return: The filter vector that was created.
    """
    filter_vec = FILTER
    while filter_vec.shape[1] < filter_size:
        filter_vec = signal.convolve(filter_vec, FILTER)
    filter_vec = filter_vec / np.sum(filter_vec)
    return filter_vec


def reduce(im, filter_vec):
    """
    Reduce the size of the image by blurring with the filter_vec.
    :param im: The image matrix.
    :param filter_vec: The 1D filter vector.
    :return: The reduced size image.
    """
    fil = signal.convolve2d(filter_vec, np.transpose(filter_vec))
    res = convolve(im, fil, mode='constant')
    return res[::2, ::2]


def expand(im, filter_vec):
    """
    Expand the size of the image by blurring with the filter_vec.
    :param im: The image matrix.
    :param filter_vec: The 1D filter vector.
    :return: The expanded size image.
    """
    res = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    res[::2, ::2] = im
    fil = filter_vec * 2
    fil = signal.convolve2d(fil, np.transpose(fil))
    return convolve(res, fil, mode='constant')


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Create gaussian pyramid for the given image by using a filter vector with the given size.
    :param im: The image to create pyramid for.
    :param max_levels: The maximum size of the pyramid.
    :param filter_size: The filter vector size.
    :return: The gaussian pyramid that was created and the filter vector.
    """
    cur_im = im
    pyr = []
    filter_vec = filter_vector(filter_size)
    for i in range(max_levels):
        pyr.append(cur_im)
        if cur_im.shape[0] / 2 < MIN_SIZE or cur_im.shape[1] / 2 < MIN_SIZE:
            break
        cur_im = reduce(cur_im, filter_vec)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Create laplacian pyramid for the given image by using a filter vector with the given size.
    :param im: The image to create pyramid for.
    :param max_levels: The maximum size of the pyramid.
    :param filter_size: The filter vector size.
    :return: The laplacian pyramid that was created and the filter vector.
    """
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(gauss_pyr) - 1):
        exp = expand(gauss_pyr[i + 1], filter_vec)
        pyr.append(gauss_pyr[i] - exp)
    pyr.append(gauss_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Convert a laplacian pyramid to an image that close to the original image.
    :param lpyr: The laplacian pyramid of the image.
    :param filter_vec: The filter vector.
    :param coeff: A vector with the same length as the pyramid, to multiply the levels of the
    pyramid with.
    :return: The reconstructed image.
    """
    res = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        res = expand(res, filter_vec)
        res += lpyr[i] * coeff[i]
    return res


def render_pyramid(pyr, levels):
    """
    Create an image that contains the given levels number of the pyramid.
    :param pyr: The pyramid to put in the result image.
    :param levels: Number of levels to put in the result from the pyramid.
    :return: The image that contains the pyramid levels.
    """
    height = pyr[0].shape[0]
    width = 0
    for i in range(levels):
        width += pyr[i].shape[0]
    res = np.zeros((height, width))
    start = 0
    for i in range(levels):
        res[0:height, start:start + height] = (pyr[i] + 1) / 2
        start += height
        height = int(height / 2)
    return res


def display_pyramid(pyr, levels):
    """
    Display an image that contains the given levels number of the pyramid.
    :param pyr: The pyramid to display.
    :param levels: Number of levels to display
    """
    im = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Blend to images by using the given mask and 2 filters - one foe the mask and one for the
    images.
    :param im1: First image to blend.
    :param im2: Second image to blend.
    :param mask: Mask to blend the two images with.
    :param max_levels: Maximum levels of the pyramids that needed to be created for the blending.
    :param filter_size_im: The size of the images filter vector.
    :param filter_size_mask: The size of the mask filer vector.
    :return: The blended image matrix.
    """
    lap1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap2, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask = mask.astype(dtype=np.float64)
    mask_g, filter_mask = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    new_lap = []
    for i in range(len(lap1)):
        new_lap.append(mask_g[i] * lap1[i] + (1 - mask_g[i]) * lap2[i])
    return np.clip(laplacian_to_image(new_lap, filter_vec, [1] * len(lap1)), 0, 1)


def relpath(filename):
    """
    :param filename: Filename to find its relative path.
    :return: The relative path of the given file.
    """
    return os.path.join(os.path.dirname(__file__), filename)


def color_im_blend(im_file1, im_file2, mask_file, max_levels, filter_size_im, filter_size_mask):
    """
    Blends 2 RGB images with the given mask, and display the results.
    :param im_file1: First image file to blend.
    :param im_file2: Second image file to blend.
    :param mask_file: Mask file to blend the two images with.
    :param max_levels: Maximum levels of the pyramids that needed to be created for the blending.
    :param filter_size_im: The size of the images filter vector.
    :param filter_size_mask: The size of the mask filer vector.
    :return: The first image matrix, second image matrix, the mask matrix and the blended image
    matrix.
    """
    im1 = read_image(relpath(os.path.join(EXTERNALS_DIR, im_file1)), 2)
    im2 = read_image(relpath(os.path.join(EXTERNALS_DIR, im_file2)), 2)
    mask = read_image(relpath(os.path.join(EXTERNALS_DIR, mask_file)), 1)
    res = np.zeros((im1.shape[0], im1.shape[1], 3))
    for i in range(3):
        res[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels,
                                        filter_size_im, filter_size_mask)
    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(im1)
    axs[0, 1].imshow(im2)
    axs[1, 0].imshow(mask, cmap=plt.cm.gray)
    axs[1, 1].imshow(res)
    plt.show()
    return im1, im2, img_as_bool(mask), res


def blending_example1():
    """
    First example for image blending.
    """
    return color_im_blend(EX1_IN1, EX1_IN2, EX1_MASK, 4, 5, 5)


def blending_example2():
    """
    Second example for image blending.
    """
    return color_im_blend(EX2_IN1, EX2_IN2, EX2_MASK, 7, 11, 11)
