"""Tools used by this project."""
import numpy as np


def normAll(image):
    """
    Normalise whole array to its mean and std.

    Args:
        image : {numpy} - [case x y] array

    Return:
        normalised array same as input
    """
    image = image.astype('float32')
    mean = np.mean(image)  # mean for data centering
    std = np.std(image)  # std for data normalization
    image -= mean
    image /= std
    return image


def normEach(image):
    """
    Normalise each slice to its mean and std array.

    Args:
        image : {numpy} - [case x y] array

    Return:
        normalised array same as input
    """
    image = image.astype('float32')
    mean = np.mean(image, axis=(1, 2))  # mean for each slice data centering
    std = np.std(image, axis=(1, 2))  # std for data normalization
    image -= mean[:, np.newaxis, np.newaxis]  # create array [case,1,1]
    image /= std[:, np.newaxis, np.newaxis]
    return image


def norm01(data):
    """Normalise to range 01."""
    mi = np.min(data)
    ret = data - mi
    ret /= np.max(ret)
    return ret


def asRgbImage(red, blue=None, green=None):
    """
    Convert planes RGB to image 8-bit.

    Any input is rescaled to 8-bit 0-255 range.

    """
    red8 = np.round(norm01(red) * 255)
    red8 = red8.astype('uint8')
    if blue is not None:
        blue8 = np.round(norm01(blue) * 255)
        blue8 = blue8.astype('uint8')
    else:
        blue8 = np.zeros(red.shape, dtype=np.uint8)
    if green is not None:
        green8 = np.round(norm01(green) * 255)
        green8 = green8.astype('uint8')
    else:
        green8 = np.zeros(red.shape, dtype=np.uint8)

    ret = np.empty(red.shape + (3,), dtype=np.uint8)
    ret[:, :, 0] = red8
    ret[:, :, 1] = green8
    ret[:, :, 2] = blue8
    return ret
