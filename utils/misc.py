import numpy as np
import torch
from modeling.DeepWB.utilities import imresize


def decode_cfa_pattern(cfa_pattern):
    cfa_dict = {0: 'B', 1: 'G', 2: 'R'}
    return "".join([cfa_dict[x] for x in cfa_pattern])


def to_tensor(im, dims=3):
    """ Converts a given ndarray image to torch tensor image.

  Args:
    im: ndarray image (height x width x channel x [sample]).
    dims: dimension number of the given image. If dims = 3, the image should
      be in (height x width x channel) format; while if dims = 4, the image
      should be in (height x width x channel x sample) format; default is 3.

  Returns:
    torch tensor in the format (channel x height x width)  or (sample x
      channel x height x width).
  """

    assert (dims == 3 or dims == 4)
    if dims == 3:
        im = im.transpose((2, 0, 1))
    elif dims == 4:
        im = im.transpose((0, 3, 1, 2))
    else:
        raise NotImplementedError

    return torch.from_numpy(im.copy())


def outOfGamutClipping(I, range=1.):
    """ Clips out-of-gamut pixels. """
    if range == 1.:
        I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
        I[I < 0] = 0  # any pixel is below 0, clip it to 0
    else:
        I[I > 255] = 255  # any pixel is higher than 255, clip it to 255
        I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def fractions2floats(fractions):
    floats = []
    for fraction in fractions:
        floats.append(float(fraction.numerator) / fraction.denominator)
    return floats


def gaussian(kernel_size, sigma):
    # calculate which number to where the grid should be
    # remember that, kernel_size[0] is the width of the kernel
    # and kernel_size[1] is the height of the kernel
    temp = np.floor(np.float32(kernel_size) / 2.)

    # create the grid
    # example: if kernel_size = [5, 3], then:
    # x: array([[-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.]])
    # y: array([[-1., -1., -1., -1., -1.],
    #           [ 0.,  0.,  0.,  0.,  0.],
    #           [ 1.,  1.,  1.,  1.,  1.]])
    x, y = np.meshgrid(np.linspace(-temp[0], temp[0], kernel_size[0]), np.linspace(-temp[1], temp[1], kernel_size[1]))

    # Gaussian equation
    temp = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    # make kernel sum equal to 1
    return temp / np.sum(temp)


def aspect_ratio_imresize(im, max_output=256):
    h, w, c = im.shape
    if max(h, w) > max_output:
        ratio = max_output / max(h, w)
        im = imresize.imresize(im, scalar_scale=ratio)
        h, w, c = im.shape

    if w % (2 ** 4) == 0:
        new_size_w = w
    else:
        new_size_w = w + (2 ** 4) - w % (2 ** 4)

    if h % (2 ** 4) == 0:
        new_size_h = h
    else:
        new_size_h = h + (2 ** 4) - h % (2 ** 4)

    new_size = (new_size_h, new_size_w)
    if not ((h, w) == new_size):
        im = imresize.imresize(im, output_shape=new_size)

    return im
