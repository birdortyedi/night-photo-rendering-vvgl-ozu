import seaborn as sns
import os
import modeling.bilateral_solver.bilateral_grid as bilateral_grid
import modeling.bilateral_solver.bilateral_solver as solver
import modeling.bilateral_solver.imresize as imresize
import numpy as np
import torch
from PIL import Image

sns.set_style('white')
sns.set_context('notebook')

grid_params = {
    'sigma_luma': 16,  # Brightness bandwidth
    'sigma_chroma': 8,  # Color bandwidth
    'sigma_spatial': 16  # Spatial bandwidth
}

bs_params = {
    'lam': 128,  # The strength of the smoothness parameter
    'A_diag_min': 1e-5,  # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
    'cg_maxiter': 25  # The number of PCG iterations
}


def im2double(im):
    """ Converts an uint image to floating-point format [0-1].

  Args:
    im: image (uint ndarray); supported input formats are: uint8 or uint16.

  Returns:
    input image in floating-point format [0-1].
  """

    if im[0].dtype == 'uint8' or im[0].dtype == 'int16':
        max_value = 255
    elif im[0].dtype == 'uint16' or im[0].dtype == 'int32':
        max_value = 65535
    return im.astype('float') / max_value


def imread(file, gray=False):
    image = Image.open(file)
    image = np.array(image)
    if not gray:
        image = image[:, :, :3]
    image = im2double(image)
    return image


def process_image(reference, target, confidence=None, tensor=False):
    if confidence is None:
        confidence = imread(os.path.join('modeling', 'bilateral_solver', 'confidence.png'), gray=True)

    if tensor:
        gpu = reference.is_cuda
        if gpu:
            reference = reference.cpu().data.numpy()
            target = target.cpu().data.numpy()
            reference = reference.transpose((1, 2, 0))
        else:
            reference = reference.data.numpy()
            target = target.data.numpy()
            reference = reference.transpose((1, 2, 0))

    im_shape = reference.shape[:2]
    assert (im_shape[0] == target.shape[0])
    assert (im_shape[1] == target.shape[1])

    confidence = imresize.imresize(confidence, output_shape=im_shape)

    assert (im_shape[0] == confidence.shape[0])
    assert (im_shape[1] == confidence.shape[1])

    grid = bilateral_grid.BilateralGrid(reference, **grid_params)

    t = target.reshape(-1, 1).astype(np.double)
    c = confidence.reshape(-1, 1).astype(np.double)
    output = solver.BilateralSolver(grid,
                                    bs_params).solve(t, c).reshape(im_shape)
    if tensor:
        if gpu:
            output = torch.from_numpy(output).to(
                device=torch.cuda.current_device())
        else:
            output = torch.from_numpy(output)

    return output
