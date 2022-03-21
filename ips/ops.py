import cv2
import math
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
from scipy import signal
from kornia.geometry.transform import resize
from fractions import Fraction
from exifread.utils import Ratio

from modeling.DeepWB.arch import deep_wb_single_task as dwb
from modeling.DeepWB.utilities.deepWB import deep_wb
from modeling.DeepWB.utilities.utils import colorTempInterpolate_w_target
from modeling import weight_refinement
from utils import color, optim, tone, misc


def linearize_raw(raw_img, img_meta):
    return raw_img


def normalize(linearized_raw, img_meta):
    return normalize_(linearized_raw, img_meta['black_level'], img_meta['white_level'])


def normalize_(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = misc.ratios2floats(black_level)
        if type(black_level[0]) is Fraction:
            black_level = misc.fractions2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    normalized_image[normalized_image > 1] = 1
    return normalized_image


def bad_pixel_correction(data, neighborhood_size=3):
    if (neighborhood_size % 2) == 0:
        print("neighborhood_size shoud be odd number, recommended value 3")
        return data

    # convert to float32 in case they were not
    # Being consistent in data format to be float32
    data = np.float32(data)

    # Separate out the quarter resolution images
    D = {0: data[::2, ::2], 1: data[::2, 1::2], 2: data[1::2, ::2], 3: data[1::2, 1::2]}  # Empty dictionary

    # number of pixels to be padded at the borders
    no_of_pixel_pad = math.floor(neighborhood_size / 2.)

    for idx in range(0, len(D)):  # perform same operation for each quarter
        img = D[idx]
        height, width = img.shape

        # pad pixels at the borders
        img = np.pad(img, (no_of_pixel_pad, no_of_pixel_pad), 'reflect')  # reflect would not repeat the border value

        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # save the middle pixel value
                mid_pixel_val = img[i, j]

                # extract the neighborhood
                neighborhood = img[i - no_of_pixel_pad: i + no_of_pixel_pad + 1, j - no_of_pixel_pad: j + no_of_pixel_pad + 1]

                # set the center pixels value same as the left pixel
                # Does not matter replace with right or left pixel
                # is used to replace the center pixels value
                neighborhood[no_of_pixel_pad, no_of_pixel_pad] = neighborhood[no_of_pixel_pad, no_of_pixel_pad - 1]

                min_neighborhood = np.min(neighborhood)
                max_neighborhood = np.max(neighborhood)

                if mid_pixel_val < min_neighborhood:
                    img[i, j] = min_neighborhood
                elif mid_pixel_val > max_neighborhood:
                    img[i, j] = max_neighborhood
                else:
                    img[i, j] = mid_pixel_val

        # Put the corrected image to the dictionary
        D[idx] = img[no_of_pixel_pad: height + no_of_pixel_pad, no_of_pixel_pad: width + no_of_pixel_pad]

    # Regrouping the data
    data[::2, ::2] = D[0]
    data[::2, 1::2] = D[1]
    data[1::2, ::2] = D[2]
    data[1::2, 1::2] = D[3]

    return data


def apply_color_space_transform(demosaiced_image, color_matrix_1, color_matrix_2):
    if isinstance(color_matrix_1[0], Fraction):
        color_matrix_1 = misc.fractions2floats(color_matrix_1)
    if isinstance(color_matrix_2[0], Fraction):
        color_matrix_2 = misc.fractions2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # inverse
    cam2xyz1 = np.linalg.inv(xyz2cam1)
    cam2xyz2 = np.linalg.inv(xyz2cam2)
    # for now, use one matrix  # TODO: interpolate btween both
    # simplified matrix multiplication
    xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * \
        demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])

    # xyz2srgb = np.linalg.inv(srgb2xyz)

    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    # normalize rows (needed?)
    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis,
                          :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def white_balance_w_style(demosaic, net, wb_settings, multi_scale=True, post_process=True, device="cpu"):
    size = demosaic.shape[:2]

    deepWB_T = dwb.deepWBnet()
    deepWB_T.load_state_dict(torch.load('DeepWB/models/net_t.pth'))
    deepWB_S = dwb.deepWBnet()
    deepWB_S.load_state_dict(torch.load('DeepWB/models/net_s.pth'))
    deepWB_T.eval().to(device)
    deepWB_S.eval().to(device)

    t_img, s_img = deep_wb(demosaic, task='editing', net_s=deepWB_S, net_t=deepWB_T, device=device)
    full_size_img = demosaic.copy()
    d_img = misc.aspect_ratio_imresize(demosaic, max_output=384)
    t_img = misc.aspect_ratio_imresize(t_img, max_output=384)
    s_img = misc.aspect_ratio_imresize(s_img, max_output=384)
    s_mapping = optim.get_mapping_func(d_img, s_img)
    t_mapping = optim.get_mapping_func(d_img, t_img)
    full_size_s = optim.apply_mapping_func(full_size_img, s_mapping)
    full_size_s = misc.outOfGamutClipping(full_size_s)
    full_size_t = optim.apply_mapping_func(full_size_img, t_mapping)
    full_size_t = misc.outOfGamutClipping(full_size_t)
    if 'F' in wb_settings:
        f_img = colorTempInterpolate_w_target(t_img, s_img, 3800)
        f_mapping = optim.get_mapping_func(d_img, f_img)
        full_size_f = optim.apply_mapping_func(full_size_img, f_mapping)
        full_size_f = misc.outOfGamutClipping(full_size_f)
    else:
        f_img = None

    if 'C' in wb_settings:
        c_img = colorTempInterpolate_w_target(t_img, s_img, 6500)
        c_mapping = optim.get_mapping_func(d_img, c_img)
        full_size_c = optim.apply_mapping_func(full_size_img, c_mapping)
        full_size_c = misc.outOfGamutClipping(full_size_c)
    else:
        c_img = None
    d_img = misc.to_tensor(d_img, dims=3)
    s_img = misc.to_tensor(s_img, dims=3)
    t_img = misc.to_tensor(t_img, dims=3)
    if f_img is not None:
        f_img = misc.to_tensor(f_img, dims=3)
    if c_img is not None:
        c_img = misc.to_tensor(c_img, dims=3)
    img = torch.cat((d_img, s_img, t_img), dim=0)
    if f_img is not None:
        img = torch.cat((img, f_img), dim=0)
    if c_img is not None:
        img = torch.cat((img, c_img), dim=0)
    full_size_img = misc.to_tensor(full_size_img, dims=3)
    full_size_s = misc.to_tensor(full_size_s, dims=3)
    full_size_t = misc.to_tensor(full_size_t, dims=3)
    if f_img is not None:
        full_size_f = misc.to_tensor(full_size_f, dims=3)
    if c_img is not None:
        full_size_c = misc.to_tensor(full_size_c, dims=3)
    imgs = [full_size_img.unsqueeze(0).to(device, dtype=torch.float32),
            full_size_s.unsqueeze(0).to(device, dtype=torch.float32),
            full_size_t.unsqueeze(0).to(device, dtype=torch.float32)]
    if c_img is not None:
        imgs.append(full_size_c.unsqueeze(0).to(device, dtype=torch.float32))
    if f_img is not None:
        imgs.append(full_size_f.unsqueeze(0).to(device, dtype=torch.float32))
    img = img.unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        _, weights = net(img)
        if multi_scale:
            img_1 = resize(img, size=(int(0.5 * img.shape[2]), int(0.5 * img.shape[3])), interpolation='bilinear', align_corners=True)
            _, weights_1 = net(img_1)
            weights_1 = resize(weights_1, size=(img.shape[2], img.shape[3]), interpolation='bilinear', align_corners=True)

            img_2 = resize(img, size=(int(0.25 * img.shape[2]), int(0.25 * img.shape[3])), interpolation='bilinear', align_corners=True)
            _, weights_2 = net(img_2)
            weights_2 = resize(weights_2, size=(img.shape[2], img.shape[3]), interpolation='bilinear', align_corners=True)

            weights = (weights + weights_1 + weights_2) / 3
    weights = resize(weights, size=size, interpolation='bilinear', align_corners=True)

    if post_process:
        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                ref = imgs[0][j, :, :, :]
                curr_weight = weights[j, i, :, :]
                refined_weight = weight_refinement.process_image(ref, curr_weight, tensor=True)
                weights[j, i, :, :] = refined_weight
                weights = weights / torch.sum(weights, dim=1)

    for i in range(weights.shape[1]):
        if i == 0:
            out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
        else:
            out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
    out_img_np = out_img.squeeze().permute(1, 2, 0).cpu().numpy()
    return misc.outOfGamutClipping(out_img_np)


def memory_color_enhancement(data, color_space="srgb", illuminant="d65", clip_range=[0, 255], cie_version="1931"):
    target_hue = [30., -125., 100.]
    hue_preference = [20., -118., 130.]
    hue_sigma = [20., 10., 5.]
    is_both_side = [True, False, False]
    multiplier = [0.6, 0.6, 0.6]
    chroma_preference = [25., 14., 30.]
    chroma_sigma = [10., 10., 5.]

    # RGB to xyz
    data = color.rgb2xyz(data, color_space, clip_range)
    # xyz to lab
    data = color.xyz2lab(data, cie_version, illuminant)
    # lab to lch
    data = color.lab2lch(data)

    # hue squeezing
    # we are traversing through different color preferences
    height, width, _ = data.shape
    hue_correction = np.zeros((height, width), dtype=np.float32)
    for i in range(0, np.size(target_hue)):

        delta_hue = data[:, :, 2] - hue_preference[i]

        if is_both_side[i]:
            weight_temp = np.exp(-np.power(data[:, :, 2] - target_hue[i], 2) / (2 * hue_sigma[i] ** 2)) + \
                          np.exp(-np.power(data[:, :, 2] + target_hue[i], 2) / (2 * hue_sigma[i] ** 2))
        else:
            weight_temp = np.exp(-np.power(data[:, :, 2] - target_hue[i], 2) / (2 * hue_sigma[i] ** 2))

        weight_hue = multiplier[i] * weight_temp / np.max(weight_temp)

        weight_chroma = np.exp(-np.power(data[:, :, 1] - chroma_preference[i], 2) / (2 * chroma_sigma[i] ** 2))

        hue_correction = hue_correction + np.multiply(np.multiply(delta_hue, weight_hue), weight_chroma)

    # correct the hue
    data[:, :, 2] = data[:, :, 2] - hue_correction

    # lch to lab
    data = color.lch2lab(data)
    # lab to xyz
    data = color.lab2xyz(data, cie_version, illuminant)
    # xyz to rgb
    data = color.xyz2rgb(data, color_space, clip_range)

    data = misc.outOfGamutClipping(data, range=clip_range[1])
    return data


def apply_gamma(x):
    return x ** 0.8
    # return x ** (1.0 / 2.2)
    # x = x.copy()
    # idx = x <= 0.0031308
    # x[idx] *= 12.92
    # x[idx == False] = (x[idx == False] ** (1.0 / 2.4)) * 1.055 - 0.055
    # return x


def apply_tone_map(x, tone_mapping='Base'):
    if tone_mapping == 'Flash':
        return tone.perform_flash(x, perform_gamma_correction=0)/255.
    elif tone_mapping == 'Storm':
        return tone.perform_storm(x, perform_gamma_correction=0)/255.
    elif tone_mapping == 'Drago':
        tonemap = cv2.createTonemapDrago()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Mantiuk':
        tonemap = cv2.createTonemapMantiuk()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Reinhard':
        tonemap = cv2.createTonemapReinhard()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Linear':
        return np.clip(x/np.sort(x.flatten())[-50000], 0, 1)
    elif tone_mapping == 'Base':
        # return 3 * x ** 2 - 2 * x ** 3
        # tone_curve = loadmat('tone_curve.mat')
        from scipy.io import loadmat
        import os
        tone_curve = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights", "tone_curve.mat"))
        tone_curve = tone_curve['tc']
        x = np.round(x * (len(tone_curve) - 1)).astype(int)
        tone_mapped_image = np.squeeze(tone_curve[x])
        return tone_mapped_image
    else:
        raise ValueError(
            'Bad tone_mapping option value! Use the following options: "Base", "Flash", "Storm", "Linear", "Drago", "Mantiuk", "Reinhard"')


def autocontrast_using_pil(img, cutoff=3):
    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil = ImageOps.autocontrast(img_pil, cutoff=cutoff)
    output_image = np.array(img_pil).astype(np.float32) / 255.
    return output_image


def to_uint8(srgb):
    return (srgb * 255).astype(np.uint8)


def resize_using_pil(img, width=1296, height=864):
    img_pil = Image.fromarray(img)
    out_size = (width, height)
    if img_pil.size == out_size:
        return img
    out_img = img_pil.resize(out_size, Image.ANTIALIAS)
    out_img = np.array(out_img)
    return out_img


def infer_denoise(img, model, window_size, device):
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float() / 255.
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        output = model(img)
        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * 1, 0:w - mod_pad_w * 1]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    output = misc.outOfGamutClipping(output, range=255.)
    return output


def unsharp_masking(data, gaussian_kernel_size=[5, 5], gaussian_sigma=2.0, slope=1.5, tau_threshold=0.05, gamma_speed=4., clip_range=[0, 255]):
    # create gaussian kernel
    gaussian_kernel = misc.gaussian(gaussian_kernel_size, gaussian_sigma)

    # convolve the image with the gaussian kernel
    # first input is the image
    # second input is the kernel
    # output shape will be the same as the first input
    # boundary will be padded by using symmetrical method while convolving
    if np.ndim(data) > 2:
        image_blur = np.empty(np.shape(data), dtype=np.float32)
        for i in range(0, np.shape(data)[2]):
            image_blur[:, :, i] = signal.convolve2d(data[:, :, i], gaussian_kernel, mode="same", boundary="symm")
    else:
        image_blur = signal.convolve2d(data, gaussian_kernel, mode="same", boundary="symm")

    # the high frequency component image
    image_high_pass = data - image_blur

    # soft coring (see in utility)
    # basically pass the high pass image via a slightly nonlinear function
    tau_threshold = tau_threshold * clip_range[1]

    # add the soft cored high pass image to the original and clip
    # within range and return
    def soft_coring(img_hp, slope, tau_threshold, gamma_speed):
        return slope * np.float32(img_hp) * (1. - np.exp(-((np.abs(img_hp / tau_threshold))**gamma_speed)))
    return np.clip(data + soft_coring(image_high_pass, slope, tau_threshold, gamma_speed), clip_range[0], clip_range[1])


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image
