import numpy as np


def rgb2gray(data):
    return 0.299 * data[:, :, 0] + \
           0.587 * data[:, :, 1] + \
           0.114 * data[:, :, 2]


def rgb2ycc(data, rule="bt601"):
    # map to select kr and kb
    kr_kb_dict = {"bt601": [0.299, 0.114],
                  "bt709": [0.2126, 0.0722],
                  "bt2020": [0.2627, 0.0593]}

    kr = kr_kb_dict[rule][0]
    kb = kr_kb_dict[rule][1]
    kg = 1 - (kr + kb)

    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = kr * data[:, :, 0] + \
                      kg * data[:, :, 1] + \
                      kb * data[:, :, 2]
    output[:, :, 1] = 0.5 * ((data[:, :, 2] - output[:, :, 0]) / (1 - kb))
    output[:, :, 2] = 0.5 * ((data[:, :, 0] - output[:, :, 0]) / (1 - kr))

    return output


def ycc2rgb(data, rule="bt601"):
    # map to select kr and kb
    kr_kb_dict = {"bt601": [0.299, 0.114],
                  "bt709": [0.2126, 0.0722],
                  "bt2020": [0.2627, 0.0593]}

    kr = kr_kb_dict[rule][0]
    kb = kr_kb_dict[rule][1]
    kg = 1 - (kr + kb)

    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = 2. * data[:, :, 2] * (1 - kr) + data[:, :, 0]
    output[:, :, 2] = 2. * data[:, :, 1] * (1 - kb) + data[:, :, 0]
    output[:, :, 1] = (data[:, :, 0] - kr * output[:, :, 0] - kb * output[:, :, 2]) / kg

    return output


def degamma_srgb(data, clip_range=[0, 65535]):
    # bring data in range 0 to 1
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.asarray(data)
    mask = data > 0.04045

    # basically, if data[x, y, c] > 0.04045, data[x, y, c] = ( (data[x, y, c] + 0.055) / 1.055 ) ^ 2.4
    #            else, data[x, y, c] = data[x, y, c] / 12.92
    data[mask] += 0.055
    data[mask] /= 1.055
    data[mask] **= 2.4

    data[np.invert(mask)] /= 12.92

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def degamma_adobe_rgb_1998(data, clip_range=[0, 65535]):
    # bring data in range 0 to 1
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.power(data, 2.2)  # originally raised to 2.19921875

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def rgb2xyz(data, color_space="srgb", clip_range=[0, 255]):
    # input rgb in range clip_range
    # output xyz is in range 0 to 1
    if color_space == "srgb":
        # degamma / linearization
        data = degamma_srgb(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])

        # matrix multiplication`
        output = np.empty(np.shape(data), dtype=np.float32)
        output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
        output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
        output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505
    elif color_space == "adobe-rgb-1998":
        # degamma / linearization
        data = degamma_adobe_rgb_1998(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])

        # matrix multiplication
        output = np.empty(np.shape(data), dtype=np.float32)
        output[:, :, 0] = data[:, :, 0] * 0.5767309 + data[:, :, 1] * 0.1855540 + data[:, :, 2] * 0.1881852
        output[:, :, 1] = data[:, :, 0] * 0.2973769 + data[:, :, 1] * 0.6273491 + data[:, :, 2] * 0.0752741
        output[:, :, 2] = data[:, :, 0] * 0.0270343 + data[:, :, 1] * 0.0706872 + data[:, :, 2] * 0.9911085
    elif color_space == "linear":
        # matrix multiplication`
        output = np.empty(np.shape(data), dtype=np.float32)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])
        output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
        output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
        output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505
    else:
        print("Warning! color_space must be srgb or adobe-rgb-1998.")
        return

    return output


def gamma_srgb(data, clip_range=[0, 65535]):
    # bring data in range 0 to 1
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.asarray(data)
    mask = data > 0.0031308

    # basically, if data[x, y, c] > 0.0031308, data[x, y, c] = 1.055 * ( var_R(i, j) ^ ( 1 / 2.4 ) ) - 0.055
    #            else, data[x, y, c] = data[x, y, c] * 12.92
    data[mask] **= 0.4167
    data[mask] *= 1.055
    data[mask] -= 0.055

    data[np.invert(mask)] *= 12.92

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def gamma_adobe_rgb_1998(data, clip_range=[0, 65535]):
    # bring data in range 0 to 1
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.power(data, 0.4545)

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def xyz2rgb(data, color_space="srgb", clip_range=[0, 255]):
    # input xyz is in range 0 to 1
    # output rgb in clip_range

    # allocate space for output
    output = np.empty(np.shape(data), dtype=np.float32)

    if color_space == "srgb":
        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 3.2406 + data[:, :, 1] * -1.5372 + data[:, :, 2] * -0.4986
        output[:, :, 1] = data[:, :, 0] * -0.9689 + data[:, :, 1] * 1.8758 + data[:, :, 2] * 0.0415
        output[:, :, 2] = data[:, :, 0] * 0.0557 + data[:, :, 1] * -0.2040 + data[:, :, 2] * 1.0570

        # gamma to retain nonlinearity
        output = gamma_srgb(output * clip_range[1], clip_range)
    elif color_space == "adobe-rgb-1998":
        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 2.0413690 + data[:, :, 1] * -0.5649464 + data[:, :, 2] * -0.3446944
        output[:, :, 1] = data[:, :, 0] * -0.9692660 + data[:, :, 1] * 1.8760108 + data[:, :, 2] * 0.0415560
        output[:, :, 2] = data[:, :, 0] * 0.0134474 + data[:, :, 1] * -0.1183897 + data[:, :, 2] * 1.0154096

        # gamma to retain nonlinearity
        output = gamma_adobe_rgb_1998(output * clip_range[1], clip_range)
    elif color_space == "linear":

        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 3.2406 + data[:, :, 1] * -1.5372 + data[:, :, 2] * -0.4986
        output[:, :, 1] = data[:, :, 0] * -0.9689 + data[:, :, 1] * 1.8758 + data[:, :, 2] * 0.0415
        output[:, :, 2] = data[:, :, 0] * 0.0557 + data[:, :, 1] * -0.2040 + data[:, :, 2] * 1.0570

        # gamma to retain nonlinearity
        output = output * clip_range[1]
    else:
        print("Warning! color_space must be srgb or adobe-rgb-1998.")
        return

    return output


def get_xyz_reference(cie_version="1931", illuminant="d65"):
    if cie_version == "1931":
        xyz_reference_dictionary = {"A": [109.850, 100.0, 35.585],
                                    "B": [99.0927, 100.0, 85.313],
                                    "C": [98.074, 100.0, 118.232],
                                    "d50": [96.422, 100.0, 82.521],
                                    "d55": [95.682, 100.0, 92.149],
                                    "d65": [95.047, 100.0, 108.883],
                                    "d75": [94.972, 100.0, 122.638],
                                    "E": [100.0, 100.0, 100.0],
                                    "F1": [92.834, 100.0, 103.665],
                                    "F2": [99.187, 100.0, 67.395],
                                    "F3": [103.754, 100.0, 49.861],
                                    "F4": [109.147, 100.0, 38.813],
                                    "F5": [90.872, 100.0, 98.723],
                                    "F6": [97.309, 100.0, 60.191],
                                    "F7": [95.044, 100.0, 108.755],
                                    "F8": [96.413, 100.0, 82.333],
                                    "F9": [100.365, 100.0, 67.868],
                                    "F10": [96.174, 100.0, 81.712],
                                    "F11": [100.966, 100.0, 64.370],
                                    "F12": [108.046, 100.0, 39.228]}
    elif cie_version == "1964":
        xyz_reference_dictionary = {"A": [111.144, 100.0, 35.200],
                                    "B": [99.178, 100.0, 84.3493],
                                    "C": [97.285, 100.0, 116.145],
                                    "D50": [96.720, 100.0, 81.427],
                                    "D55": [95.799, 100.0, 90.926],
                                    "D65": [94.811, 100.0, 107.304],
                                    "D75": [94.416, 100.0, 120.641],
                                    "E": [100.0, 100.0, 100.0],
                                    "F1": [94.791, 100.0, 103.191],
                                    "F2": [103.280, 100.0, 69.026],
                                    "F3": [108.968, 100.0, 51.965],
                                    "F4": [114.961, 100.0, 40.963],
                                    "F5": [93.369, 100.0, 98.636],
                                    "F6": [102.148, 100.0, 62.074],
                                    "F7": [95.792, 100.0, 107.687],
                                    "F8": [97.115, 100.0, 81.135],
                                    "F9": [102.116, 100.0, 67.826],
                                    "F10": [99.001, 100.0, 83.134],
                                    "F11": [103.866, 100.0, 65.627],
                                    "F12": [111.428, 100.0, 40.353]}
    else:
        print("Warning! cie_version must be 1931 or 1964.")
        return
    return np.divide(xyz_reference_dictionary[illuminant], 100.0)


def xyz2lab(data, cie_version="1931", illuminant="d65"):
    xyz_reference = get_xyz_reference(cie_version, illuminant)

    data = data
    data[:, :, 0] = data[:, :, 0] / xyz_reference[0]
    data[:, :, 1] = data[:, :, 1] / xyz_reference[1]
    data[:, :, 2] = data[:, :, 2] / xyz_reference[2]

    data = np.asarray(data)

    # if data[x, y, c] > 0.008856, data[x, y, c] = data[x, y, c] ^ (1/3)
    # else, data[x, y, c] = 7.787 * data[x, y, c] + 16/116
    mask = data > 0.008856
    data[mask] **= 1. / 3.
    data[np.invert(mask)] *= 7.787
    data[np.invert(mask)] += 16. / 116.

    data = np.float32(data)
    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = 116. * data[:, :, 1] - 16.
    output[:, :, 1] = 500. * (data[:, :, 0] - data[:, :, 1])
    output[:, :, 2] = 200. * (data[:, :, 1] - data[:, :, 2])

    return output


def lab2xyz(data, cie_version="1931", illuminant="d65"):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 1] = (data[:, :, 0] + 16.) / 116.
    output[:, :, 0] = (data[:, :, 1] / 500.) + output[:, :, 1]
    output[:, :, 2] = output[:, :, 1] - (data[:, :, 2] / 200.)

    # if output[x, y, c] > 0.008856, output[x, y, c] ^ 3
    # else, output[x, y, c] = ( output[x, y, c] - 16/116 ) / 7.787
    output = np.asarray(output)
    mask = output > 0.008856
    output[mask] **= 3.
    output[np.invert(mask)] -= 16 / 116
    output[np.invert(mask)] /= 7.787

    xyz_reference = get_xyz_reference(cie_version, illuminant)

    output = np.float32(output)
    output[:, :, 0] = output[:, :, 0] * xyz_reference[0]
    output[:, :, 1] = output[:, :, 1] * xyz_reference[1]
    output[:, :, 2] = output[:, :, 2] * xyz_reference[2]

    return output


def lab2lch(data):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 0] = data[:, :, 0]  # L transfers directly
    output[:, :, 1] = np.power(np.power(data[:, :, 1], 2) + np.power(data[:, :, 2], 2), 0.5)
    output[:, :, 2] = np.arctan2(data[:, :, 2], data[:, :, 1]) * 180 / np.pi

    return output


def lch2lab(data):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 0] = data[:, :, 0]  # L transfers directly
    output[:, :, 1] = np.multiply(np.cos(data[:, :, 2] * np.pi / 180), data[:, :, 1])
    output[:, :, 2] = np.multiply(np.sin(data[:, :, 2] * np.pi / 180), data[:, :, 1])

    return output
