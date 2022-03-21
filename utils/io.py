import cv2
import json
import torch
from pathlib import Path
from fractions import Fraction


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fraction_from_json(json_object):
    if 'Fraction' in json_object:
        return Fraction(*json_object['Fraction'])
    return json_object


def json_read(fname, **kwargs):
    with open(fname) as j:
        data = json.load(j, **kwargs)
    return data


def read_image(path):
    png_path = Path(path)
    raw_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    metadata = json_read(png_path.with_suffix('.json'), object_hook=fraction_from_json)
    return raw_image, metadata


def write_processed_as_jpg(out, dst_path, quality=100):
    cv2.imwrite(dst_path, out, [cv2.IMWRITE_JPEG_QUALITY, quality])
