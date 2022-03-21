import os
import argparse
import glog as log

from utils import io
from modeling import swin_ir, mixed_wb
import ips

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Night Photography Rendering Challenge - Team VVGL OzU')
    parser.add_argument('-d', '--data_dir', type=str, default="data/", help="data directory")
    parser.add_argument('-s', '--submission_name', type=str, default="vvgl-ozu", help='submission name')
    args = parser.parse_args()

    base_dir = args.data_dir
    expected_landscape_img_height = 866
    expected_landscape_img_width = 1300
    device = io.get_device()

    denoiser_model_path = "weights/005_colorDN_DFWB_s128w8_SwinIR-M_noise{}.pth"
    window_size = 8
    img_range = 1.
    denoiser = swin_ir.build_model(window_size=window_size, img_range=img_range, device=device)

    wb_model_name = "WB_model_p_64_D_S_T_F_C"
    wb_settings = wb_model_name.split("_")[4:]
    wb_model_path = os.path.join("weights", wb_model_name + ".pth")
    wb_network = mixed_wb.build_model(wb_model_path=wb_model_path, wb_settings=wb_settings, device=device)

    for img_name in os.listdir(base_dir):
        if ".png" not in img_name: continue
        log.info("Processing image {}".format(img_name))

        path = os.path.join(base_dir, img_name)
        assert os.path.exists(path)

        raw_image, metadata = io.read_image(path)
        swin_ir.load_weights_by_noise_level(metadata, denoiser, denoiser_model_path)
        metadata["exp_height"] = expected_landscape_img_height
        metadata["exp_width"] = expected_landscape_img_width
        metadata["wb_settings"] = wb_settings
        metadata["window_size"] = window_size
        metadata["device"] = device

        out = ips.process(
            raw_image=raw_image,
            metadata=metadata,
            wb_network=wb_network,
            denoiser_network=denoiser
        )
        out_path = os.path.join("/" + base_dir, img_name.replace("png", "jpg"))
        print(out_path)
        io.write_processed_as_jpg(out, out_path)
