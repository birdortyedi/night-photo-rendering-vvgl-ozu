from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from colour import cctf_encoding

from utils import misc
from ips import ops


def process(raw_image, metadata, wb_network, denoiser_network):
    img = ops.linearize_raw(raw_image, metadata)
    out = ops.normalize(img, metadata)
    out = ops.bad_pixel_correction(out)
    out = demosaicing_CFA_Bayer_Menon2007(out, misc.decode_cfa_pattern(metadata['cfa_pattern']))
    out = cctf_encoding(out)
    out = misc.outOfGamutClipping(out)
    out = ops.apply_color_space_transform(out, metadata['color_matrix_1'], metadata['color_matrix_2'])
    out = ops.transform_xyz_to_srgb(out)
    out = ops.white_balance_w_style(out, wb_network, metadata["wb_settings"], True, True, device=metadata["device"])
    out = out[:, :, ::-1]
    out = ops.memory_color_enhancement(out, clip_range=[0, 1])
    out = ops.apply_gamma(out)
    out = ops.apply_tone_map(out, "Flash")
    out = ops.autocontrast_using_pil(out)
    out = ops.to_uint8(out)
    out = ops.resize_using_pil(out, metadata["exp_width"] * 5 // 4, metadata["exp_height"] * 5 // 4)
    out = out[:, :, ::-1]
    out = ops.infer_denoise(out, denoiser_network, metadata["window_size"], device=metadata["device"])
    out = ops.fix_orientation(out, metadata['orientation'])
    out = ops.resize_using_pil(out, metadata["exp_width"], metadata["exp_height"])
    out = ops.unsharp_masking(out)
    return out
