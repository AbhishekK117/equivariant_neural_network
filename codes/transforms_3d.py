import torch
import torch.nn.functional as F
import numpy as np

def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        iz_nw = torch.floor(iz)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        iz_ne = iz_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        iz_sw = iz_nw
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1
        iz_se = iz_nw
        ix_nw_d = ix_nw
        iy_nw_d = iy_nw
        iz_nw_d = iz_nw + 1
        ix_ne_d = ix_ne
        iy_ne_d = iy_ne
        iz_ne_d = iz_ne + 1
        ix_sw_d = ix_sw
        iy_sw_d = iy_sw
        iz_sw_d = iz_sw + 1
        ix_se_d = ix_se
        iy_se_d = iy_se
        iz_se_d = iz_se + 1

    nw = (ix_se - ix) * (iy_se - iy) * (iz_se - iz)
    ne = (ix - ix_sw) * (iy_sw - iy) * (iz_sw - iz)
    sw = (ix_ne - ix) * (iy - iy_ne) * (iz - iz_ne)
    se = (ix - ix_nw) * (iy - iy_nw) * (iz - iz_nw)
    nw_d = (ix_se_d - ix) * (iy_se_d - iy) * (iz - iz_nw_d)
    ne_d = (ix - ix_sw_d) * (iy_sw_d - iy) * (iz - iz_sw_d)
    sw_d = (ix_ne_d - ix) * (iy - iy_ne_d) * (iz - iz_ne_d)
    se_d = (ix - ix_nw_d) * (iy - iy_nw_d) * (iz - iz_nw_d)

    with torch.no_grad():
        ix_nw = IW - 1 - (IW - 1 - ix_nw.abs()).abs()
        iy_nw = IH - 1 - (IH - 1 - iy_nw.abs()).abs()
        iz_nw = ID - 1 - (ID - 1 - iz_nw.abs()).abs()

        ix_ne = IW - 1 - (IW - 1 - ix_ne.abs()).abs()
        iy_ne = IH - 1 - (IH - 1 - iy_ne.abs()).abs()
        iz_ne = ID - 1 - (ID - 1 - iz_ne.abs()).abs()

        ix_sw = IW - 1 - (IW - 1 - ix_sw.abs()).abs()
        iy_sw = IH - 1 - (IH - 1 - iy_sw.abs()).abs()
        iz_sw = ID - 1 - (ID - 1 - iz_sw.abs()).abs()

        ix_se = IW - 1 - (IW - 1 - ix_se.abs()).abs()
        iy_se = IH - 1 - (IH - 1 - iy_se.abs()).abs()
        iz_se = ID - 1 - (ID - 1 - iz_se.abs()).abs()

        ix_nw_d = IW - 1 - (IW - 1 - ix_nw_d.abs()).abs()
        iy_nw_d = IH - 1 - (IH - 1 - iy_nw_d.abs()).abs()
        iz_nw_d = ID - 1 - (ID - 1 - iz_nw_d.abs()).abs()

        ix_ne_d = IW - 1 - (IW - 1 - ix_ne_d.abs()).abs()
        iy_ne_d = IH - 1 - (IH - 1 - iy_ne_d.abs()).abs()
        iz_ne_d = ID - 1 - (ID - 1 - iz_ne_d.abs()).abs()

        ix_sw_d = IW - 1 - (IW - 1 - ix_sw_d.abs()).abs()
        iy_sw_d = IH - 1 - (IH - 1 - iy_sw_d.abs()).abs()
        iz_sw_d = ID - 1 - (ID - 1 - iz_sw_d.abs()).abs()

        ix_se_d = IW - 1 - (IW - 1 - ix_se_d.abs()).abs()
        iy_se_d = IH - 1 - (IH - 1 - iy_se_d.abs()).abs()
        iz_se_d = ID - 1 - (ID - 1 - iz_se_d.abs()).abs()

    image = image.view(N, C, ID * IH * IW)

    nw_val = torch.gather(image, 2, (iz_nw * IH * IW + iy_nw * IW + ix_nw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iz_ne * IH * IW + iy_ne * IW + ix_ne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iz_sw * IH * IW + iy_sw * IW + ix_sw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iz_se * IH * IW + iy_se * IW + ix_se).long().view(N, 1, D * H * W).repeat(1, C, 1))
    
    nw_d_val = torch.gather(image, 2, (iz_nw_d * IH * IW + iy_nw_d * IW + ix_nw_d).long().view(N, 1, D * H * W).repeat(1, C, 1))
    ne_d_val = torch.gather(image, 2, (iz_ne_d * IH * IW + iy_ne_d * IW + ix_ne_d).long().view(N, 1, D * H * W).repeat(1, C, 1))
    sw_d_val = torch.gather(image, 2, (iz_sw_d * IH * IW + iy_sw_d * IW + ix_sw_d).long().view(N, 1, D * H * W).repeat(1, C, 1))
    se_d_val = torch.gather(image, 2, (iz_se_d * IH * IW + iy_se_d * IW + ix_se_d).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (
        nw_val.view(N, C, D, H, W) * nw.view(N, 1, D, H, W)
        + ne_val.view(N, C, D, H, W) * ne.view(N, 1, D, H, W)
        + sw_val.view(N, C, D, H, W) * sw.view(N, 1, D, H, W)
        + se_val.view(N, C, D, H, W) * se.view(N, 1, D, H, W)
        + nw_d_val.view(N, C, D, H, W) * nw_d.view(N, 1, D, H, W)
        + ne_d_val.view(N, C, D, H, W) * ne_d.view(N, 1, D, H, W)
        + sw_d_val.view(N, C, D, H, W) * sw_d.view(N, 1, D, H, W)
        + se_d_val.view(N, C, D, H, W) * se_d.view(N, 1, D, H, W)
    )

    return out_val

def img_like_3d(img_shape):
    bcdhw = len(img_shape) == 5 and img_shape[-3:] != (1, 1, 1)
    is_cube = int(int(np.cbrt(img_shape[1])) + 0.5) ** 3 == img_shape[1]
    is_one_off_cube = int(int(np.cbrt(img_shape[1])) + 0.5) ** 3 == img_shape[1] - 1
    is_two_off_cube = int(int(np.cbrt(img_shape[1])) + 0.5) ** 3 == img_shape[1] - 2
    bnc = (
        len(img_shape) == 3
        and img_shape[1] != 1
        and (is_cube or is_one_off_cube or is_two_off_cube)
    )
    return bcdhw or bnc

def num_tokens_3d(img_shape):
    if len(img_shape) == 5 and img_shape[-3:] != (1, 1, 1):
        return 0
    is_one_off_cube = int(int(np.cbrt(img_shape[1])) + 0.5) ** 3 == img_shape[1] - 1
    is_two_off_cube = int(int(np.cbrt(img_shape[1])) + 0.5) ** 3 == img_shape[1] - 2
    return int(is_one_off_cube * 1 or is_two_off_cube * 2)

def bnc2bcdhw(bnc, num_tokens):
    b, n, c = bnc.shape
    d = h = w = int(np.cbrt(n))
    extra = bnc[:, :num_tokens, :]
    img = bnc[:, num_tokens:, :]
    return img.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3), extra

def bcdhw2bnc(bcdhw, tokens):
    b, c, d, h, w = bcdhw.shape
    n = d * h * w
    bnc = bcdhw.permute(0, 2, 3, 4, 1).reshape(b, n, c)
    return torch.cat([tokens, bnc], dim=1)  # assumes tokens are at the start

def affine_transform_3d(affineMatrices, img):
    assert img_like_3d(img.shape)
    if len(img.shape) == 3:
        ntokens = num_tokens_3d(img.shape)
        x, extra = bnc2bcdhw(img, ntokens)
    else:
        x = img
    flowgrid = F.affine_grid(
        affineMatrices, size=x.size(), align_corners=True
    )  # .double()
    transformed = grid_sample_3d(x, flowgrid)
    if len(img.shape) == 3:
        transformed = bcdhw2bnc(transformed, extra)
    return transformed

def translate_3d(img, t, axis="x"):
    """Translates a 3D volume by a fraction of the size (tx, ty, tz) in (0,1)"""
    affineMatrices = torch.zeros(img.shape[0], 3, 4).to(img.device)
    affineMatrices[:, 0, 0] = 1
    affineMatrices[:, 1, 1] = 1
    affineMatrices[:, 2, 2] = 1
    if axis == "x":
        affineMatrices[:, 0, 3] = t
    elif axis == "y":
        affineMatrices[:, 1, 3] = t
    else:
        affineMatrices[:, 2, 3] = t
    return affine_transform_3d(affineMatrices, img)

def rotate_3d(img, angle, axis="z"):
    """Rotates a 3D volume by angle around the specified axis"""
    affineMatrices = torch.zeros(img.shape[0], 3, 4).to(img.device)
    if axis == "x":
        affineMatrices[:, 1, 1] = torch.cos(angle)
        affineMatrices[:, 1, 2] = -torch.sin(angle)
        affineMatrices[:, 2, 1] = torch.sin(angle)
        affineMatrices[:, 2, 2] = torch.cos(angle)
    elif axis == "y":
        affineMatrices[:, 0, 0] = torch.cos(angle)
        affineMatrices[:, 0, 2] = torch.sin(angle)
        affineMatrices[:, 2, 0] = -torch.sin(angle)
        affineMatrices[:, 2, 2] = torch.cos(angle)
    else:  # axis == "z"
        affineMatrices[:, 0, 0] = torch.cos(angle)
        affineMatrices[:, 0, 1] = -torch.sin(angle)
        affineMatrices[:, 1, 0] = torch.sin(angle)
        affineMatrices[:, 1, 1] = torch.cos(angle)
    return affine_transform_3d(affineMatrices, img)

def shear_3d(img, t, axis="x"):
    """Shear a 3D volume by an amount t along the specified axis"""
    affineMatrices = torch.zeros(img.shape[0], 3, 4).to(img.device)
    affineMatrices[:, 0, 0] = 1
    affineMatrices[:, 1, 1] = 1
    affineMatrices[:, 2, 2] = 1
    if axis == "x":
        affineMatrices[:, 0, 1] = t
    elif axis == "y":
        affineMatrices[:, 1, 0] = t
    else:
        affineMatrices[:, 2, 0] = t
    return affine_transform_3d(affineMatrices, img)

def stretch_3d(img, x, axis="x"):
    """Stretch a 3D volume by an amount x along the specified axis"""
    affineMatrices = torch.zeros(img.shape[0], 3, 4).to(img.device)
    if axis == "x":
        affineMatrices[:, 0, 0] = 1 * (1 + x)
    elif axis == "y":
        affineMatrices[:, 1, 1] = 1 * (1 + x)
    else:
        affineMatrices[:, 2, 2] = 1 * (1 + x)
    return affine_transform_3d(affineMatrices, img)

def hyperbolic_rotate_3d(img, angle):
    """Hyperbolic rotation in 3D"""
    affineMatrices = torch.zeros(img.shape[0], 3, 4).to(img.device)
    affineMatrices[:, 0, 0] = torch.cosh(angle)
    affineMatrices[:, 0, 1] = torch.sinh(angle)
    affineMatrices[:, 1, 0] = torch.sinh(angle)
    affineMatrices[:, 1, 1] = torch.cosh(angle)
    return affine_transform_3d(affineMatrices, img)

def scale_3d(img, s):
    """Scale a 3D volume by a factor s"""
    affineMatrices = torch.zeros(img.shape[0], 3, 4).to(img.device)
    affineMatrices[:, 0, 0] = 1 - s
    affineMatrices[:, 1, 1] = 1 - s
    affineMatrices[:, 2, 2] = 1 - s
    return affine_transform_3d(affineMatrices, img)

def saturate_3d(img, t):
    """Saturate a 3D volume by a factor t"""
    img = img.clone()
    img *= 1 + t
    return img
