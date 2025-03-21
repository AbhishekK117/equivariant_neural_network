import torch
from transforms_3d import *  # Import the 3D transformation functions

def jvp(f, x, u):
    """Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).
    Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u"""
    with torch.enable_grad():
        y = f(x)
        v = torch.ones_like(
            y, requires_grad=True
        )  # Dummy variable (could take any value)
        vJ = torch.autograd.grad(y, [x], [v], create_graph=True)
        Ju = torch.autograd.grad(vJ, [v], [u], create_graph=True)
        return Ju[0]


def translation_lie_deriv_3d(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to translation vector, output can be a scalar or a 3D volume"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def shifted_model(t):
        shifted_img = translate_3d(inp_imgs, t, axis)
        z = model(shifted_img)
        if img_like_3d(z.shape):
            z = translate_3d(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(shifted_model, t, torch.ones_like(t, requires_grad=True))
    return lie_deriv


def rotation_lie_deriv_3d(model, inp_imgs, axis="z"):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def rotated_model(t):
        rotated_img = rotate_3d(inp_imgs, t, axis)
        z = model(rotated_img)
        if img_like_3d(z.shape):
            z = rotate_3d(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def hyperbolic_rotation_lie_deriv_3d(model, inp_imgs):
    """Lie derivative of model with respect to hyperbolic rotation, assumes scalar output"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def rotated_model(t):
        rotated_img = hyperbolic_rotate_3d(inp_imgs, t)
        z = model(rotated_img)
        if img_like_3d(z.shape):
            z = hyperbolic_rotate_3d(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def scale_lie_deriv_3d(model, inp_imgs):
    """Lie derivative of model with respect to scaling, assumes scalar output"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def scaled_model(t):
        scaled_img = scale_3d(inp_imgs, t)
        z = model(scaled_img)
        if img_like_3d(z.shape):
            z = scale_3d(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(scaled_model, t, torch.ones_like(t))
    return lie_deriv


def shear_lie_deriv_3d(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to shear, assumes scalar output"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def sheared_model(t):
        sheared_img = shear_3d(inp_imgs, t, axis)
        z = model(sheared_img)
        if img_like_3d(z.shape):
            z = shear_3d(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(sheared_model, t, torch.ones_like(t))
    return lie_deriv


def stretch_lie_deriv_3d(model, inp_imgs, axis="x"):
    """Lie derivative of model with respect to stretch, assumes scalar output"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def stretched_model(t):
        stretched_img = stretch_3d(inp_imgs, t, axis)
        z = model(stretched_img)
        if img_like_3d(z.shape):
            z = stretch_3d(z, -t, axis)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(stretched_model, t, torch.ones_like(t))
    return lie_deriv


def saturate_lie_deriv_3d(model, inp_imgs):
    """Lie derivative of model with respect to saturation, assumes scalar output"""
    if not img_like_3d(inp_imgs.shape):
        return 0.0

    def saturated_model(t):
        saturated_img = saturate_3d(inp_imgs, t)
        z = model(saturated_img)
        if img_like_3d(z.shape):
            z = saturate_3d(z, -t)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(saturated_model, t, torch.ones_like(t))
    return lie_deriv
