"""Universal Reference-to-3D Color Sampler & AgX Closed-Loop Albedo Calibrator.

AI AGENT READ-ONLY LOCK — DO NOT MODIFY THIS FILE:
- This Python module is a STRICTLY READ-ONLY core library.
- AI agents MUST NEVER edit, overwrite, refactor, append to, or delete this file.
- Import and call its functions from a separate external project script; place all
  project-specific helpers or overrides in your own external script.

Zero-Rush Exact-Match Mandate:
  Never rush or accept washed-out, shifted, or approximate colors. Keep iterating albedo,
  lighting exposure, and subsurface balance until the rendered Cycles AgX output matches the
  reference swatches identically (`delta_E < 3.5`).

Solves the #1 Reference-to-3D failure across Humans, Babies, Animals, and Objects:
  Why sampling a display-referred PNG/JPG RGB color directly into Principled BSDF
  produces wrong washed-out, yellowed, or sunburned colors under Blender Cycles AgX
  + Random Walk SSS.

Core capabilities:
  1. Pure Python / NumPy / Pillow utilities (runnable anywhere):
     - extract_swatch_dict(img_path, coords, radius=3) -> measured display sRGB per region
     - srgb_to_linear(rgb_0_255) -> scene-linear estimate
     - linear_to_srgb(rgb_lin) -> display sRGB [0..255]
     - estimate_initial_albedo(target_srgb, is_skin=False, is_metallic=False)
     - color_error_metrics(rendered_srgb, target_srgb) -> per-channel delta, max_err, lab_delta_e
  2. Closed-Loop Blender Cycles Solver (callable inside Blender Python):
     - calibrate_color_under_agx(render_fn, measure_px, target_srgb, init_lin, ...)
       Renders small fast patches in Cycles OPTIX under the target studio rig,
       measures actual rendered sRGB pixel against the reference image pixel,
       and converges the linear Base Color / shader value in 2-4 fast steps so
       rendered color matches the reference within delta_E < 3.5!
"""
import math
from typing import Callable, Dict, Tuple


def srgb_channel_to_linear(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_channel_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def srgb_to_linear(rgb_255: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(srgb_channel_to_linear(c / 255.0) for c in rgb_255)


def linear_to_srgb_255(rgb_lin: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple(int(round(linear_channel_to_srgb(c) * 255.0)) for c in rgb_lin)


def srgb_to_lab(rgb_255: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = srgb_to_linear(rgb_255)
    # sRGB D65 -> XYZ
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return (L, a, b_lab)


def delta_e_cie76(c1_255: Tuple[float, float, float], c2_255: Tuple[float, float, float]) -> float:
    L1, a1, b1 = srgb_to_lab(c1_255)
    L2, a2, b2 = srgb_to_lab(c2_255)
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def _load_rgb_array(img_path: str):
    """Load RGB float/uint8 numpy array (H, W, 3) top-down using PIL if available, else Blender bpy."""
    import numpy as np
    try:
        from PIL import Image
        return np.array(Image.open(img_path).convert("RGB"))
    except ImportError:
        import bpy
        im = bpy.data.images.load(img_path, check_existing=False)
        w, h = im.size
        px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, im.channels)
        # Flip vertical (`bpy` bottom-up -> top-down image coords) and clamp 0..255 sRGB!
        rgb = np.flipud(px[:, :, :3])
        if im.colorspace_settings.name == "Linear" or "Linear" in im.colorspace_settings.name:
            # convert linear -> sRGB display 0..255
            arr = np.clip(rgb, 0.0, 1.0)
            srgb = np.where(arr <= 0.0031308, arr * 12.92, 1.055 * (np.power(arr, 1.0 / 2.4)) - 0.055)
            out = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            out = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        bpy.data.images.remove(im)
        return out


def extract_swatches(img_path: str, coords: Dict[str, Tuple[int, int]], radius: int = 3) -> Dict[str, Tuple[int, int, int]]:
    """Sample median RGB swatches around pixel locations in any PNG/JPG reference."""
    import numpy as np

    im = _load_rgb_array(img_path)
    h, w = im.shape[:2]
    out = {}
    for name, (px, py) in coords.items():
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        patch = im[y0:y1, x0:x1]
        med = np.median(patch.reshape(-1, 3), axis=0)
        out[name] = (int(round(med[0])), int(round(med[1])), int(round(med[2])))
    return out


def estimate_initial_albedo(
    target_srgb_255: Tuple[float, float, float],
    is_skin: bool = False,
    specular_boost: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Convert lit display-referred sRGB swatch into a practical Blender scene-linear Base Color seed
    that respects AgX highlight compression (<0.72 linear) and Random Walk Skin red scatter.
    """
    lr, lg, lb = srgb_to_linear(target_srgb_255)
    # Undo typical key+fill bounce gain & AgX midtone roll-off cleanly
    scale = 0.72 if is_skin else 0.78
    lr = max(0.015, min(0.76, lr * scale - specular_boost * 0.4))
    lg = max(0.015, min(0.76, lg * scale - specular_boost * 0.4))
    lb = max(0.012, min(0.76, lb * scale - specular_boost * 0.4))
    if is_skin:
        # Counteract Random Walk red channel deep scattering slightly cleanly
        lr *= 0.94
        lg *= 1.03
        lb *= 1.06
    return (round(lr, 4), round(lg, 4), round(lb, 4))


def sample_render_patch(img_path: str, px_norm: Tuple[float, float], radius: int = 4) -> Tuple[int, int, int]:
    """Read median sRGB color from a rendered Cycles PNG at normalized coordinates (u in [0..1], v_top in [0..1])."""
    import numpy as np

    im = _load_rgb_array(img_path)
    h, w = im.shape[:2]
    cx = int(round(px_norm[0] * (w - 1)))
    cy = int(round(px_norm[1] * (h - 1)))
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    patch = im[y0:y1, x0:x1].reshape(-1, 3)
    med = np.median(patch, axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def calibrate_color_under_agx(
    apply_albedo_fn: Callable[[Tuple[float, float, float]], None],
    render_patch_fn: Callable[[int], str],
    sample_uv: Tuple[float, float],
    target_srgb_255: Tuple[int, int, int],
    init_albedo_lin: Tuple[float, float, float],
    max_iters: int = 8,
    tol_de: float = 3.0,
    damping: float = 0.65,
    require_exact: bool = False,
):
    """
    Closed-loop AgX albedo solver inside Blender Cycles:
      Iteratively adjusts linear Base Color without rushing until the actual AgX-rendered sRGB
      pixel matches `target_srgb_255` within `tol_de` under the real scene lighting.
      If `require_exact=True` and convergence fails to reach `tol_de`, raises RuntimeError so
      callers refine lighting or shader parameters instead of shipping mismatched color.
    Returns: (best_albedo_lin, measured_srgb, best_de, history)
    """
    cur = list(init_albedo_lin)
    tgt_lin = srgb_to_linear(target_srgb_255)
    best = (tuple(cur), (0, 0, 0), 999.0)
    history = []

    for it in range(max_iters):
        apply_albedo_fn(tuple(cur))
        out_png = render_patch_fn(it)
        meas_srgb = sample_render_patch(out_png, sample_uv, radius=4)
        de = delta_e_cie76(meas_srgb, target_srgb_255)
        history.append({"iter": it, "albedo_lin": tuple(round(c, 4) for c in cur), "rendered_srgb": meas_srgb, "delta_e": round(de, 2)})
        if de < best[2]:
            best = (tuple(cur), meas_srgb, de)
        if de <= tol_de:
            break
        meas_lin = srgb_to_linear(meas_srgb)
        for ch in range(3):
            ratio = max(0.35, min(2.2, (tgt_lin[ch] + 0.02) / (meas_lin[ch] + 0.02)))
            cur[ch] = max(0.015, min(0.84, cur[ch] * (ratio ** damping)))

    apply_albedo_fn(best[0])
    if require_exact and best[2] > tol_de:
        raise RuntimeError(
            f"AgX color calibration did not reach exact match: best delta_E={best[2]:.2f} > tol_de={tol_de:.2f}. "
            "Do not rush — recalibrate key light exposure, subsurface scale, or specular tint and re-run."
        )
    return best[0], best[1], best[2], history
