"""Universal Reference-Photo Skin De-Lighter, Fitzpatrick/ITA Profiler & Blender 5.2 Shader Builder.

AI AGENT READ-ONLY LOCK — DO NOT MODIFY THIS FILE:
- This Python module is a STRICTLY READ-ONLY core library.
- AI agents MUST NEVER edit, overwrite, refactor, append to, or delete this file.
- Import and call its functions from a separate external project script; place all
  project-specific helpers or overrides in your own external script.

Zero-Rush Exact-Match Mandate:
  Never rush skin lookdev or accept waxy, orange-bleeding, chalky, or mismatched skin tones.
  Iterate albedo de-lighting, Fitzpatrick SSS depth, sebum coat, and facial zone glazes against
  the reference portrait until the rendered Cycles output matches the reference 1:1 (`delta_E < 3.5`).

Combines biophysical Beer-Lambert chromophore absorption modeling, calibrated Fitzpatrick I-VI
linear PBR skin albedos, 6-zone facial color theory, and Cycles RANDOM_WALK_SKIN kernel calibration
(subsurface_ior=1.40, scale/radius adaptation per melanin tier, dual-lobe sebum coat,
cavity pore occlusion, and vellus peach-fuzz sheen).

Pure Python / NumPy / Pillow utilities run inside or outside Blender; Blender helpers build
calibrated Principled BSDF materials directly from any reference portrait or Fitzpatrick tier.
"""
import math
import os
import sys
from typing import Dict, Optional, Tuple

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import color_calibrator as cc  # Reuse srgb_to_linear, linear_to_srgb_255, srgb_to_lab, delta_e_cie76, _load_rgb_array


# ==============================================================================
# 1. PHYSICALLY BASED FITZPATRICK PRESETS & BIOPHYSICAL CHROMOPHORE MODEL
# ==============================================================================
FITZPATRICK_PRESETS: Dict[str, dict] = {
    "I": {
        "label": "Fitzpatrick I (Pale Fair / Ivory)",
        "linear_albedo": (0.740, 0.558, 0.483),   # Scaled cleanly for AgX highlight headroom
        "ref_linear_raw": (0.847, 0.638, 0.552),
        "subsurface_radius": (0.482, 0.169, 0.109),
        "subsurface_scale": 0.0044,
        "roughness_base": 0.48,
        "coat_weight": 0.28,
        "coat_roughness": 0.24,
        "sheen_weight": 0.06,
        "cavity_darkening": 0.24,
        "upper_lip_linear": (0.440, 0.165, 0.145),
        "lower_lip_linear": (0.520, 0.205, 0.180),
        "cheek_tint_shift": (0.98, 0.72, 0.68),
        "forehead_tint_shift": (1.03, 1.01, 0.94),
        "jaw_tint_shift": (0.94, 0.97, 1.00),
    },
    "II": {
        "label": "Fitzpatrick II (Light / Fair Peach)",
        "linear_albedo": (0.640, 0.435, 0.335),
        "ref_linear_raw": (0.799, 0.485, 0.347),
        "subsurface_radius": (0.482, 0.169, 0.109),
        "subsurface_scale": 0.0038,
        "roughness_base": 0.46,
        "coat_weight": 0.32,
        "coat_roughness": 0.23,
        "sheen_weight": 0.06,
        "cavity_darkening": 0.26,
        "upper_lip_linear": (0.390, 0.140, 0.118),
        "lower_lip_linear": (0.470, 0.182, 0.152),
        "cheek_tint_shift": (1.00, 0.73, 0.66),
        "forehead_tint_shift": (1.03, 1.02, 0.93),
        "jaw_tint_shift": (0.93, 0.97, 0.99),
    },
    "III": {
        "label": "Fitzpatrick III (Medium / Golden Olive)",
        "linear_albedo": (0.520, 0.328, 0.220),
        "ref_linear_raw": (0.623, 0.433, 0.343),
        "subsurface_radius": (0.482, 0.169, 0.109),
        "subsurface_scale": 0.0032,
        "roughness_base": 0.44,
        "coat_weight": 0.35,
        "coat_roughness": 0.22,
        "sheen_weight": 0.05,
        "cavity_darkening": 0.28,
        "upper_lip_linear": (0.305, 0.120, 0.098),
        "lower_lip_linear": (0.385, 0.160, 0.128),
        "cheek_tint_shift": (1.02, 0.76, 0.68),
        "forehead_tint_shift": (1.04, 1.03, 0.91),
        "jaw_tint_shift": (0.91, 0.96, 0.96),
    },
    "IV": {
        "label": "Fitzpatrick IV (Tan / South Asian Wheat-Brown)",
        "linear_albedo": (0.405, 0.220, 0.130),
        "ref_linear_raw": (0.436, 0.227, 0.131),
        "subsurface_radius": (0.367, 0.137, 0.068),
        "subsurface_scale": 0.0025,
        "roughness_base": 0.43,
        "coat_weight": 0.38,
        "coat_roughness": 0.22,
        "sheen_weight": 0.05,
        "cavity_darkening": 0.30,
        "upper_lip_linear": (0.205, 0.086, 0.070),
        "lower_lip_linear": (0.305, 0.135, 0.102),
        "cheek_tint_shift": (1.04, 0.78, 0.70),
        "forehead_tint_shift": (1.02, 1.03, 0.90),
        "jaw_tint_shift": (0.90, 0.94, 0.95),
    },
    "V": {
        "label": "Fitzpatrick V (Rich Caramel / Brown)",
        "linear_albedo": (0.255, 0.135, 0.072),
        "ref_linear_raw": (0.283, 0.148, 0.079),
        "subsurface_radius": (0.367, 0.137, 0.068),
        "subsurface_scale": 0.0019,
        "roughness_base": 0.41,
        "coat_weight": 0.42,
        "coat_roughness": 0.21,
        "sheen_weight": 0.04,
        "cavity_darkening": 0.30,
        "upper_lip_linear": (0.120, 0.060, 0.048),
        "lower_lip_linear": (0.210, 0.102, 0.078),
        "cheek_tint_shift": (1.06, 0.80, 0.74),
        "forehead_tint_shift": (1.03, 1.02, 0.91),
        "jaw_tint_shift": (0.92, 0.93, 0.94),
    },
    "VI": {
        "label": "Fitzpatrick VI (Deep Ebony / Rich Black)",
        "linear_albedo": (0.092, 0.050, 0.020),
        "ref_linear_raw": (0.090, 0.050, 0.020),
        "subsurface_radius": (0.367, 0.137, 0.068),
        "subsurface_scale": 0.0013,
        "roughness_base": 0.40,   # Balanced warm bronze highlight without sweaty gloss or dusty ash!
        "coat_weight": 0.44,
        "coat_roughness": 0.22,
        "sheen_weight": 0.035,
        "cavity_darkening": 0.26,
        "upper_lip_linear": (0.046, 0.025, 0.020),
        "lower_lip_linear": (0.135, 0.068, 0.052),  # Lower lip lower melanocyte density!
        "cheek_tint_shift": (1.08, 0.82, 0.76),
        "forehead_tint_shift": (1.04, 1.02, 0.92),
        "jaw_tint_shift": (0.93, 0.94, 0.95),
    },
}


def chromophore_albedo(
    melanin: float,
    pheomelanin: float = 0.25,
    oxy_hb: float = 0.35,
    deoxy_hb: float = 0.05,
) -> Tuple[float, float, float]:
    """Biophysical Beer-Lambert skin diffuse albedo model."""
    base = (0.84, 0.66, 0.58)
    k_eu = (1.0, 1.22, 1.50)
    k_ph = (0.15, 0.45, 1.00)
    k_ox = (0.02, 0.55, 0.45)
    k_dx = (0.60, 0.45, 0.25)
    out = []
    for i in range(3):
        absorb = melanin * k_eu[i] + pheomelanin * k_ph[i] + oxy_hb * k_ox[i] + deoxy_hb * k_dx[i]
        out.append(round(max(0.020, min(0.810, base[i] * math.exp(-absorb))), 4))
    return tuple(out)


def compute_ita_degrees(rgb_lin: Tuple[float, float, float]) -> Tuple[float, str]:
    """Compute Individual Typology Angle (ITA deg) & map to Fitzpatrick I-VI tier."""
    srgb = cc.linear_to_srgb_255(rgb_lin)
    L, _, b_lab = cc.srgb_to_lab(srgb)
    # Standard Chardon / Del Bino dermatology ITA formula
    ita = math.degrees(math.atan2(L - 50.0, max(1e-4, b_lab) if abs(b_lab) < 1e-4 else b_lab))
    if ita > 55.0:
        tier = "I"
    elif ita > 41.0:
        tier = "II"
    elif ita > 28.0:
        tier = "III"
    elif ita > 10.0:
        tier = "IV"
    elif ita > -30.0:
        tier = "V"
    else:
        tier = "VI"
    return round(ita, 2), tier


# ==============================================================================
# 2. REFERENCE IMAGE WHITE-BALANCE, SPECULAR/SHADOW REJECTION & DE-LIGHTING
# ==============================================================================
def _crop_norm_box(arr, roi_norm: Tuple[float, float, float, float]):
    h, w = arr.shape[:2]
    x0 = max(0, min(w - 1, int(round(roi_norm[0] * w))))
    y0 = max(0, min(h - 1, int(round(roi_norm[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(roi_norm[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(roi_norm[3] * h))))
    return arr[y0:y1, x0:x1]


def _arr_srgb_to_linear(arr_u8):
    import numpy as np
    f = arr_u8.astype(np.float32) / 255.0
    return np.where(f <= 0.04045, f / 12.92, np.power(np.clip((f + 0.055) / 1.055, 0.0, 1.0), 2.4))


def estimate_illuminant_gains(
    img_path: str,
    neutral_roi: Optional[Tuple[float, float, float, float]] = None,
    max_shift: float = 0.22,
) -> Tuple[float, float, float]:
    """
    Estimate conservative linear RGB white-balance gains from a neutral cue (sclera mid-highlight,
    white collar, gray card). Clamped strictly within [1 - max_shift, 1 + max_shift] so natural
    skin warmth or artistic grading isn't over-stripped.
    """
    import numpy as np

    if neutral_roi is None:
        return (1.0, 1.0, 1.0)
    im = cc._load_rgb_array(img_path)
    patch = _crop_norm_box(im, neutral_roi)
    lin = _arr_srgb_to_linear(patch).reshape(-1, 3)
    lum = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    # Keep upper 40%-85% clean mid-highlights (ignore clipped 1.0 specular & shadowed corners)
    lo, hi = np.percentile(lum, [40.0, 88.0])
    mask = (lum >= lo) & (lum <= hi) & (np.max(lin, axis=1) < 0.96)
    pts = lin[mask] if np.any(mask) else lin
    med = np.median(pts, axis=0)
    gray = float(np.mean(med))
    gains = []
    for ch in range(3):
        g = gray / max(1e-4, float(med[ch]))
        gains.append(round(max(1.0 - max_shift, min(1.0 + max_shift, g)), 4))
    return tuple(gains)


def sample_delit_skin_region(
    img_path: str,
    roi_norm: Tuple[float, float, float, float],
    wb_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spec_drop: float = 0.15,
    shadow_drop: float = 0.20,
    min_saturation: float = 0.08,
) -> dict:
    """
    De-lit diffuse skin sampler for ANY reference photo:
      1. Linearizes sRGB ROI and applies neutral illuminant balance `wb_gains`.
      2. Rejects top `spec_drop` high-luminance / low-saturation specular glare pixels.
      3. Rejects bottom `shadow_drop` cast shadow / pore / hair pixels.
      4. Enforces warm skin hue sanity (`R >= B * 0.95`, `R >= G * 0.88`) and Pointer's Gamut limits.
      5. Returns clean trimmed median diffuse linear RGB, ITA deg, and fitted Fitzpatrick tier!
    """
    import numpy as np

    im = cc._load_rgb_array(img_path)
    patch_u8 = _crop_norm_box(im, roi_norm)
    lin = _arr_srgb_to_linear(patch_u8).reshape(-1, 3)
    lin_wb = np.clip(lin * np.array(wb_gains, dtype=np.float32)[None, :], 0.001, 1.0)

    lum = 0.2126 * lin_wb[:, 0] + 0.7152 * lin_wb[:, 1] + 0.0722 * lin_wb[:, 2]
    c_max = np.max(lin_wb, axis=1)
    c_min = np.min(lin_wb, axis=1)
    sat = np.where(c_max > 1e-4, (c_max - c_min) / c_max, 0.0)

    lo_lum = np.percentile(lum, shadow_drop * 100.0)
    hi_lum = np.percentile(lum, (1.0 - spec_drop) * 100.0)
    # Specular rejection + shadow rejection + skin chromophore sanity check (R > B)
    valid = (
        (lum >= lo_lum)
        & (lum <= hi_lum)
        & (sat >= min_saturation)
        & (lin_wb[:, 0] >= lin_wb[:, 2] * 1.02)
        & (lin_wb[:, 0] >= lin_wb[:, 1] * 0.88)
    )
    # Fallback gently if extreme moody lighting leaves too few valid samples
    if int(np.sum(valid)) < max(8, int(0.10 * len(lum))):
        valid = (lum >= lo_lum) & (lum <= hi_lum)
    pts = lin_wb[valid] if np.any(valid) else lin_wb
    med = np.median(pts, axis=0)

    # Convert illuminated diffuse reading toward neutral AgX diffuse reflectance prior & clamp Pointer's Gamut
    # Brighter readings attenuate slightly stronger against studio key boost; deep readings preserve low-lum chroma!
    lum_med = float(0.2126 * med[0] + 0.7152 * med[1] + 0.0722 * med[2])
    att = 0.76 + 0.16 * max(0.0, min(1.0, 1.0 - lum_med / 0.55))
    albedo_lin = tuple(round(max(0.020, min(0.810, float(med[ch]) * att)), 4) for ch in range(3))
    ita, tier = compute_ita_degrees(albedo_lin)
    return {
        "measured_delit_linear": tuple(round(float(c), 4) for c in med),
        "albedo_linear": albedo_lin,
        "albedo_srgb_255": cc.linear_to_srgb_255(albedo_lin),
        "ita_deg": ita,
        "fitzpatrick": tier,
        "kept_fraction": round(float(len(pts)) / max(1, len(lum)), 3),
    }


def _gaussian_blur_2d_np(arr2d, radius: float):
    """Pure NumPy separable 3-pass box blur fallback approximating Gaussian blur when PIL is unavailable."""
    import numpy as np

    r = max(1, int(round(radius * 0.6)))
    cur = arr2d.astype(np.float32)
    for _ in range(3):
        # Horizontal pass
        pad_x = np.pad(cur, ((0, 0), (r, r)), mode="edge")
        cs_x = np.cumsum(np.pad(pad_x, ((0, 0), (1, 0))), axis=1)
        cur = (cs_x[:, 2 * r + 1:] - cs_x[:, :cur.shape[1]]) / float(2 * r + 1)
        # Vertical pass
        pad_y = np.pad(cur, ((r, r), (0, 0)), mode="edge")
        cs_y = np.cumsum(np.pad(pad_y, ((1, 0), (0, 0))), axis=0)
        cur = (cs_y[2 * r + 1:, :] - cs_y[:cur.shape[0], :]) / float(2 * r + 1)
    return cur


def save_rgb_u8(arr_u8, out_path: str) -> str:
    """Save (H, W, 3) uint8 RGB image via Pillow if available, else Blender bpy.data.images."""
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(arr_u8).save(out_path)
    except ImportError:
        import bpy
        h, w = arr_u8.shape[:2]
        im = bpy.data.images.new("_tmp_save_img", width=w, height=h, alpha=True)
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = np.flipud(arr_u8).astype(np.float32) / 255.0
        im.pixels.foreach_set(rgba.ravel())
        im.filepath_raw = out_path
        im.file_format = "PNG"
        im.save()
        bpy.data.images.remove(im)
    return out_path


def frequency_separate_delight(
    img_path: str,
    out_path: str,
    sigma: float = 28.0,
    flatten_strength: float = 0.65,
) -> str:
    """
    2D Single-Photo De-Lighting via Frequency Separation:
      - Linearizes RGB pixels.
      - Extracts smooth Low-Frequency (LF) lighting gradient via Gaussian blur (PIL or pure NumPy).
      - Divides out LF directional key/shadow falloff softly (`flatten_strength`) while keeping
        100% of High-Frequency (HF) pores, freckles, beauty marks, and dermal grain intact!
      - Encodes clean texture sRGB back to `out_path` for UV texture projection.
    """
    import numpy as np

    im_u8 = cc._load_rgb_array(img_path)
    lin = _arr_srgb_to_linear(im_u8)
    try:
        from PIL import Image, ImageFilter
        use_pil = True
    except ImportError:
        use_pil = False

    lf_ch = []
    for ch in range(3):
        if use_pil:
            ch_im = Image.fromarray((np.clip(lin[:, :, ch], 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
            bl = np.asarray(ch_im.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
        else:
            bl = _gaussian_blur_2d_np(lin[:, :, ch], sigma)
        lf_ch.append(bl)
    lf = np.stack(lf_ch, axis=-1)
    lf_lum = 0.2126 * lf[:, :, 0] + 0.7152 * lf[:, :, 1] + 0.0722 * lf[:, :, 2]
    target_lum = float(np.median(lf_lum))
    gain = target_lum / np.maximum(0.04, lf_lum)
    gain = np.clip(gain, 0.55, 1.85)
    blend_gain = (1.0 - flatten_strength) + flatten_strength * gain
    delit_lin = np.clip(lin * blend_gain[:, :, None], 0.020, 0.810)
    delit_srgb = np.where(
        delit_lin <= 0.0031308,
        delit_lin * 12.92,
        1.055 * np.power(delit_lin, 1.0 / 2.4) - 0.055,
    )
    out_u8 = (np.clip(delit_srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return save_rgb_u8(out_u8, out_path)


def profile_from_fitzpatrick(
    tier: str = "III",
    style: str = "photoreal",
    linear_override: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Create a complete calibrated SkinProfile from a Fitzpatrick I-VI tier and style target."""
    if tier not in FITZPATRICK_PRESETS:
        raise ValueError(f"Unknown Fitzpatrick tier '{tier}' (expected I..VI)")
    p = dict(FITZPATRICK_PRESETS[tier])
    base = tuple(linear_override) if linear_override is not None else p["linear_albedo"]
    ita, detected = compute_ita_degrees(base)

    # Style calibration matrix (Photoreal Digital Human vs Semi-Real vs Stylized Animation)
    if style in ("stylized", "stylized_animation"):
        sss_w = 0.70
        sss_scale = min(0.0032, max(0.0014, p["subsurface_scale"] * 0.82))
        anis = 0.50
        pore_strength = 0.08
        bump_dist = 0.00010
        sheen_w = max(0.14, p["sheen_weight"] * 2.5)
        coat_mult = 0.65
    elif style == "semi_real":
        sss_w = 0.88
        sss_scale = p["subsurface_scale"] * 0.92
        anis = 0.68
        pore_strength = 0.45
        bump_dist = 0.00016
        sheen_w = max(0.08, p["sheen_weight"] * 1.5)
        coat_mult = 0.82
    else:  # photoreal
        sss_w = 1.0
        sss_scale = p["subsurface_scale"]
        anis = 0.80
        pore_strength = 1.0
        bump_dist = 0.00018
        sheen_w = p["sheen_weight"]
        coat_mult = 1.0

    # Derive zonal colors cleanly around `base` using facial zone shift ratios
    def _scale_rgb(rgb, mul):
        return tuple(round(max(0.015, min(0.82, rgb[i] * mul[i])), 4) for i in range(3))

    return {
        "style": style,
        "fitzpatrick": tier,
        "measured_ita_tier": detected,
        "ita_deg": ita,
        "base_linear": base,
        "forehead_linear": _scale_rgb(base, p["forehead_tint_shift"]),
        "cheek_linear": _scale_rgb(base, p["cheek_tint_shift"]),
        "jaw_linear": _scale_rgb(base, p["jaw_tint_shift"]),
        "upper_lip_linear": p["upper_lip_linear"],
        "lower_lip_linear": p["lower_lip_linear"],
        "subsurface_weight": sss_w,
        "subsurface_radius": p["subsurface_radius"],
        "subsurface_scale": round(sss_scale, 5),
        "subsurface_ior": 1.40,
        "subsurface_anisotropy": anis,
        "roughness_base": p["roughness_base"],
        "coat_weight": round(p["coat_weight"] * coat_mult, 3),
        "coat_roughness": p["coat_roughness"],
        "coat_ior": 1.45,
        "sheen_weight": round(sheen_w, 3),
        "sheen_roughness": 0.50,
        "cavity_darkening": round(p["cavity_darkening"] * (0.35 + 0.65 * pore_strength), 3),
        "pore_strength": pore_strength,
        "bump_distance": bump_dist,
    }


def profile_reference_portrait(
    img_path: str,
    base_skin_roi: Tuple[float, float, float, float],
    zone_rois: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    neutral_roi: Optional[Tuple[float, float, float, float]] = None,
    style: str = "photoreal",
) -> dict:
    """
    Extract a complete de-lit SkinProfile directly from a reference portrait image.
      - `base_skin_roi`: Clean mid-cheek / temple normalized bounding box (x0, y0, x1, y1).
      - `zone_rois`: Optional overrides for `forehead`, `cheek`, `jaw`, `upper_lip`, `lower_lip`.
      - `neutral_roi`: Optional sclera / white-collar neutral reference for illuminant balance.
    """
    wb = estimate_illuminant_gains(img_path, neutral_roi=neutral_roi)
    base_info = sample_delit_skin_region(img_path, base_skin_roi, wb_gains=wb)
    prof = profile_from_fitzpatrick(
        tier=base_info["fitzpatrick"],
        style=style,
        linear_override=base_info["albedo_linear"],
    )
    prof["wb_gains"] = wb
    prof["base_sample_stats"] = base_info

    for z_name, prof_key in (
        ("forehead", "forehead_linear"),
        ("cheek", "cheek_linear"),
        ("jaw", "jaw_linear"),
        ("upper_lip", "upper_lip_linear"),
        ("lower_lip", "lower_lip_linear"),
    ):
        if zone_rois and z_name in zone_rois:
            z_info = sample_delit_skin_region(
                img_path,
                zone_rois[z_name],
                wb_gains=wb,
                spec_drop=0.12,
                shadow_drop=0.18,
            )
            prof[prof_key] = z_info["albedo_linear"]
    return prof


# ==============================================================================
# 3. BLENDER 5.2 UNIVERSAL MULTI-ZONE SKIN SHADER BUILDER (`RANDOM_WALK_SKIN`)
# ==============================================================================
def build_universal_skin_shader(
    name: str,
    profile: dict,
    center_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    head_height_m: float = 0.22,
    front_axis: str = "-Y",
    allow_vertex_zones: bool = True,
):
    """
    Build a calibrated Blender 5.2 Principled BSDF skin shader driven by `profile`:
      - Subsurface Method = RANDOM_WALK_SKIN, Subsurface IOR = 1.40, calibrated Radius & Scale
      - 3-Zone Facial Spatial Mixing (Forehead gold-ochre, Cheeks/Nose rose flush, Jaw cool undertone)
      - Dual-Lobe Specular: Primary IOR 1.40 + Cavity Pore Occlusion + Sebum Coat IOR 1.45 (T-zone biased)
      - Multi-scale mm Pores (0.9mm & 0.4mm Voronoi F1) + Micro-Relief (0.7mm DISTANCE_TO_EDGE) + Bump
      - Soft Vellus Peach-Fuzz Sheen tinted subtly by skin Base Color!
    Optional vertex-color attributes supported cleanly if painted: `ZoneFlush`, `TZoneOil`, `SSSWeightMask`.
    """
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.subsurface_method = "RANDOM_WALK_SKIN"
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Configure calibrated SSS physics
    bsdf.inputs["Subsurface Weight"].default_value = profile["subsurface_weight"]
    bsdf.inputs["Subsurface Radius"].default_value = profile["subsurface_radius"]
    bsdf.inputs["Subsurface Scale"].default_value = profile["subsurface_scale"]
    bsdf.inputs["Subsurface IOR"].default_value = profile["subsurface_ior"]
    bsdf.inputs["Subsurface Anisotropy"].default_value = profile["subsurface_anisotropy"]
    bsdf.inputs["IOR"].default_value = 1.40
    bsdf.inputs["Coat IOR"].default_value = profile["coat_ior"]
    bsdf.inputs["Coat Roughness"].default_value = profile["coat_roughness"]
    bsdf.inputs["Sheen Weight"].default_value = profile["sheen_weight"]
    bsdf.inputs["Sheen Roughness"].default_value = profile["sheen_roughness"]

    # Normalized local head coordinates around center_xyz (scaled by head_height_m)
    tex = n.new("ShaderNodeTexCoord")
    sub_c = n.new("ShaderNodeVectorMath")
    sub_c.operation = "SUBTRACT"
    sub_c.inputs[1].default_value = center_xyz
    L(tex.outputs["Object"], sub_c.inputs[0])

    norm_c = n.new("ShaderNodeVectorMath")
    norm_c.operation = "SCALE"
    norm_c.inputs["Scale"].default_value = 1.0 / max(0.05, head_height_m)
    L(sub_c.outputs["Vector"], norm_c.inputs[0])

    sep = n.new("ShaderNodeSeparateXYZ")
    L(norm_c.outputs["Vector"], sep.inputs["Vector"])

    # --- Spatial Facial Zonal Masks (works automatically on any metric head bust!) ---
    # Forehead: z > 0.14; Jaw/Muzzle: z < -0.14; Cheek/Nose hemoglobin flush: |z| < 0.14 & front hemisphere
    z_fore = n.new("ShaderNodeMapRange")
    z_fore.clamp = True
    z_fore.inputs["From Min"].default_value = 0.08
    z_fore.inputs["From Max"].default_value = 0.28
    L(sep.outputs["Z"], z_fore.inputs["Value"])

    z_jaw = n.new("ShaderNodeMapRange")
    z_jaw.clamp = True
    z_jaw.inputs["From Min"].default_value = -0.10
    z_jaw.inputs["From Max"].default_value = -0.32
    L(sep.outputs["Z"], z_jaw.inputs["Value"])

    # Cheek flush distance near (+/-0.18, -0.35, -0.02) & nose tip (0, -0.45, -0.02) in normalized units
    front_sign = -1.0 if front_axis == "-Y" else 1.0
    flush_out = None
    for fx, fy, fz, rad_f, amp in (
        (-0.20, 0.34 * front_sign, -0.02, 0.24, 0.65),
        (0.20, 0.34 * front_sign, -0.02, 0.24, 0.65),
        (0.0, 0.44 * front_sign, -0.02, 0.16, 0.72),
    ):
        d_f = n.new("ShaderNodeVectorMath")
        d_f.operation = "DISTANCE"
        d_f.inputs[1].default_value = (fx, fy, fz)
        L(norm_c.outputs["Vector"], d_f.inputs[0])
        mr_f = n.new("ShaderNodeMapRange")
        mr_f.clamp = True
        mr_f.inputs["From Min"].default_value = 0.02
        mr_f.inputs["From Max"].default_value = rad_f
        mr_f.inputs["To Min"].default_value = amp
        mr_f.inputs["To Max"].default_value = 0.0
        L(d_f.outputs["Value"], mr_f.inputs["Value"])
        if flush_out is None:
            flush_out = mr_f.outputs["Result"]
        else:
            mx_f = n.new("ShaderNodeMath")
            mx_f.operation = "MAXIMUM"
            L(flush_out, mx_f.inputs[0])
            L(mr_f.outputs["Result"], mx_f.inputs[1])
            flush_out = mx_f.outputs["Value"]

    # T-Zone Oiliness Mask (Forehead center + Nose ridge + Chin tip)
    abs_x = n.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    L(sep.outputs["X"], abs_x.inputs[0])
    t_center = n.new("ShaderNodeMapRange")
    t_center.clamp = True
    t_center.inputs["From Min"].default_value = 0.04
    t_center.inputs["From Max"].default_value = 0.24
    t_center.inputs["To Min"].default_value = 1.0
    t_center.inputs["To Max"].default_value = 0.32
    L(abs_x.outputs["Value"], t_center.inputs["Value"])

    # --- Mix 3-Zone Albedos + Mottling + Micro-Pore Cavity Darkening ---
    base_rgba = (*profile["base_linear"], 1.0)
    fore_rgba = (*profile["forehead_linear"], 1.0)
    cheek_rgba = (*profile["cheek_linear"], 1.0)
    jaw_rgba = (*profile["jaw_linear"], 1.0)

    mix_1 = n.new("ShaderNodeMix")
    mix_1.data_type = "RGBA"
    L(z_fore.outputs["Result"], mix_1.inputs[0])
    mix_1.inputs[6].default_value = base_rgba
    mix_1.inputs[7].default_value = fore_rgba

    mix_2 = n.new("ShaderNodeMix")
    mix_2.data_type = "RGBA"
    L(flush_out, mix_2.inputs[0])
    L(mix_1.outputs[2], mix_2.inputs[6])
    mix_2.inputs[7].default_value = cheek_rgba

    mix_3 = n.new("ShaderNodeMix")
    mix_3.data_type = "RGBA"
    L(z_jaw.outputs["Result"], mix_3.inputs[0])
    L(mix_2.outputs[2], mix_3.inputs[6])
    mix_3.inputs[7].default_value = jaw_rgba

    # Organic subcutaneous hemoglobin/melanin mottling (`~12 mm` noise)
    ns_mot = n.new("ShaderNodeTexNoise")
    ns_mot.inputs["Scale"].default_value = 75.0
    ns_mot.inputs["Detail"].default_value = 4.0
    L(tex.outputs["Object"], ns_mot.inputs["Vector"])

    mr_mot = n.new("ShaderNodeMapRange")
    mr_mot.clamp = True
    mr_mot.inputs["From Min"].default_value = 0.25
    mr_mot.inputs["From Max"].default_value = 0.75
    mr_mot.inputs["To Min"].default_value = 0.94
    mr_mot.inputs["To Max"].default_value = 1.05
    L(ns_mot.outputs["Fac"], mr_mot.inputs["Value"])

    mul_mot = n.new("ShaderNodeMix")
    mul_mot.data_type = "RGBA"
    mul_mot.blend_type = "MULTIPLY"
    mul_mot.inputs[0].default_value = 0.60
    L(mix_3.outputs[2], mul_mot.inputs[6])
    L(mr_mot.outputs["Result"], mul_mot.inputs[7])

    # --- Multi-Scale Physical Pores (`0.9mm` + `0.4mm` Voronoi) & Dermis Relief ---
    ns_warp = n.new("ShaderNodeTexNoise")
    ns_warp.inputs["Scale"].default_value = 250.0
    ns_warp.inputs["Detail"].default_value = 2.0
    L(tex.outputs["Object"], ns_warp.inputs["Vector"])

    warp_vec = n.new("ShaderNodeMix")
    warp_vec.data_type = "VECTOR"
    warp_vec.inputs[0].default_value = 0.0012
    L(tex.outputs["Object"], warp_vec.inputs[4])
    L(ns_warp.outputs["Color"], warp_vec.inputs[5])

    # Primary pores (~0.9mm cell scale -> Scale ~ 1100 per meter) & micro-facets (~0.7mm DISTANCE_TO_EDGE)
    v_pore = n.new("ShaderNodeTexVoronoi")
    v_pore.feature = "F1"
    v_pore.inputs["Scale"].default_value = 1100.0
    L(warp_vec.outputs[1], v_pore.inputs["Vector"])

    # Convert F1 Distance to soft circular pit factor (1.0 at pit center -> 0.0 flat skin)
    pore_pit = n.new("ShaderNodeMapRange")
    pore_pit.clamp = True
    pore_pit.inputs["From Min"].default_value = 0.0
    pore_pit.inputs["From Max"].default_value = 0.24
    pore_pit.inputs["To Min"].default_value = 1.0
    pore_pit.inputs["To Max"].default_value = 0.0
    L(v_pore.outputs["Distance"], pore_pit.inputs["Value"])

    v_facet = n.new("ShaderNodeTexVoronoi")
    v_facet.feature = "DISTANCE_TO_EDGE"
    v_facet.inputs["Scale"].default_value = 1400.0
    L(warp_vec.outputs[1], v_facet.inputs["Vector"])

    # Dermis fine grain (~0.4mm noise)
    ns_derm = n.new("ShaderNodeTexNoise")
    ns_derm.inputs["Scale"].default_value = 2400.0
    ns_derm.inputs["Detail"].default_value = 4.0
    ns_derm.inputs["Roughness"].default_value = 0.55
    L(tex.outputs["Object"], ns_derm.inputs["Vector"])

    # Combined Height Map (for Bump & Cavity Occlusion)
    h_neg_pore = n.new("ShaderNodeMath")
    h_neg_pore.operation = "MULTIPLY_ADD"
    h_neg_pore.inputs[1].default_value = -0.55 * profile["pore_strength"]
    L(pore_pit.outputs["Result"], h_neg_pore.inputs[0])
    L(v_facet.outputs["Distance"], h_neg_pore.inputs[2])

    h_comb = n.new("ShaderNodeMath")
    h_comb.operation = "MULTIPLY_ADD"
    h_comb.inputs[1].default_value = 0.35
    L(ns_derm.outputs["Fac"], h_comb.inputs[0])
    L(h_neg_pore.outputs["Value"], h_comb.inputs[2])

    bmp = n.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.35 * (0.25 + 0.75 * profile["pore_strength"])
    bmp.inputs["Distance"].default_value = profile["bump_distance"]
    L(h_comb.outputs["Value"], bmp.inputs["Height"])
    L(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    L(bmp.outputs["Normal"], bsdf.inputs["Coat Normal"])

    # Cavity Darkening on Base Color (kill flat plastic pores cleanly!)
    cav_dark = n.new("ShaderNodeMapRange")
    cav_dark.clamp = True
    cav_dark.inputs["From Min"].default_value = 0.0
    cav_dark.inputs["From Max"].default_value = 1.0
    cav_dark.inputs["To Min"].default_value = 1.0
    cav_dark.inputs["To Max"].default_value = max(0.55, 1.0 - profile["cavity_darkening"])
    L(pore_pit.outputs["Result"], cav_dark.inputs["Value"])

    mul_cav = n.new("ShaderNodeMix")
    mul_cav.data_type = "RGBA"
    mul_cav.blend_type = "MULTIPLY"
    mul_cav.inputs[0].default_value = 1.0
    L(mul_mot.outputs[2], mul_cav.inputs[6])
    L(cav_dark.outputs["Result"], mul_cav.inputs[7])
    L(mul_cav.outputs[2], bsdf.inputs["Base Color"])

    # Specular Pore Occlusion (`Specular IOR Level` drops inside pore pits so zero sparkle dots!)
    spec_occ = n.new("ShaderNodeMapRange")
    spec_occ.clamp = True
    spec_occ.inputs["From Min"].default_value = 0.0
    spec_occ.inputs["From Max"].default_value = 1.0
    spec_occ.inputs["To Min"].default_value = 0.50
    spec_occ.inputs["To Max"].default_value = 0.22
    L(pore_pit.outputs["Result"], spec_occ.inputs["Value"])
    L(spec_occ.outputs["Result"], bsdf.inputs["Specular IOR Level"])

    # Primary Roughness: base + subtle T-zone drop + pore pit roughness rise
    r_zone = n.new("ShaderNodeMapRange")
    r_zone.clamp = True
    r_zone.inputs["From Min"].default_value = 0.32
    r_zone.inputs["From Max"].default_value = 1.0
    r_zone.inputs["To Min"].default_value = min(0.65, profile["roughness_base"] + 0.04)
    r_zone.inputs["To Max"].default_value = max(0.26, profile["roughness_base"] - 0.06)
    L(t_center.outputs["Result"], r_zone.inputs["Value"])

    r_final = n.new("ShaderNodeMath")
    r_final.operation = "MULTIPLY_ADD"
    r_final.inputs[1].default_value = 0.14 * profile["pore_strength"]
    L(pore_pit.outputs["Result"], r_final.inputs[0])
    L(r_zone.outputs["Result"], r_final.inputs[2])
    L(r_final.outputs["Value"], bsdf.inputs["Roughness"])

    # Secondary Oil/Sebum Coat Weight scaled smoothly by T-zone oil mask
    cw_mul = n.new("ShaderNodeMath")
    cw_mul.operation = "MULTIPLY"
    cw_mul.inputs[1].default_value = profile["coat_weight"]
    L(t_center.outputs["Result"], cw_mul.inputs[0])
    L(cw_mul.outputs["Value"], bsdf.inputs["Coat Weight"])

    # Soft Vellus Peach-Fuzz Sheen Tint (50% neutral white + 50% Base Color)
    sh_tint = tuple(round(0.5 * 1.0 + 0.5 * c, 4) for c in profile["base_linear"]) + (1.0,)
    bsdf.inputs["Sheen Tint"].default_value = sh_tint

    return mat
