"""Pure Computer Vision, Surface Relief & SciPy Solver for Universal Exact 3D Reconstruction.

AI AGENT READ-ONLY LOCK — DO NOT MODIFY THIS FILE:
- This Python module is a STRICTLY READ-ONLY core library.
- AI agents MUST NEVER edit, overwrite, refactor, append to, or delete this file.
- Invoke it via CLI or import its functions from a separate external project script;
  place all project-specific logic in your own external script.

Zero-Rush Exact-Match Mandate:
- Never rush or settle for approximate silhouette contours, landmark alignment, or relief maps.
- Iterate until extracted geometry, depth, and surface relief match the target 1:1.

Runs in Python 3.10+ with OpenCV, MediaPipe, ONNXRuntime, SciPy (RBFInterpolator), NumPy, and Pillow.
Configure local vision model weights path via the `VISION_MODELS_DIR` environment variable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator

MODELS_DIR = os.environ.get("VISION_MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
FACE_TASK_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
DEPTH_ONNX_PATH = os.path.join(MODELS_DIR, "depth_v2_small", "onnx", "model.onnx")
RMBG_ONNX_PATH = os.path.join(MODELS_DIR, "rmbg14", "onnx", "model_quantized.onnx")

# ---------------------------------------------------------------------------
# Coordinate-Verified MediaPipe 478 <-> MPFB2 Basemesh (body vgroup) Mapping
# MPFB2 faces -Y, +X is character's left (image right), +Z is up.
# ---------------------------------------------------------------------------
LANDMARK_PAIRS: List[Tuple[str, int, int, float]] = [
    # (name, mediapipe_idx, mpfb2_body_vid, pnp_rigidity_weight)
    ("nose_tip", 4, 294, 1.0),
    ("nose_dorsum", 197, 5063, 1.0),
    ("nose_bridge_top", 168, 133, 1.0),
    ("nose_subnasale", 2, 317, 1.0),
    ("nose_ala_L", 279, 7057, 1.0),
    ("nose_ala_R", 49, 288, 1.0),
    ("eye_L_inner", 362, 6853, 1.0),
    ("eye_L_outer", 263, 7859, 1.0),
    ("eye_L_top", 386, 7774, 0.5),
    ("eye_L_bot", 374, 7813, 0.6),
    ("eye_R_inner", 133, 69, 1.0),
    ("eye_R_outer", 33, 1167, 1.0),
    ("eye_R_top", 159, 1082, 0.5),
    ("eye_R_bot", 145, 1121, 0.6),
    ("brow_L_inner", 285, 6957, 0.8),
    ("brow_L_mid", 295, 6979, 0.7),
    ("brow_L_outer", 300, 7024, 0.8),
    ("brow_R_inner", 55, 181, 0.8),
    ("brow_R_mid", 65, 204, 0.7),
    ("brow_R_outer", 70, 255, 0.8),
    ("mouth_center_top", 0, 509, 0.5),
    ("mouth_center_bot", 17, 495, 0.5),
    ("upper_lip_L", 267, 11955, 0.4),
    ("upper_lip_R", 37, 5353, 0.4),
    ("lower_lip_mid", 18, 492, 0.4),
    ("mouth_corner_L", 291, 10372, 0.6),
    ("mouth_corner_R", 61, 3704, 0.6),
    ("chin_tip", 152, 5225, 0.9),
    ("jaw_L", 397, 7461, 0.7),
    ("jaw_R", 172, 741, 0.7),
    ("cheek_L", 330, 11681, 0.7),
    ("cheek_R", 101, 5065, 0.7),
    ("cheek_lower_L", 432, 11896, 0.5),
    ("cheek_lower_R", 212, 5290, 0.5),
    ("zygoma_L", 352, 11794, 0.8),
    ("zygoma_R", 123, 5182, 0.8),
    ("forehead_mid", 151, 244, 0.8),
    ("forehead_top", 10, 1015, 0.6),
    ("forehead_L", 251, 7012, 0.7),
    ("forehead_R", 21, 241, 0.7),
    ("temple_L", 389, 11810, 0.7),
    ("temple_R", 162, 5199, 0.7),
]

FACE_OVAL_MP = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
EYE_R_OVAL_MP = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L_OVAL_MP = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def _srgb_to_linear_np(arr01: np.ndarray) -> np.ndarray:
    a = np.clip(arr01, 0.0, 1.0)
    return np.where(a <= 0.04045, a / 12.92, np.power((a + 0.055) / 1.055, 2.4))


def _linear_to_srgb_np(arr01: np.ndarray) -> np.ndarray:
    a = np.clip(arr01, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)


def run_rmbg14(rgb_u8: np.ndarray) -> np.ndarray:
    """Return crisp float32 [0..1] foreground alpha matte at original image resolution."""
    import onnxruntime as ort

    H, W = rgb_u8.shape[:2]
    sess = ort.InferenceSession(RMBG_ONNX_PATH, providers=["CPUExecutionProvider"])
    inp = cv2.resize(rgb_u8, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    inp = np.transpose(inp - 0.5, (2, 0, 1))[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0, 0]
    omin, omax = float(out.min()), float(out.max())
    norm = np.clip((out - omin) / max(1e-6, omax - omin), 0.0, 1.0)
    return cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def run_depth_v2_small(rgb_u8: np.ndarray) -> np.ndarray:
    """Return normalized [0..1] Depth-Anything-V2-Small map (1.0 = nearest camera)."""
    import onnxruntime as ort

    H, W = rgb_u8.shape[:2]
    sess = ort.InferenceSession(DEPTH_ONNX_PATH, providers=["CPUExecutionProvider"])
    inp = cv2.resize(rgb_u8, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = np.transpose((inp - mean) / std, (2, 0, 1))[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
    dmin, dmax = float(out.min()), float(out.max())
    dnorm = np.clip((out - dmin) / max(1e-6, dmax - dmin), 0.0, 1.0)
    return cv2.resize(dnorm, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _detect_face_landmarks(img_path: str):
    """Detect 478 MediaPipe landmarks + 52 blendshapes + facial transform matrix."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_TASK_PATH),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.25,
    )
    with vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp.Image.create_from_file(img_path))
    if not res.face_landmarks:
        raise RuntimeError(f"No human face detected in reference image: {img_path}")
    lms = [[float(p.x), float(p.y), float(p.z)] for p in res.face_landmarks[0]]
    bs = {}
    if res.face_blendshapes:
        bs = {b.category_name: round(float(b.score), 4) for b in res.face_blendshapes[0]}
    xf = res.facial_transformation_matrixes[0].tolist() if res.facial_transformation_matrixes else None
    return lms, bs, xf


def _build_portrait_masks(
    H: int, W: int, pts_px: np.ndarray, matte: np.ndarray, rgb_u8: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build cleanly separated masks: full face+ear skin region, eye mask, hair halo, and collar/coat mask."""
    face_oval = np.zeros((H, W), dtype=np.float32)
    oval_poly = np.round(pts_px[FACE_OVAL_MP, :2]).astype(np.int32)
    cv2.fillConvexPoly(face_oval, oval_poly, 1.0)

    chin = pts_px[152, :2]
    jaw_l = pts_px[397, :2]
    jaw_r = pts_px[172, :2]
    zy_l = pts_px[352, :2]
    zy_r = pts_px[123, :2]
    f_top = pts_px[10, :2]
    face_h = float(np.linalg.norm(chin - f_top))
    face_w = max(20.0, float(np.linalg.norm(zy_l - zy_r)))

    # Expanded full-head + tall forehead + ears + deep throat contour so ears, jaw, hairline & neck all receive crisp UV detail!
    ear_pad = 0.28 * face_w
    exp_poly = np.array([
        f_top + np.array([0.0, -0.25 * face_h]),
        pts_px[251, :2] + np.array([0.20 * face_w, -0.20 * face_h]),
        zy_l + np.array([ear_pad, -0.08 * face_h]),
        jaw_l + np.array([ear_pad * 0.92, 0.06 * face_h]),
        jaw_l + np.array([0.14 * face_h, 0.48 * face_h]),
        chin + np.array([0.0, 0.68 * face_h]),
        jaw_r + np.array([-0.14 * face_h, 0.48 * face_h]),
        jaw_r + np.array([-ear_pad * 0.92, 0.06 * face_h]),
        zy_r + np.array([-ear_pad, -0.08 * face_h]),
        pts_px[21, :2] + np.array([-0.20 * face_w, -0.20 * face_h]),
    ], dtype=np.int32)
    head_skin_poly = np.zeros((H, W), dtype=np.float32)
    cv2.fillConvexPoly(head_skin_poly, exp_poly, 1.0)

    luma_img = rgb_u8.mean(axis=2) / 255.0
    core_luma = float(np.percentile(luma_img[face_oval > 0.5], 88)) if np.any(face_oval > 0.5) else 0.65

    skin_region = np.maximum(face_oval, head_skin_poly * matte)
    skin_region = cv2.GaussianBlur(skin_region, (0, 0), max(5.0, W * 0.009))

    eye_mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(eye_mask, [np.round(pts_px[EYE_R_OVAL_MP, :2]).astype(np.int32)], 1.0)
    cv2.fillPoly(eye_mask, [np.round(pts_px[EYE_L_OVAL_MP, :2]).astype(np.int32)], 1.0)
    eye_mask = cv2.dilate(eye_mask, np.ones((5, 5), np.uint8))

    # Preserve real temple, above-ear & hairline wisps cleanly by eroding interior face oval instead of bloating it outward!
    er_k = max(3, int(face_h * 0.014) | 1)
    face_inner_hair_guard = cv2.GaussianBlur(
        cv2.erode(face_oval, np.ones((er_k, er_k), np.uint8)),
        (0, 0),
        max(5.0, face_h * 0.022),
    )
    y_grid = np.arange(H, dtype=np.float32)[:, None]
    jaw_start = float(jaw_l[1] * 0.5 + jaw_r[1] * 0.5)
    jaw_end = float(chin[1] + 0.08 * face_h)
    vert_taper = np.clip(1.0 - (y_grid - jaw_start) / max(8.0, jaw_end - jaw_start), 0.0, 1.0)
    hair_halo = np.clip(matte * vert_taper - face_inner_hair_guard * 0.98, 0.0, 1.0)

    # Coat/Collar starts smoothly from chest upward into collar (blur luma so salt-and-pepper tweed stays solid!)
    top_bound_y = int(max(0, min(pts_px[4, 1], pts_px[101, 1], pts_px[330, 1])))
    luma_coat = cv2.GaussianBlur(luma_img.astype(np.float32), (0, 0), max(4.0, W * 0.005))
    coat_cand = (matte > 0.28) & (face_oval < 0.25) & (luma_coat < max(0.76, core_luma * 1.08))
    coat_cand[:top_bound_y, :] = False
    # Sever thin ear-hair bridges while keeping shoulder weave silhouette unbroken
    coat_clean = cv2.morphologyEx(coat_cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))

    n_c, lbls_c, stats_c, _ = cv2.connectedComponentsWithStats(coat_clean, connectivity=8)
    col_bin = np.zeros((H, W), dtype=np.uint8)
    chest_floor = int(min(H - 1, chin[1] + 0.12 * face_h))
    chest_lbls = set(np.unique(lbls_c[chest_floor:, :])) - {0}
    for cid in chest_lbls:
        if stats_c[cid, cv2.CC_STAT_AREA] > 200:
            col_bin |= (lbls_c == cid).astype(np.uint8)
    if not np.any(col_bin):
        col_bin = (matte > 0.32).astype(np.uint8)
        col_bin[:int(chin[1] - 0.02 * face_h), :] = 0
    col_bin = cv2.morphologyEx(col_bin, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    collar_mask = np.clip(cv2.GaussianBlur(col_bin.astype(np.float32), (0, 0), 2.0), 0.0, 1.0) * matte

    # Keep chin_y for hair_halo cleanly above curved coat collar
    hair_halo = np.clip(hair_halo - collar_mask, 0.0, 1.0)

    return skin_region, eye_mask, hair_halo, collar_mask


def _delight_and_colorize_portrait(
    rgb_u8: np.ndarray,
    face_oval_mask: np.ndarray,
    eye_mask: np.ndarray,
    pts_px: np.ndarray,
    colorize_monochrome: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict]:
    """
    Frequency-separated delighting inside face/skin + optional B&W chromophore tint.
    When `colorize_monochrome=False` (default for exact B&W look-matching), preserves clean
    neutral monochrome tonal fidelity matching reference skin values without orange/peach shift.
    """
    H, W = rgb_u8.shape[:2]
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    face_pix = face_oval_mask > 0.5
    med_sat = float(np.median(hsv[face_pix, 1])) / 255.0 if np.any(face_pix) else 0.0
    is_monochrome = med_sat < 0.08

    lin = _srgb_to_linear_np(rgb_u8.astype(np.float32) / 255.0)
    lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

    # True macro-only illumination field (`sigma >= 13% width`) preserves ALL forehead wrinkles, eye bags, nasolabial folds & age spots!
    sigma_lf = max(62.0, W * 0.128)
    lf_lum = cv2.GaussianBlur(lum, (0, 0), sigma_lf)

    mid_mask = (face_oval_mask > 0.6) & (eye_mask < 0.1) & (lum > 0.015)
    if np.any(mid_mask):
        face_p15 = float(np.percentile(lum[mid_mask], 15))
        face_p50 = float(np.percentile(lum[mid_mask], 50))
        face_p88 = float(np.percentile(lum[mid_mask], 88))
    else:
        face_p15, face_p50, face_p88 = 0.05, 0.20, 0.42
    photo_contrast_ratio = float(np.clip(face_p88 / max(0.015, face_p15), 1.8, 12.0))

    # Universal Photo Key-Light Direction Estimator (`left/right cheek & forehead/chin quadrant luminance`)
    def _quad_lum(pt_list: List[int]) -> float:
        vals = []
        r_q = max(8, int(W * 0.025))
        for pi in pt_list:
            px, py = int(np.clip(round(pts_px[pi, 0]), r_q, W - r_q - 1)), int(np.clip(round(pts_px[pi, 1]), r_q, H - r_q - 1))
            vals.append(float(np.median(lf_lum[py - r_q : py + r_q + 1, px - r_q : px + r_q + 1])))
        return float(np.mean(vals)) if vals else face_p50

    lum_L_side = _quad_lum([330, 352, 432])  # +X in Blender (viewer's right cheek)
    lum_R_side = _quad_lum([101, 123, 212])  # -X in Blender (viewer's left cheek)
    lum_top = _quad_lum([151, 10])
    lum_bot = _quad_lum([152])
    key_side_x = float(np.clip(((lum_L_side - lum_R_side) / max(0.05, lum_L_side + lum_R_side)) * 3.6, -1.0, 1.0))
    key_elev_z = float(np.clip(0.55 + ((lum_top - lum_bot) / max(0.05, lum_top + lum_bot)) * 1.4, 0.20, 1.15))

    # Balanced portrait delighting (`flatten_strength=0.44`: preserves rich tonal depth & under-eye/wrinkle shadows without dirty soot!)
    target_lum = max(0.14, min(0.25, face_p50 * 0.95))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    face_w_px = max(24.0, float(np.linalg.norm(pts_px[352, :2] - pts_px[123, :2])))

    # Soft orbital & upper-lip cushion prevents pitch-black eye caves and cleans 2D mustache hair imprints off skin
    eye_orbit_soft = cv2.GaussianBlur(cv2.dilate(eye_mask, np.ones((21, 21), np.uint8)), (0, 0), max(9.0, 0.05 * face_w_px))
    m_cx = 0.5 * (pts_px[61, 0] + pts_px[291, 0])
    m_cy = 0.46 * pts_px[0, 1] + 0.54 * pts_px[2, 1]
    m_rx = max(16.0, 0.72 * float(np.linalg.norm(pts_px[291, :2] - pts_px[61, :2])))
    m_ry = max(8.0, 0.95 * float(abs(pts_px[0, 1] - pts_px[2, 1])))
    must_cushion = np.exp(-(((xx - m_cx) / m_rx) ** 2 + ((yy - m_cy) / m_ry) ** 2) * 1.15).astype(np.float32)
    # Outer whisker-bleed guard on lateral mouth-corner cheek extensions (`eliminates Turn R white whisker smear!`)
    lat_whisker_x = np.clip((np.abs(xx - m_cx) - 0.36 * m_rx) / (0.62 * m_rx), 0.0, 1.0)
    whisker_cheek_guard = np.maximum(
        must_cushion * 0.72,
        np.exp(-(((xx - m_cx) / (m_rx * 1.42)) ** 2 + ((yy - (m_cy + 0.04 * face_w_px)) / (m_ry * 1.50)) ** 2) * 1.20) * lat_whisker_x,
    ).astype(np.float32)

    lf_socket = cv2.GaussianBlur(lum, (0, 0), max(14.0, W * 0.024))
    eff_lf = lf_lum * (1.0 - 0.35 * eye_orbit_soft) + np.minimum(lf_lum, lf_socket) * (0.35 * eye_orbit_soft)
    gain = np.clip(target_lum / np.maximum(0.032, eff_lf), 0.80, 2.05)
    flatten_strength = 0.44 + 0.14 * eye_orbit_soft
    blend_gain = (1.0 - flatten_strength) + flatten_strength * gain

    # Crisp 4-scale facial relief bands (`unblurred so fine pores, crow's feet, forehead furrows & nasolabial folds stay razor-sharp!`)
    lf_pore = cv2.GaussianBlur(lum, (0, 0), max(2.2, W * 0.0038))
    lf_fine = cv2.GaussianBlur(lum, (0, 0), max(6.5, W * 0.011))
    lf_mid = cv2.GaussianBlur(lum, (0, 0), max(18.0, W * 0.028))
    lf_deep = cv2.GaussianBlur(lum, (0, 0), max(38.0, W * 0.056))
    band_hi = (lum - lf_pore) / np.maximum(0.05, lf_pore)
    band_mid = (lf_pore - lf_fine) / np.maximum(0.06, lf_fine)
    band_lo = (lf_fine - lf_mid) / np.maximum(0.07, lf_mid)
    band_deep = (lf_mid - lf_deep) / np.maximum(0.08, lf_deep)

    # Soft highlight rolloff above 1.24 * target_lum prevents studio cheek glare baking as white patches in side views!
    hi_cap = max(0.18, target_lum * 1.24)

    def _soft_shoulder(x_lin: np.ndarray) -> np.ndarray:
        over = np.maximum(0.0, x_lin - hi_cap)
        return np.where(x_lin > hi_cap, hi_cap + over / (1.0 + over * 3.6), x_lin)

    cheek_l = pts_px[330, :2]
    cheek_r = pts_px[101, :2]
    nose_t = pts_px[4, :2]
    rad_warm = max(20.0, float(np.linalg.norm(cheek_l - cheek_r)) * 0.50)
    d_cl = np.hypot(xx - cheek_l[0], yy - cheek_l[1])
    d_cr = np.hypot(xx - cheek_r[0], yy - cheek_r[1])
    d_nt = np.hypot(xx - nose_t[0], yy - nose_t[1]) * 1.05
    warm_cheek_nose = np.clip(1.0 - np.minimum(np.minimum(d_cl, d_cr), d_nt) / rad_warm, 0.0, 1.0) ** 1.3

    # Clean 2D photo whisker white bleed inside `whisker_cheek_guard` toward smooth local cheek tone (eliminates Turn R cheek smear!)
    clean_cheek_lum = cv2.GaussianBlur(lum * blend_gain, (0, 0), max(20.0, 0.11 * face_w_px))
    w_clean_fac = np.clip(whisker_cheek_guard * 0.85, 0.0, 0.88)
    base_delit_lum = _soft_shoulder(lum * blend_gain)
    base_delit_lum = base_delit_lum * (1.0 - w_clean_fac) + clean_cheek_lum * w_clean_fac

    if is_monochrome and colorize_monochrome:
        delit_gray = np.clip(base_delit_lum, 0.018, 0.74)
        norm_val = np.power(np.clip(delit_gray / max(0.16, target_lum), 0.05, 1.45), 0.90)

        fore_y = pts_px[168, 1]
        jaw_y = pts_px[0, 1]
        fore_w = np.clip((fore_y - yy) / max(20.0, 0.14 * H), 0.0, 1.0)
        jaw_w = np.clip((yy - jaw_y) / max(20.0, 0.18 * H), 0.0, 1.0)

        base_tint = np.array([0.485, 0.295, 0.215], dtype=np.float32)
        fore_tint = np.array([0.510, 0.325, 0.220], dtype=np.float32)
        flush_tint = np.array([0.555, 0.250, 0.180], dtype=np.float32)
        jaw_tint = np.array([0.415, 0.268, 0.208], dtype=np.float32)

        tint = base_tint[None, None, :] * (1.0 - 0.38 * fore_w[:, :, None]) + fore_tint[None, None, :] * (0.38 * fore_w[:, :, None])
        tint = tint * (1.0 - 0.65 * warm_cheek_nose[:, :, None]) + flush_tint[None, None, :] * (0.65 * warm_cheek_nose[:, :, None])
        tint = tint * (1.0 - 0.42 * jaw_w[:, :, None]) + jaw_tint[None, None, :] * (0.42 * jaw_w[:, :, None])
        tint = tint * (1.0 - 0.85 * eye_mask[:, :, None]) + np.array([0.48, 0.46, 0.44], dtype=np.float32)[None, None, :] * (0.85 * eye_mask[:, :, None])
        delit_lin = np.clip(tint * norm_val[:, :, None], 0.016, 0.76)
    elif is_monochrome:
        # Clean, rich silver-platinum monochrome albedo + subtle dermal warmth on cheeks & nose (ZERO dirty soot!)
        delit_gray = np.clip(base_delit_lum, 0.015, 0.74)
        w_tint = warm_cheek_nose * (1.0 - eye_mask) * np.clip((delit_gray - 0.04) / 0.14, 0.0, 1.0)
        r_ch = delit_gray * (1.0 + 0.045 * w_tint)
        g_ch = delit_gray * (1.0 + 0.011 * w_tint)
        b_ch = delit_gray * (1.0 - 0.042 * w_tint)
        delit_lin = np.clip(np.stack([r_ch, g_ch, b_ch], axis=2), 0.015, 0.76)
    else:
        raw_c = _soft_shoulder(lin * blend_gain[:, :, None])
        clean_c = cv2.GaussianBlur(raw_c, (0, 0), max(20.0, 0.11 * face_w_px))
        delit_lin = np.clip(raw_c * (1.0 - w_clean_fac[:, :, None]) + clean_c * w_clean_fac[:, :, None], 0.016, 0.78)

    delit_srgb_u8 = (_linear_to_srgb_np(delit_lin) * 255.0).round().clip(0, 255).astype(np.uint8)

    # Smooth-yet-detailed 3-Band 16-bit Facial Relief Heightmap (`soft natural wrinkles + pores, zero gritty cheek craters or whisker bumps!`)
    band_hi_clean = cv2.GaussianBlur(band_hi, (0, 0), 0.75)
    band_mid_clean = cv2.GaussianBlur(band_mid, (0, 0), 0.85)
    hf_raw = cv2.GaussianBlur(0.24 * band_hi_clean + 0.34 * band_mid_clean + 0.30 * band_lo + 0.18 * band_deep, (0, 0), 0.70)

    # Measure universal photo skin texture / wrinkle energy (`0=young smooth, 1=deeply weathered/aged`)
    skin_tex_energy = float(np.clip(np.std(band_mid[mid_mask]) * 5.2, 0.08, 1.0)) if np.any(mid_mask) else 0.45

    eye_inner = cv2.erode(eye_mask, np.ones((5, 5), np.uint8))
    chin_y_rim = float(pts_px[152, 1])
    chin_taper = np.clip((chin_y_rim - 0.025 * H - yy) / max(10.0, 0.05 * H), 0.35, 1.0).astype(np.float32)
    face_weight_soft = cv2.GaussianBlur(face_oval_mask.astype(np.float32), (0, 0), max(14.0, W * 0.024)) * chin_taper
    hf = np.clip(
        (hf_raw * face_weight_soft * (1.0 - 0.34 * eye_orbit_soft) * (1.0 - 0.90 * must_cushion) * (1.0 - 0.86 * whisker_cheek_guard)) * (1.0 - eye_inner)
        + must_cushion * 0.04,
        -0.42,
        0.42,
    )
    h01 = np.clip(0.5 + hf * 0.78, 0.0, 1.0)
    height_16 = (h01 * 65535.0).round().astype(np.uint16)

    # Universal Anatomical Zonal Sebum / Roughness Map (`matte cheeks/jaw + subtle T-zone oil on nose/forehead/lips/eyelid rim!`)
    nose_tip_p = pts_px[4, :2]
    nose_br_p = pts_px[197, :2]
    fore_c_p = pts_px[151, :2]
    lip_lo_p = pts_px[17, :2]
    tzone_nose = np.maximum(
        np.exp(-(((xx - nose_tip_p[0]) / (0.13 * face_w_px)) ** 2 + ((yy - nose_tip_p[1]) / (0.14 * face_w_px)) ** 2)),
        np.exp(-(((xx - nose_br_p[0]) / (0.09 * face_w_px)) ** 2 + ((yy - nose_br_p[1]) / (0.16 * face_w_px)) ** 2)) * 0.85,
    )
    tzone_fore = np.exp(-(((xx - fore_c_p[0]) / (0.28 * face_w_px)) ** 2 + ((yy - fore_c_p[1]) / (0.14 * face_w_px)) ** 2)) * 0.72
    lip_moist = np.exp(-(((xx - lip_lo_p[0]) / (0.22 * face_w_px)) ** 2 + ((yy - lip_lo_p[1]) / (0.08 * face_w_px)) ** 2)) * 0.80
    eye_rim_w = np.clip(cv2.dilate(eye_mask, np.ones((7, 7), np.uint8)) - eye_inner, 0.0, 1.0) * 0.75
    sebum_w = np.clip(np.maximum.reduce([tzone_nose, tzone_fore, lip_moist, eye_rim_w]), 0.0, 1.0)

    # Universal 3-Channel Anatomical Skin Zone Mask (`R=Nose/Inner-Cheek Pores, G=Forehead/Neck Horizontal Lines, B=Delicate Eyes/Lips`)
    zone_nose_pore = np.clip(
        np.maximum.reduce([
            tzone_nose,
            np.exp(-(((xx - cheek_l[0]) / (0.19 * face_w_px)) ** 2 + ((yy - cheek_l[1]) / (0.16 * face_w_px)) ** 2)) * 0.78,
            np.exp(-(((xx - cheek_r[0]) / (0.19 * face_w_px)) ** 2 + ((yy - cheek_r[1]) / (0.16 * face_w_px)) ** 2)) * 0.78,
        ]),
        0.0,
        1.0,
    )
    brow_y_mid = 0.5 * (pts_px[295, 1] + pts_px[65, 1])
    chin_y_px = float(pts_px[152, 1])
    zone_fore_neck = np.clip(
        np.maximum(
            np.clip((brow_y_mid - yy) / max(12.0, 0.08 * face_w_px), 0.0, 1.0),
            np.clip((yy - (chin_y_px + 0.02 * face_w_px)) / max(14.0, 0.12 * face_w_px), 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    eye_delic = cv2.GaussianBlur(cv2.dilate(eye_mask, np.ones((19, 19), np.uint8)), (0, 0), max(8.0, 0.05 * face_w_px))
    zone_delicate = np.clip(np.maximum(eye_delic, lip_moist * 1.15), 0.0, 1.0)
    skin_zones_rgb = np.dstack([zone_nose_pore, zone_fore_neck, zone_delicate]).clip(0.0, 1.0)
    skin_zones_u8 = (skin_zones_rgb * 255.0).round().astype(np.uint8)

    cavity = np.clip(0.5 - h01, 0.0, 0.5) * 2.0
    # Base matte dermal roughness ~0.72..0.88; subtle natural T-zone sebum lowers ridges ONLY to ~0.36..0.52 (cavities stay matte!)
    rough01 = np.clip(0.74 - 0.34 * sebum_w * (1.0 - 0.75 * cavity) + 0.16 * cavity, 0.32, 0.90)
    rough_8 = (rough01 * 255.0).round().astype(np.uint8)

    metrics = {
        "median_face_sat": round(med_sat, 4),
        "is_monochrome": bool(is_monochrome),
        "colorize_monochrome": bool(colorize_monochrome),
        "target_skin_lum_lin": round(target_lum, 4),
        "skin_texture_energy": round(skin_tex_energy, 4),
        "key_side_x": round(key_side_x, 4),
        "key_elev_z": round(key_elev_z, 4),
        "photo_contrast_ratio": round(photo_contrast_ratio, 4),
        "photo_face_p15_lin": round(face_p15, 4),
        "photo_face_p50_lin": round(face_p50, 4),
        "photo_face_p88_lin": round(face_p88, 4),
    }
    return delit_srgb_u8, height_16, rough_8, skin_zones_u8, is_monochrome, metrics


def analyze_portrait(ref_path: str, out_dir: str, colorize_monochrome: bool = False) -> Dict:
    """Analyze ANY human portrait photo and write intermediate maps + coat/hair grids + portrait_analysis.json."""
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(ref_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {ref_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    matte = run_rmbg14(rgb)
    depth01 = run_depth_v2_small(rgb)
    lms_norm, blendshapes, xf = _detect_face_landmarks(ref_path)
    pts_px = np.array([[lm[0] * W, lm[1] * H, lm[2] * W] for lm in lms_norm], dtype=np.float32)

    skin_region, eye_mask, hair_halo, collar_mask = _build_portrait_masks(H, W, pts_px, matte, rgb)
    delit_srgb, height_16, rough_8, skin_zones_u8, is_mono, color_metrics = _delight_and_colorize_portrait(
        rgb, skin_region, eye_mask, pts_px, colorize_monochrome=colorize_monochrome
    )

    # Clean hair halo stats (drives automatic procedural hair crown lift & side fluff!)
    hair_ys, hair_xs = np.where(hair_halo > 0.35)
    face_xs = pts_px[FACE_OVAL_MP, 0]
    face_ys = pts_px[FACE_OVAL_MP, 1]
    face_w = max(20.0, float(face_xs.max() - face_xs.min()))
    face_h = max(20.0, float(face_ys.max() - face_ys.min()))
    face_cx = float(0.5 * (face_xs.min() + face_xs.max()))
    if len(hair_xs) > 50:
        crown_lift_ratio = float(max(0.0, (face_ys.min() - hair_ys.min()) / face_h))
        side_l_flare = float(max(0.0, (hair_xs.max() - face_xs.max()) / face_w))
        side_r_flare = float(max(0.0, (face_xs.min() - hair_xs.min()) / face_w))
        hair_luma_vals = rgb[hair_ys, hair_xs].mean(axis=1) / 255.0
        hair_med_luma = float(np.percentile(hair_luma_vals, 70))
        hair_med_rgb = (rgb[hair_ys, hair_xs].mean(axis=0) / 255.0).tolist()
    else:
        crown_lift_ratio, side_l_flare, side_r_flare, hair_med_luma = 0.14, 0.08, 0.08, 0.18
        hair_med_rgb = [0.22, 0.20, 0.19]

    # Sample collar/clothing average brightness, contrast & linear quantiles for seamless 360° shader match
    col_ys, col_xs = np.where(collar_mask > 0.45)
    if len(col_xs) > 50:
        col_vals = rgb[col_ys, col_xs].mean(axis=1) / 255.0
        collar_luma = float(np.median(col_vals))
        collar_std = float(np.std(col_vals))
        collar_rgb = (np.median(rgb[col_ys, col_xs], axis=0) / 255.0).tolist()
        c_lin_s = _srgb_to_linear_np(rgb[col_ys, col_xs].astype(np.float32) / 255.0)
        c_luma_s = 0.2126 * c_lin_s[:, 0] + 0.7152 * c_lin_s[:, 1] + 0.0722 * c_lin_s[:, 2]
        collar_p15_lin = float(np.percentile(c_luma_s, 15))
        collar_p50_lin = float(np.percentile(c_luma_s, 50))
        collar_p85_lin = float(np.percentile(c_luma_s, 85))
    else:
        collar_luma, collar_std, collar_rgb = 0.28, 0.08, [0.25, 0.24, 0.22]
        collar_p15_lin, collar_p50_lin, collar_p85_lin = 0.025, 0.065, 0.145

    # Sample eyeball sclera & iris color roughly around detected iris
    iris_rgb = [0.10, 0.07, 0.05]
    ir_x, ir_y = int(round(pts_px[468, 0])), int(round(pts_px[468, 1]))
    if 4 <= ir_x < W - 4 and 4 <= ir_y < H - 4:
        patch = rgb[ir_y - 3 : ir_y + 4, ir_x - 3 : ir_x + 4]
        v = float(np.percentile(patch.mean(axis=2), 25)) / 255.0
        iris_rgb = [round(v * 0.95, 3), round(v * 0.72, 3), round(v * 0.52, 3)] if is_mono else (
            patch.mean(axis=(0, 1)) / 255.0
        ).round(3).tolist()

    # Extract crisp torso/collar photo-guided 3D geometry & PBR maps (`zero manual cylinder fallback!`)
    coat_bin = collar_mask > 0.35
    cys, cxs = np.where(coat_bin)
    torso_coat = None
    coat_albedo_p = os.path.join(out_dir, "coat_albedo.png")
    coat_normal_p = os.path.join(out_dir, "coat_normal.png")
    coat_rough_p = os.path.join(out_dir, "coat_rough.png")
    if len(cxs) > 100:
        cx0, cx1 = int(cxs.min()), int(cxs.max())
        cy0, cy1 = int(cys.min()), int(cys.max())
        # Organic portrait exclusion mask: excludes true skin oval/neck & hair without chopping rectangular coat patches!
        coat_face_u8 = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(coat_face_u8, np.round(pts_px[FACE_OVAL_MP, :2]).astype(np.int32), 1)
        coat_face_excl = (
            (cv2.dilate(coat_face_u8, np.ones((7, 7), np.uint8)) > 0)
            | (skin_region > 0.34)
        )

        coat_lin = _srgb_to_linear_np(rgb.astype(np.float32) / 255.0)
        c_lum_lin = 0.2126 * coat_lin[:, :, 0] + 0.7152 * coat_lin[:, :, 1] + 0.0722 * coat_lin[:, :, 2]
        base_coat_mask = (collar_mask > 0.42) & (~coat_face_excl) & (hair_halo < 0.40) & (matte > 0.36)
        if np.sum(base_coat_mask) > 250:
            base_coat_mask = cv2.erode(base_coat_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if np.sum(base_coat_mask) < 100:
            base_coat_mask = collar_mask > 0.38
        c_target_lin = float(np.median(c_lum_lin[base_coat_mask])) if np.any(base_coat_mask) else 0.055

        # Macro illumination delighting (ONLY lift occluded macro shadow pockets — preserve individual dark tweed yarns!)
        sig_meso = max(12.0, face_w * 0.085)
        sig_macro = max(16.0, face_w * 0.105)
        c_lf_meso = cv2.GaussianBlur(np.where(base_coat_mask, c_lum_lin, c_target_lin), (0, 0), sig_meso)
        c_lf_macro = cv2.GaussianBlur(np.where(base_coat_mask, c_lum_lin, c_target_lin), (0, 0), sig_macro)
        c_gain = np.clip((c_target_lin / np.maximum(0.018, c_lf_meso)) ** 0.65, 0.85, 1.85)
        delit_coat_lin = np.clip(coat_lin * (0.28 + 0.72 * c_gain[:, :, None]), 0.008, 0.90)
        # Blend macro shadow fill strictly via blurred macro-illumination (`zero per-pixel dark yarn honeycomb rings!`)
        dead_w = np.clip((c_target_lin * 0.26 - c_lf_macro) / max(1e-4, c_target_lin * 0.22), 0.0, 0.55)
        ys_v, xs_v = np.where(base_coat_mask & (c_lf_macro > c_target_lin * 0.70))
        if len(ys_v) > 200:
            y0_t, y1_t = int(np.percentile(ys_v, 30)), int(np.percentile(ys_v, 72))
            x0_t, x1_t = int(np.percentile(xs_v, 22)), int(np.percentile(xs_v, 78))
            if y1_t > y0_t + 32 and x1_t > x0_t + 32:
                patch_lin = delit_coat_lin[y0_t:y1_t, x0_t:x1_t]
                mir_x = np.concatenate([patch_lin, patch_lin[:, ::-1]], axis=1)
                mir_xy = np.concatenate([mir_x, mir_x[::-1, :]], axis=0)
                ph, pw_t = mir_xy.shape[:2]
                tile_lin = np.tile(mir_xy, (H // ph + 1, W // pw_t + 1, 1))[:H, :W]
                delit_coat_lin = delit_coat_lin * (1.0 - dead_w[:, :, None]) + tile_lin * dead_w[:, :, None]
        else:
            tile_lin = np.full_like(delit_coat_lin, c_target_lin)

        delit_coat_srgb = (_linear_to_srgb_np(delit_coat_lin) * 255.0).round().clip(0, 255).astype(np.uint8)
        tile_srgb = (_linear_to_srgb_np(np.clip(tile_lin, 0.008, 0.90)) * 255.0).round().clip(0, 255).astype(np.uint8)
        if np.any(base_coat_mask):
            dl_v = 0.2126 * delit_coat_lin[:, :, 0] + 0.7152 * delit_coat_lin[:, :, 1] + 0.0722 * delit_coat_lin[:, :, 2]
            dv_vals = dl_v[base_coat_mask]
            collar_p15_lin = float(np.percentile(dv_vals, 15))
            collar_p50_lin = float(np.percentile(dv_vals, 50))
            collar_p85_lin = float(np.percentile(dv_vals, 85))
        # Outside garment silhouette fill seamlessly with mirrored tweed cloth + smooth feathered alpha
        clean_coat = np.where(base_coat_mask[:, :, None], delit_coat_srgb, tile_srgb)
        if is_mono and not colorize_monochrome:
            c_gray = clean_coat.mean(axis=2, keepdims=True)
            clean_coat = np.repeat(c_gray, 3, axis=2).clip(0, 255).astype(np.uint8)
        coat_alpha = np.clip(cv2.GaussianBlur(base_coat_mask.astype(np.float32), (0, 0), 5.5), 0.0, 1.0)
        coat_rgba = np.dstack([clean_coat, (coat_alpha * 255.0).round().astype(np.uint8)])
        Image.fromarray(coat_rgba, mode="RGBA").save(coat_albedo_p)

        cd_vals = depth01[coat_bin]
        cd_lo, cd_hi = float(np.percentile(cd_vals, 2)), float(np.percentile(cd_vals, 98))
        cd_norm = np.clip((depth01 - cd_lo) / max(1e-4, cd_hi - cd_lo), 0.0, 1.0)
        cd_norm = cv2.GaussianBlur(cd_norm, (0, 0), 2.5)
        c_lum = clean_coat.mean(axis=2).astype(np.float32) / 255.0
        c_hf = c_lum - cv2.GaussianBlur(c_lum, (0, 0), 3.0)
        coat_norm_rgb = _depth_to_tangent_normal(cd_norm + 0.14 * c_hf, strength=2.4)
        Image.fromarray(coat_norm_rgb).save(coat_normal_p)
        coat_r = np.clip(0.82 - 0.55 * np.abs(c_hf), 0.45, 0.94)
        cv2.imwrite(coat_rough_p, (coat_r * 255.0).round().astype(np.uint8))

        # Export clean smoothed photo silhouette contour (filtering foreground hair overlap cleanly!)
        g_res = 96
        c_d_crop = cv2.resize(cd_norm[cy0 : cy1 + 1, cx0 : cx1 + 1], (g_res, g_res), interpolation=cv2.INTER_AREA)
        c_m_crop = cv2.resize(collar_mask[cy0 : cy1 + 1, cx0 : cx1 + 1], (g_res, g_res), interpolation=cv2.INTER_AREA)

        chin_xy = pts_px[152, :2]
        top_prof_x = np.linspace(cx0, cx1, 25)
        top_y_rel = []
        col_mask_clean = (collar_mask > 0.35) & (hair_halo < 0.40)
        for sx in top_prof_x:
            ix = int(np.clip(round(sx), cx0, cx1))
            ys_col = np.where(col_mask_clean[:, ix])[0]
            # Keep realistic shoulder height strictly below jaw midpoint (`skip stray upward artifacts`)
            ys_valid = ys_col[ys_col >= int(chin_xy[1] - 0.22 * face_h)]
            if len(ys_valid):
                top_y_rel.append(round(float((ys_valid.min() - chin_xy[1]) / face_h), 4))
            else:
                top_y_rel.append(0.35)
        # Compute neck V-opening center & half-width + detect center lapel/placket button center universally
        neck_band_y0 = int(max(0, min(H - 1, chin_xy[1] + 0.04 * face_h)))
        neck_band_y1 = int(max(neck_band_y0 + 2, min(H, chin_xy[1] + 0.38 * face_h)))
        open_gap = (skin_region[neck_band_y0:neck_band_y1, :] > 0.35) & (~col_mask_clean[neck_band_y0:neck_band_y1, :])
        og_ys, og_xs = np.where(open_gap)
        v_open_hw = float(np.percentile(np.abs(og_xs - chin_xy[0]), 85) / face_w) if len(og_xs) > 15 else 0.24
        v_open_cx_norm = float(np.median(og_xs) / W) if len(og_xs) > 15 else float(chin_xy[0] / W)
        v_drop_r = float((og_ys.max() + neck_band_y0 - chin_xy[1]) / face_h) if len(og_ys) > 15 else 0.32

        # Detect placket/collar button inside central upper chest ROI using dark smoothed minima within coat mask
        btn_x0 = int(max(cx0, chin_xy[0] - 0.36 * face_w))
        btn_x1 = int(min(cx1, chin_xy[0] + 0.42 * face_w))
        btn_y0 = int(max(cy0, chin_xy[1] + 0.22 * face_h))
        btn_y1 = int(min(cy1 - max(6, int(0.04 * H)), chin_xy[1] + 0.78 * face_h))
        button_norm = None
        if btn_x1 > btn_x0 + 16 and btn_y1 > btn_y0 + 16:
            patch_b = clean_coat[btn_y0:btn_y1, btn_x0:btn_x1].mean(axis=2).astype(np.float32) / 255.0
            cm_b = col_mask_clean[btn_y0:btn_y1, btn_x0:btn_x1]
            # Look for dark compact button disk relative to local chest wool luminance (`blur r ~ 3% face_w vs local 14%`)
            loc_bg = cv2.GaussianBlur(patch_b, (0, 0), max(12.0, 0.14 * face_w))
            btn_sm = cv2.GaussianBlur(patch_b, (0, 0), max(3.5, 0.032 * face_w))
            score_b = np.where(cm_b, loc_bg - btn_sm, -1.0)
            by_i, bx_i = np.unravel_index(np.argmax(score_b), score_b.shape)
            if float(score_b[by_i, bx_i]) > 0.06:
                button_norm = [round((btn_x0 + bx_i) / W, 4), round((btn_y0 + by_i) / H, 4)]

        torso_coat = {
            "bbox": [cx0, cy0, cx1, cy1],
            "grid_res": g_res,
            "grid_depth": np.round(c_d_crop, 4).tolist(),
            "grid_matte": np.round(c_m_crop, 4).tolist(),
            "albedo": coat_albedo_p,
            "normal": coat_normal_p,
            "rough": coat_rough_p,
            "shoulder_top_y_rel": top_y_rel,
            "coat_width_ratio": round(float((cx1 - cx0) / face_w), 4),
            "coat_drop_ratio": round(float((cy1 - chin_xy[1]) / face_h), 4),
            "v_open_hw_ratio": round(float(np.clip(v_open_hw, 0.14, 0.48)), 4),
            "v_open_cx_norm": round(v_open_cx_norm, 4),
            "v_drop_ratio": round(float(np.clip(v_drop_r, 0.18, 0.65)), 4),
            "button_norm": button_norm,
        }

    # Universal Facial Hair (Mustache) Detector (`texture/edge energy & dark-vs-light contrast on upper lip vs cheek skin`)
    gray_u8 = rgb.mean(axis=2).astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3))
    mx0 = int(max(0, min(pts_px[61, 0], pts_px[291, 0]) - 0.14 * face_w))
    mx1 = int(min(W, max(pts_px[61, 0], pts_px[291, 0]) + 0.14 * face_w))
    my0 = int(max(0, pts_px[2, 1] - 0.02 * face_h))
    my1 = int(min(H, max(pts_px[0, 1], pts_px[37, 1], pts_px[267, 1]) + 0.05 * face_h))
    # Smooth reference cheek energy just below outer eyes
    ck_x = int(np.clip(round(pts_px[330, 0]), 8, W - 9))
    ck_y = int(np.clip(round(pts_px[330, 1]), 8, H - 9))
    cheek_lap = float(np.mean(lap[ck_y - 6 : ck_y + 7, ck_x - 6 : ck_x + 7]))
    if mx1 > mx0 + 8 and my1 > my0 + 6:
        lip_patch = gray_u8[my0:my1, mx0:mx1]
        lip_lap = float(np.mean(lap[my0:my1, mx0:mx1]))
        lip_std = float(np.std(lip_patch))
        # Also detect horizontal whisker span where edge variance stays active past mouth corners
        col_energy = np.mean(lap[my0:my1, mx0:mx1], axis=0)
        e_thresh = max(cheek_lap * 1.15, float(np.percentile(col_energy, 35)))
        active_cols = np.where(col_energy > e_thresh)[0]
        m_mouth_w = max(12.0, float(abs(pts_px[291, 0] - pts_px[61, 0])))
        span_px = float(active_cols.max() - active_cols.min()) if len(active_cols) > 4 else m_mouth_w * 1.1
        must_span_ratio = float(np.clip((span_px / m_mouth_w) * 0.62, 0.56, 0.90))
        tex_ratio = lip_lap / max(0.008, cheek_lap)
        must_density = float(np.clip((tex_ratio - 1.18) / 1.45 + (lip_std - 0.065) * 3.2, 0.0, 1.0))
        m_rgb_med = (np.median(rgb[my0:my1, mx0:mx1], axis=(0, 1)) / 255.0).tolist()
        m_luma_lo = float(np.percentile(lip_patch, 22))
        m_luma_hi = float(np.percentile(lip_patch, 82))
    else:
        must_density, must_span_ratio, m_luma_lo, m_luma_hi = 0.0, 0.62, 0.25, 0.65
        m_rgb_med = [0.40, 0.39, 0.37]

    facial_hair = {
        "mustache_density": round(must_density, 4),
        "mustache_span_ratio": round(must_span_ratio, 4),
        "mustache_droop_ratio": round(float(np.clip(0.55 + 0.45 * must_density, 0.45, 1.0)), 4),
        "luma_lo": round(m_luma_lo, 4),
        "luma_hi": round(m_luma_hi, 4),
        "med_rgb_srgb": [round(float(c), 4) for c in m_rgb_med],
    }

    # Inner face polygon for forehead hairline overlap & ribbon bridge guard
    face_inner = np.zeros((H, W), dtype=np.float32)
    oval_inner = np.round(pts_px[FACE_OVAL_MP, :2]).astype(np.int32)
    cv2.fillConvexPoly(face_inner, oval_inner, 1.0)

    # Hair silhouette mask (`bridges narrow graphic ribbon splits & removes isolated specks if monochrome!`)
    chroma_ribbon_u8 = np.zeros((H, W), dtype=np.uint8)
    if is_mono and not colorize_monochrome:
        hsv_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        chroma_ribbon_u8 = ((hsv_full[:, :, 1].astype(np.float32) / 255.0) > 0.14).astype(np.uint8)
        chroma_bg = cv2.GaussianBlur(cv2.dilate(chroma_ribbon_u8, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2.5)
        hair_halo = np.clip(hair_halo * (1.0 - chroma_bg * 0.96), 0.0, 1.0)
        # Bridge narrow horizontal graphic ribbon slots slicing across hair locks without bloating edges
        hb_raw = (hair_halo > 0.12).astype(np.uint8)
        hb_bridged = cv2.morphologyEx(hb_raw, cv2.MORPH_CLOSE, np.ones((25, 9), np.uint8))
        bridge_fill = (hb_bridged > 0) & (hb_raw == 0) & (chroma_ribbon_u8 > 0) & (face_inner < 0.35)
        hair_halo = np.where(bridge_fill, 0.72, hair_halo)

    # Keep ONLY contiguous hair islands connected near head cranium (removes floating ribbon fragments!)
    hb_u8 = (hair_halo > 0.10).astype(np.uint8)
    nh_c, hlbls, hstats, _ = cv2.connectedComponentsWithStats(hb_u8, connectivity=8)
    if nh_c > 1:
        areas = [int(hstats[i, cv2.CC_STAT_AREA]) for i in range(1, nh_c)]
        max_h_area = max(areas) if areas else 0
        keep_h = np.zeros((H, W), dtype=np.float32)
        for cid in range(1, nh_c):
            if int(hstats[cid, cv2.CC_STAT_AREA]) >= max(120, int(0.10 * max_h_area)):
                keep_h[hlbls == cid] = 1.0
        hair_halo = hair_halo * keep_h
    hair_halo = cv2.GaussianBlur(hair_halo, (0, 0), 1.8)

    ek = max(5, int(face_h * 0.04) | 1)
    face_inner_er = cv2.erode(face_inner, np.ones((ek, ek), np.uint8))
    face_inner_soft = cv2.GaussianBlur(face_inner_er, (0, 0), max(12.0, face_h * 0.055))

    # Mesh footprint covers true `hair_halo` + tiny 7px pad for semi-transparent fringe,
    # extending smoothly across upper forehead so inner boundary tucks unseen inside basemesh skin!
    crown_cut_y = int(pts_px[10, 1] + 0.20 * face_h)
    upper_scalp_pad = face_inner.copy()
    upper_scalp_pad[crown_cut_y:, :] = 0.0
    raw_hair_area = np.clip(np.maximum(hair_halo, upper_scalp_pad * 0.85) - (collar_mask > 0.35).astype(np.float32), 0.0, 1.0)
    hair_vol_mask = np.clip(cv2.dilate(raw_hair_area, np.ones((7, 7), np.uint8)), 0.0, 1.0)
    hair_vol_mask = np.clip(hair_vol_mask * (1.0 - np.clip((face_inner_soft - 0.65) / 0.35, 0.0, 1.0)), 0.0, 1.0)

    # Smooth continuous hair alpha across upper forehead hairline (prevents moat gap between brows & hair!)
    hair_plus_crown = np.maximum(hair_halo, upper_scalp_pad * matte * 0.90)
    wispy_hair_alpha = np.clip(hair_plus_crown * (1.0 - face_inner_soft * 0.95), 0.0, 1.0)

    hys, hxs = np.where(hair_vol_mask > 0.14)
    hair_sculpt = None
    hair_albedo_p = os.path.join(out_dir, "hair_albedo.png")
    hair_alpha_p = os.path.join(out_dir, "hair_alpha.png")
    hair_normal_p = os.path.join(out_dir, "hair_normal.png")
    hair_rough_p = os.path.join(out_dir, "hair_rough.png")
    if len(hxs) > 100:
        hx0, hx1 = int(hxs.min()), int(hxs.max())
        hy0, hy1 = int(hys.min()), int(hys.max())
        # Solid core hair pixels dilated 21px outward ensure translucent wispy tips never sample background color
        hair_core = (hair_halo > 0.48)
        hair_src = rgb.copy()
        if is_mono and not colorize_monochrome:
            hg = hair_src.mean(axis=2, keepdims=True)
            hair_src = np.repeat(hg, 3, axis=2).clip(0, 255).astype(np.uint8)
        fg_hair = np.where(hair_core[:, :, None], hair_src, 0)
        dil_hair = cv2.dilate(fg_hair, np.ones((21, 21), np.uint8))
        clean_hair_alb = np.where(hair_halo[:, :, None] > 0.38, hair_src, dil_hair)
        Image.fromarray(clean_hair_alb).save(hair_albedo_p)

        # True feathery wispy silhouette alpha directly from RMBG-1.4 (zero flat balloon halo!)
        soft_alpha = np.clip((wispy_hair_alpha - 0.04) / 0.72, 0.0, 1.0)
        cv2.imwrite(hair_alpha_p, (soft_alpha * 255.0).round().astype(np.uint8))

        hd_vals = depth01[hair_vol_mask > 0.25]
        hd_lo = float(np.percentile(hd_vals, 2)) if len(hd_vals) else 0.0
        hd_hi = float(np.percentile(hd_vals, 98)) if len(hd_vals) else 1.0
        hd_norm = np.clip((depth01 - hd_lo) / max(1e-4, hd_hi - hd_lo), 0.0, 1.0)
        hd_norm = cv2.GaussianBlur(hd_norm, (0, 0), 2.5)
        hlum = clean_hair_alb.mean(axis=2).astype(np.float32) / 255.0
        h_curl_hf = hlum - cv2.GaussianBlur(hlum, (0, 0), 7.0)
        hair_norm_rgb = _depth_to_tangent_normal(hd_norm * 0.45 + h_curl_hf * 0.55, strength=3.6)
        Image.fromarray(hair_norm_rgb).save(hair_normal_p)
        hair_r = np.clip(0.86 - 0.65 * np.clip(h_curl_hf, -0.35, 0.35), 0.54, 0.94)
        cv2.imwrite(hair_rough_p, (hair_r * 255.0).round().astype(np.uint8))

        hg_res = 96
        hd_crop = cv2.resize(hd_norm[hy0 : hy1 + 1, hx0 : hx1 + 1], (hg_res, hg_res), interpolation=cv2.INTER_AREA)
        hm_crop = cv2.resize(hair_vol_mask[hy0 : hy1 + 1, hx0 : hx1 + 1], (hg_res, hg_res), interpolation=cv2.INTER_AREA)
        hi_crop = cv2.resize(face_inner_soft[hy0 : hy1 + 1, hx0 : hx1 + 1], (hg_res, hg_res), interpolation=cv2.INTER_AREA)
        hair_sculpt = {
            "bbox": [hx0, hy0, hx1, hy1],
            "grid_res": hg_res,
            "grid_depth": np.round(hd_crop, 4).tolist(),
            "grid_matte": np.round(hm_crop, 4).tolist(),
            "grid_face_inner": np.round(hi_crop, 4).tolist(),
            "albedo": hair_albedo_p,
            "alpha": hair_alpha_p,
            "normal": hair_normal_p,
            "rough": hair_rough_p,
        }

    # Compact hair silhouette + cleaned luminance field (224x224) for 3D hair & whisker curves
    h_res = 224
    hair_halo_map = cv2.resize(hair_halo, (h_res, h_res), interpolation=cv2.INTER_AREA)
    gray_f = (rgb.mean(axis=2)).astype(np.uint8)
    if np.any(chroma_ribbon_u8):
        rib_dil = cv2.dilate(chroma_ribbon_u8, np.ones((5, 5), np.uint8))
        gray_f = cv2.inpaint(gray_f, rib_dil, 9, cv2.INPAINT_TELEA)
    photo_luma_map = cv2.resize(gray_f.astype(np.float32) / 255.0, (h_res, h_res), interpolation=cv2.INTER_AREA)

    # Save generated maps
    delit_path = os.path.join(out_dir, "portrait_delit_color.png")
    height_path = os.path.join(out_dir, "portrait_height_16.png")
    rough_path = os.path.join(out_dir, "portrait_rough_8.png")
    skin_zones_path = os.path.join(out_dir, "portrait_skin_zones.png")
    depth_path = os.path.join(out_dir, "portrait_depth_16.png")
    matte_path = os.path.join(out_dir, "portrait_matte.png")
    skin_mask_path = os.path.join(out_dir, "portrait_skin_mask.png")

    # Keep full tall forehead, ears & throat cleanly inside `skin_clean` (wide smooth feather blends neck cleanly!)
    skin_clean = np.clip(
        skin_region * matte * (1.0 - np.clip((collar_mask - 0.38) * 1.15, 0.0, 1.0)),
        0.0,
        1.0,
    )
    skin_clean = cv2.GaussianBlur(skin_clean, (0, 0), max(24.0, face_h * 0.085))

    Image.fromarray(delit_srgb).save(delit_path)
    cv2.imwrite(height_path, height_16)
    cv2.imwrite(rough_path, rough_8)
    Image.fromarray(skin_zones_u8).save(skin_zones_path)
    cv2.imwrite(depth_path, (depth01 * 65535.0).round().astype(np.uint16))
    cv2.imwrite(matte_path, (matte * 255.0).round().astype(np.uint8))
    cv2.imwrite(skin_mask_path, (skin_clean * 255.0).round().astype(np.uint8))

    # Estimate gaze offset (normalized pupil shift inside eye corners, + toward image right / char left)
    l_eye_mid_x = 0.5 * (lms_norm[362][0] + lms_norm[263][0])
    r_eye_mid_x = 0.5 * (lms_norm[133][0] + lms_norm[33][0])
    gaze_yaw_ratio_L = (lms_norm[473][0] - l_eye_mid_x) / max(1e-4, abs(lms_norm[263][0] - lms_norm[362][0]))
    gaze_yaw_ratio_R = (lms_norm[468][0] - r_eye_mid_x) / max(1e-4, abs(lms_norm[133][0] - lms_norm[33][0]))

    analysis = {
        "ref_path": ref_path,
        "width": W,
        "height": H,
        "landmarks_norm": lms_norm,
        "blendshapes": blendshapes,
        "facial_transform": xf,
        "color_metrics": color_metrics,
        "gaze_yaw_ratio": round(0.5 * (gaze_yaw_ratio_L + gaze_yaw_ratio_R), 4),
        "iris_rgb_srgb": iris_rgb,
        "facial_hair": facial_hair,
        "hair_stats": {
            "crown_lift_ratio": round(crown_lift_ratio, 4),
            "side_L_flare_ratio": round(side_l_flare, 4),
            "side_R_flare_ratio": round(side_r_flare, 4),
            "med_luma": round(hair_med_luma, 4),
            "med_rgb_srgb": [round(float(c), 4) for c in hair_med_rgb],
            "face_center_x_norm": round(face_cx / W, 4),
        },
        "hair_guide": {
            "res": h_res,
            "halo": np.round(hair_halo_map, 3).tolist(),
            "luma": np.round(photo_luma_map, 3).tolist(),
        },
        "hair_sculpt": hair_sculpt,
        "torso_coat": torso_coat,
        "collar_stats": {
            "luma": round(collar_luma, 4),
            "std": round(collar_std, 4),
            "rgb_srgb": [round(float(c), 4) for c in collar_rgb],
            "p15_lin": round(collar_p15_lin, 4),
            "p50_lin": round(collar_p50_lin, 4),
            "p85_lin": round(collar_p85_lin, 4),
        },
        "maps": {
            "delit_color": delit_path,
            "height_16": height_path,
            "rough_8": rough_path,
            "skin_zones": skin_zones_path,
            "depth_16": depth_path,
            "matte": matte_path,
            "skin_mask": skin_mask_path,
        },
    }
    json_path = os.path.join(out_dir, "portrait_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print("PORTRAIT_ANALYSIS_OK:", json_path, flush=True)
    return analysis


# ---------------------------------------------------------------------------
# Pinhole Camera Solver (focal sweep + SQPNP/LM), Anchored 3D RBF, UV Baker
# ---------------------------------------------------------------------------
def _solve_camera_focal_sweep(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    weights: np.ndarray,
    W: int,
    H: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Sweep candidate studio/portrait focal lengths and solve exact K, R, t."""
    cx, cy = W * 0.5, H * 0.5
    best = None
    hi_mask = weights >= 0.7
    obj_hi = obj_pts[hi_mask].astype(np.float64)
    img_hi = img_pts[hi_mask].astype(np.float64)
    obj_all = obj_pts.astype(np.float64)
    img_all = img_pts.astype(np.float64)

    for focal_mm in (38.0, 48.0, 60.0, 75.0, 88.0, 105.0, 125.0):
        fx = (focal_mm / 36.0) * W
        K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_hi, img_hi, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj_hi, img_hi, K, None, rvec, tvec)
        proj, _ = cv2.projectPoints(obj_all, rvec, tvec, K, None)
        proj = proj.reshape(-1, 2)
        err_px = np.sqrt(np.average(np.sum((proj - img_all) ** 2, axis=1), weights=weights))
        R, _ = cv2.Rodrigues(rvec)
        cam_w = -R.T @ tvec.reshape(3)
        # Ensure camera sits on front -Y hemisphere looking at +Y head
        if cam_w[1] > -0.15:
            continue
        # Slight regularization toward classic 75-95mm portrait perspective on tie
        score = err_px + 0.04 * abs(focal_mm - 82.0)
        if best is None or score < best[0]:
            best = (score, focal_mm, K, R, tvec.reshape(3), float(err_px))

    if best is None:
        raise RuntimeError("solvePnP failed to find valid front-hemisphere camera pose")
    _, focal_mm, K, R, t, err_px = best
    return focal_mm, K, R, t, err_px


def _opencv_to_blender_cam_matrix(R: np.ndarray, t: np.ndarray) -> List[List[float]]:
    """
    Convert OpenCV X_cv = R * X_world + t into exact Blender 4x4 camera world matrix.
    Blender Camera looks along -Z, +X right, +Y up -> [-Y_cv = +Y_bl, -Z_cv = +Z_bl].
    det([r0, -r1, -r2]) == +1 (exact SO(3)).
    """
    cam_pos = (-R.T @ t.reshape(3)).tolist()
    r0 = R.T[:, 0]
    r1 = -R.T[:, 1]
    r2 = -R.T[:, 2]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = r0
    mat[:3, 1] = r1
    mat[:3, 2] = r2
    mat[:3, 3] = cam_pos
    return mat.tolist()


def _rasterize_head_uv_textures(
    body_verts: np.ndarray,
    body_normals: np.ndarray,
    tris_uv_vidx: np.ndarray,
    tris_uv_coords: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    delit_rgb: np.ndarray,
    height_16: np.ndarray,
    rough_8: np.ndarray,
    skin_mask: np.ndarray,
    out_dir: str,
    sym_mid_pt: Optional[np.ndarray] = None,
    sym_axis_lr: Optional[np.ndarray] = None,
    hair_halo_img: Optional[np.ndarray] = None,
    hair_stats: Optional[Dict] = None,
    skin_zones_rgb: Optional[np.ndarray] = None,
    chin_z_w: Optional[float] = None,
    brow_z_w: Optional[float] = None,
    tex_size: int = 2048,
) -> Dict[str, str]:
    """
    Universal Full-Head + Ears + Neck Vectorized UV Texture Baker:
      1. Projects primary camera rays onto MPFB2 head, ears, forehead & full throat (`z >= 1.24`).
      2. For oblique/grazing cheeks, jaws & far ears (`vis_front < vis_sym`), reflects across head sagittal
         symmetry plane to borrow high-detail photo pores/wrinkles from the camera-facing side!
      3. Synthesizes subtle multi-scale dermal pore grain matched to neck/cheek tone for rear skull & nape
         so ZERO flat empty grey mud ever appears in UV BaseColor or turnaround angles!
    """
    H_img, W_img = delit_rgb.shape[:2]
    T = len(tris_uv_vidx)
    id_img = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)

    uv_px = tris_uv_coords.copy()
    uv_px[:, :, 0] = uv_px[:, :, 0] * (tex_size - 1)
    uv_px[:, :, 1] = (1.0 - uv_px[:, :, 1]) * (tex_size - 1)

    for tid in range(T):
        val = tid + 1
        col = (val & 255, (val >> 8) & 255, (val >> 16) & 255)
        pts = np.round(uv_px[tid]).astype(np.int32)
        cv2.fillConvexPoly(id_img, pts, col)

    tid_buf = (
        id_img[:, :, 0].astype(np.int32)
        + (id_img[:, :, 1].astype(np.int32) << 8)
        + (id_img[:, :, 2].astype(np.int32) << 16)
        - 1
    )
    valid = tid_buf >= 0
    vy, vx = np.where(valid)
    tf = tid_buf[vy, vx]

    p = np.stack([vx.astype(np.float32), vy.astype(np.float32)], axis=1)
    a = uv_px[tf, 0]
    b = uv_px[tf, 1]
    c = uv_px[tf, 2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    good_d = np.abs(denom) > 1e-6
    denom_s = np.where(good_d, denom, 1.0)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / denom_s
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / denom_s
    w0 = 1.0 - w1 - w2
    w0 = np.clip(w0, 0.0, 1.0)
    w1 = np.clip(w1, 0.0, 1.0)
    w2 = np.clip(w2, 0.0, 1.0)
    ws = np.maximum(1e-6, w0 + w1 + w2)
    w0 /= ws; w1 /= ws; w2 /= ws

    vi = tris_uv_vidx[tf]
    pw = w0[:, None] * body_verts[vi[:, 0]] + w1[:, None] * body_verts[vi[:, 1]] + w2[:, None] * body_verts[vi[:, 2]]
    nw = w0[:, None] * body_normals[vi[:, 0]] + w1[:, None] * body_normals[vi[:, 1]] + w2[:, None] * body_normals[vi[:, 2]]
    nw /= np.maximum(1e-6, np.linalg.norm(nw, axis=1, keepdims=True))

    cam_pos = -R.T @ t.reshape(3)

    def _sample_view(pts_w: np.ndarray, nrm_w: np.ndarray):
        vdir = cam_pos[None, :] - pts_w
        vdir /= np.maximum(1e-6, np.linalg.norm(vdir, axis=1, keepdims=True))
        vdot = np.sum(nrm_w * vdir, axis=1)
        xcv = (R @ pts_w.T).T + t[None, :]
        in_f = xcv[:, 2] > 0.05
        ui = (K[0, 0] * xcv[:, 0] / np.maximum(0.05, xcv[:, 2]) + K[0, 2]).astype(np.float32)
        vi_y = (K[1, 1] * xcv[:, 1] / np.maximum(0.05, xcv[:, 2]) + K[1, 2]).astype(np.float32)
        in_b = in_f & (ui >= 1.0) & (ui < W_img - 2.0) & (vi_y >= 1.0) & (vi_y < H_img - 2.0)

        mx = np.zeros((tex_size, tex_size), dtype=np.float32)
        my = np.zeros((tex_size, tex_size), dtype=np.float32)
        mx[vy, vx] = np.clip(ui, 0.0, W_img - 1.0)
        my[vy, vx] = np.clip(vi_y, 0.0, H_img - 1.0)
        s_rgb = cv2.remap(delit_rgb, mx, my, cv2.INTER_LINEAR)[vy, vx].astype(np.float32)
        s_msk = (cv2.remap(skin_mask, mx, my, cv2.INTER_LINEAR)[vy, vx] / 255.0).astype(np.float32)
        s_h16 = cv2.remap(height_16, mx, my, cv2.INTER_LINEAR)[vy, vx].astype(np.float32)
        s_r8 = cv2.remap(rough_8, mx, my, cv2.INTER_LINEAR)[vy, vx].astype(np.float32)

        # Split obliquity by zone: clip grazing side-cheek & nose rays (`vdot < 0.32`) so 2D whiskers/highlights NEVER smear across side cheeks!
        is_nose_zone = (np.abs(pts_w[:, 0] - sym_mid_pt[0]) < 0.025) if (sym_mid_pt is not None) else np.zeros(len(pts_w), dtype=bool)
        td_nose = np.clip((vdot - 0.32) / 0.34, 0.0, 1.0)
        td_cheek = np.clip((vdot - 0.32) / 0.38, 0.0, 1.0)
        td = np.where(is_nose_zone, td_nose, td_cheek)
        sm = td * td * (3.0 - 2.0 * td)
        # Cover full head, ears & throat smoothly down to clavicles (z >= 1.24 m)
        hz = np.clip((pts_w[:, 2] - 1.24) / 0.06, 0.0, 1.0).astype(np.float32)
        vw = np.where(in_b, sm * s_msk * hz, 0.0)
        return s_rgb, s_h16, s_r8, vw

    samp_rgb, samp_h16, samp_r8, vis_front = _sample_view(pw, nw)

    # Anchor neutral skin tone to true median (`50th percentile`) so side cheeks & temples NEVER get a pale ring!
    front_ok = vis_front > 0.62
    if np.any(front_ok):
        fallback_rgb = np.percentile(samp_rgb[front_ok], 50, axis=0).astype(np.float32)
        cheek_std = float(np.clip(np.std(samp_rgb[front_ok]) * 0.28, 6.5, 13.0))
    else:
        fallback_rgb = np.array([178.0, 140.0, 120.0], dtype=np.float32)
        cheek_std = 8.5

    # Smooth multi-scale Gaussian UV boundary propagation from front camera (ZERO light-dark-light side patches!)
    diff_sz = 512
    uv_acc = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    w_acc = np.zeros((tex_size, tex_size), dtype=np.float32)
    uv_acc[vy, vx] = samp_rgb * vis_front[:, None]
    w_acc[vy, vx] = vis_front
    acc_sm = cv2.resize(uv_acc, (diff_sz, diff_sz), interpolation=cv2.INTER_AREA)
    w_sm = cv2.resize(w_acc, (diff_sz, diff_sz), interpolation=cv2.INTER_AREA)

    diff_sm = np.broadcast_to(fallback_rgb[None, None, :], (diff_sz, diff_sz, 3)).copy()
    for sig in (72.0, 32.0, 12.0, 4.5):
        b_col = cv2.GaussianBlur(acc_sm, (0, 0), sig)
        b_w = cv2.GaussianBlur(w_sm, (0, 0), sig)
        alpha_l = np.clip(b_w * 5.5, 0.0, 1.0)[:, :, None]
        est = b_col / np.maximum(1e-4, b_w[:, :, None])
        diff_sm = diff_sm * (1.0 - alpha_l) + est * alpha_l
    diff_full = cv2.resize(diff_sm, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)

    # Soft cranial hair-bed UV shading (`only inside non-skin scalp/temple/back skull, blurred smoothly across hairline!`)
    scalp_uv_w = np.zeros(len(pw), dtype=np.float32)
    scalp_fib_rgb = None
    if hair_halo_img is not None and hair_stats is not None:
        xcv_h = (R @ pw.T).T + t[None, :]
        in_fh = xcv_h[:, 2] > 0.05
        ui_h = np.clip((K[0, 0] * xcv_h[:, 0] / np.maximum(0.05, xcv_h[:, 2]) + K[0, 2]).astype(np.float32), 0.0, W_img - 1.0)
        vi_h = np.clip((K[1, 1] * xcv_h[:, 1] / np.maximum(0.05, xcv_h[:, 2]) + K[1, 2]).astype(np.float32), 0.0, H_img - 1.0)
        mx_h = np.zeros((tex_size, tex_size), dtype=np.float32)
        my_h = np.zeros((tex_size, tex_size), dtype=np.float32)
        mx_h[vy, vx] = ui_h
        my_h[vy, vx] = vi_h
        s_hhalo = cv2.remap(hair_halo_img, mx_h, my_h, cv2.INTER_LINEAR)[vy, vx].astype(np.float32)
        s_fmsk = (cv2.remap(skin_mask, mx_h, my_h, cv2.INTER_LINEAR)[vy, vx] / 255.0).astype(np.float32)
        mid_y = float(sym_mid_pt[1]) if sym_mid_pt is not None else -0.05
        mid_z = float(sym_mid_pt[2]) if sym_mid_pt is not None else 1.56
        mid_x = float(sym_mid_pt[0]) if sym_mid_pt is not None else 0.0
        # Strictly protect ears, face, jawline & throat while blending above-ear parietal/temporal scalp hair bed!
        in_cranial_rear = (pw[:, 1] > (mid_y + 0.118)) & (pw[:, 2] > (mid_z - 0.022))
        in_crown_vault = (pw[:, 2] > (mid_z + 0.024)) & (pw[:, 1] > (mid_y + 0.056))
        in_front_scalp = in_fh & (s_hhalo > 0.32) & (s_fmsk < 0.26) & (pw[:, 2] > (mid_z + 0.036))
        raw_scalp = np.where(in_cranial_rear | in_crown_vault | in_front_scalp, 1.0, 0.0).astype(np.float32)
        # Protect outer ears (`z < mid_z + 0.022 and y < mid_y + 0.118`) and full front/lateral face skin
        ear_face_keep = (pw[:, 2] < (mid_z + 0.022)) & (pw[:, 1] < (mid_y + 0.118))
        raw_scalp[ear_face_keep] = 0.0
        sc_img = np.zeros((tex_size, tex_size), dtype=np.float32)
        sc_img[vy, vx] = raw_scalp
        sc_sm = cv2.GaussianBlur(cv2.resize(sc_img, (diff_sz, diff_sz), interpolation=cv2.INTER_AREA), (0, 0), 14.0)
        # Suppress any spill back into visible front face skin
        scalp_uv_w = np.clip(cv2.resize(sc_sm, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)[vy, vx], 0.0, 0.92)
        scalp_uv_w *= np.clip(1.0 - s_fmsk * 1.35, 0.0, 1.0)
        h_srgb = np.array(hair_stats.get("med_rgb_srgb", [0.72, 0.71, 0.69]), dtype=np.float32) * 255.0
        h_luma = float(hair_stats.get("med_luma", 0.72))
        # Natural recessed root bed tone: shadowed mid-silver on white/grey hair so gaps between locks read seamlessly!
        bed_mul = 0.64 if h_luma > 0.50 else 0.42
        scalp_fib_rgb = np.clip(h_srgb * bed_mul, 24.0, 165.0).astype(np.float32)

    # Transfer multi-scale dermal wrinkles, crow's feet & pores (`sig=34`) onto shadowed/oblique cheeks!
    best_h16, best_r8, vis_w = samp_h16.copy(), samp_r8.copy(), vis_front.copy()
    pore_extra = np.zeros_like(samp_rgb)
    sym_hf_full = np.zeros_like(samp_rgb)
    sym_boost = np.zeros(len(samp_rgb), dtype=np.float32)
    if sym_mid_pt is not None and sym_axis_lr is not None:
        rel = pw - sym_mid_pt[None, :]
        lat = np.sum(rel * sym_axis_lr[None, :], axis=1)
        pw_sym = pw - 2.0 * lat[:, None] * sym_axis_lr[None, :]
        nw_lat = np.sum(nw * sym_axis_lr[None, :], axis=1)
        nw_sym = nw - 2.0 * nw_lat[:, None] * sym_axis_lr[None, :]
        sym_rgb, sym_h16, sym_r8, vis_sym = _sample_view(pw_sym, nw_sym)
        sym_img = np.broadcast_to(fallback_rgb[None, None, :], (tex_size, tex_size, 3)).copy()
        sym_img[vy, vx] = sym_rgb
        # Multi-scale high-pass captures BOTH fine skin pores AND broader cheek/crow's-feet wrinkles (`sig=34`)
        sym_lf = cv2.GaussianBlur(sym_img, (0, 0), 34.0)[vy, vx]
        sym_hf = sym_rgb - sym_lf

        # Transfer soft pore detail (`±9/255`) onto oblique side cheeks while keeping 16-bit displacement transition 100% seamless!
        sym_hf = np.clip(sym_rgb - sym_lf, -9.0, 9.0)
        sym_valid = np.clip((vis_sym - 0.42) / 0.34, 0.0, 1.0)

        lat_ok = np.clip((np.abs(lat) - 0.010) / 0.018, 0.0, 1.0).astype(np.float32)
        sym_luma_adv = np.clip((sym_rgb.mean(axis=1) - samp_rgb.mean(axis=1)) / 65.0, 0.0, 0.65)
        angle_adv = np.clip((vis_sym - vis_front) * 1.15, 0.0, 0.80)
        detail_w = np.clip((1.0 - vis_front) * sym_valid * lat_ok, 0.0, 1.0)
        sym_boost = np.clip(np.maximum(detail_w, np.maximum(angle_adv, sym_luma_adv) * sym_valid) * lat_ok, 0.0, 0.80)

        pore_extra = sym_hf * detail_w[:, None] * 0.65
        sym_hf_full = sym_hf * sym_boost[:, None] * 0.50
        best_h16 = samp_h16 * (1.0 - 0.45 * sym_boost) + (32768.0 + (sym_h16 - 32768.0) * 0.55) * (0.45 * sym_boost)
        best_r8 = samp_r8 * (1.0 - sym_boost) + sym_r8 * sym_boost
        vis_w = np.maximum(vis_front, np.maximum(detail_w, sym_boost) * 0.82)

    # Region-aware biological UV synthesis (`natural pores & creases on side cheeks/jaw + loose horizontal neck folds!`)
    rng_uv = np.random.default_rng(7)
    synth_std = cheek_std * 0.88
    noise_fine = rng_uv.normal(0.0, synth_std * 0.38, size=(tex_size, tex_size)).astype(np.float32)
    noise_med = cv2.GaussianBlur(rng_uv.normal(0.0, synth_std * 0.88, size=(tex_size // 2, tex_size // 2)).astype(np.float32), (0, 0), 1.6)
    noise_med = cv2.resize(noise_med, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)
    # Soft organic horizontal neck creases (`no periodic sine contour rings on UV sheet!`)
    line_neck = cv2.GaussianBlur(rng_uv.normal(0.0, synth_std * 1.25, size=(tex_size // 2, tex_size // 2)).astype(np.float32), (0, 0), sigmaX=11.5, sigmaY=1.2)
    line_neck = cv2.resize(line_neck, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)
    mottle_raw = cv2.GaussianBlur(rng_uv.normal(0.0, 1.0, size=(tex_size // 4, tex_size // 4)).astype(np.float32), (0, 0), 5.0)
    mottle_full = cv2.resize(mottle_raw, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)

    cz_ref = float(chin_z_w) if chin_z_w is not None else (float(sym_mid_pt[2]) - 0.085 if sym_mid_pt is not None else 1.48)
    bz_ref = float(brow_z_w) if brow_z_w is not None else (float(sym_mid_pt[2]) + 0.032 if sym_mid_pt is not None else 1.60)
    is_neck_v = np.clip((cz_ref - 0.002 - pw[:, 2]) / 0.032, 0.0, 1.0).astype(np.float32)
    cheek_grain = noise_fine[vy, vx] * 0.14 + noise_med[vy, vx] * 0.36 + mottle_full[vy, vx] * (synth_std * 0.52)
    neck_grain = line_neck[vy, vx] * 0.52 + noise_med[vy, vx] * 0.22 + mottle_full[vy, vx] * (synth_std * 0.42)
    region_grain = cheek_grain * (1.0 - is_neck_v) + neck_grain * is_neck_v

    pore_grain_bg = noise_fine * 0.12 + noise_med * 0.32 + mottle_full * (synth_std * 0.45)
    diff_textured = np.clip(diff_full + pore_grain_bg[:, :, None], 10.0, 238.0)

    smooth_base = np.clip(diff_full[vy, vx] + region_grain[:, None], 10.0, 238.0) + pore_extra
    if scalp_fib_rgb is not None:
        fib_tex = np.clip(scalp_fib_rgb[None, :] + pore_grain_bg[vy, vx, None] * 1.10, 18.0, 185.0)
        smooth_base = smooth_base * (1.0 - scalp_uv_w[:, None]) + fib_tex * scalp_uv_w[:, None]
    final_rgb = np.clip(samp_rgb * vis_front[:, None] + smooth_base * (1.0 - vis_front[:, None]) + sym_hf_full * vis_front[:, None], 10.0, 240.0)
    h_blend = (vis_front ** 1.35) * (1.0 - 0.45 * is_neck_v)
    # Organic cheek & neck micro-relief (`seamless front-to-side transition + loose horizontal neck folds!`)
    cheek_h16 = 32768.0 + (noise_fine[vy, vx] * 0.18 + noise_med[vy, vx] * 0.55 + line_neck[vy, vx] * 0.25) * 185.0
    neck_h16 = 32768.0 + (line_neck[vy, vx] * 0.95 + noise_med[vy, vx] * 0.28) * 240.0
    pore_h16 = cheek_h16 * (1.0 - is_neck_v) + neck_h16 * is_neck_v
    final_h16 = best_h16 * h_blend + pore_h16 * (1.0 - h_blend)
    # Keep neck deeply matte (`>= 206/255 = 0.81`, ZERO plastic shine on neck!) while preserving T-zone sebum on nose/forehead
    base_matte_r8 = 186.0 * (1.0 - is_neck_v) + 212.0 * is_neck_v
    final_r8 = np.clip(best_r8 * vis_w * (1.0 - 0.65 * is_neck_v) + (base_matte_r8 - region_grain * 0.70) * (1.0 - vis_w * (1.0 - 0.65 * is_neck_v)), 72.0, 236.0)

    # Bake UV-aligned 3-channel anatomical skin zone mask (`R=nose/malar pores, G=forehead/neck horizontal, B=delicate eyes/lips`)
    uv_zones = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    uv_zones[:, :, 1] = 160  # default soft horizontal grain outside front face
    if skin_zones_rgb is not None:
        xcv_z = (R @ pw.T).T + t[None, :]
        in_fz = xcv_z[:, 2] > 0.05
        ui_z = np.clip((K[0, 0] * xcv_z[:, 0] / np.maximum(0.05, xcv_z[:, 2]) + K[0, 2]).astype(np.float32), 0.0, W_img - 1.0)
        vi_z = np.clip((K[1, 1] * xcv_z[:, 1] / np.maximum(0.05, xcv_z[:, 2]) + K[1, 2]).astype(np.float32), 0.0, H_img - 1.0)
        mx_z = np.zeros((tex_size, tex_size), dtype=np.float32)
        my_z = np.zeros((tex_size, tex_size), dtype=np.float32)
        mx_z[vy, vx] = ui_z
        my_z[vy, vx] = vi_z
        sz_samp = cv2.remap(skin_zones_rgb, mx_z, my_z, cv2.INTER_LINEAR)[vy, vx].astype(np.float32)
        in_z_w = np.where(in_fz, vis_w, 0.0)[:, None]
        is_fore_3d = np.clip((pw[:, 2] - bz_ref) / 0.025, 0.0, 1.0).astype(np.float32)
        g_3d = np.clip(np.maximum(is_neck_v, is_fore_3d) * 235.0, 0.0, 255.0)
        z_default = np.stack([np.zeros_like(g_3d), g_3d, np.zeros_like(g_3d)], axis=1)
        z_blend = sz_samp * in_z_w + z_default * (1.0 - in_z_w)
        z_blend[:, 1] = np.maximum(z_blend[:, 1], is_neck_v * 235.0)
        uv_zones[vy, vx] = np.clip(z_blend, 0, 255).round().astype(np.uint8)

    uv_col = diff_textured.round().astype(np.uint8)
    uv_h16 = np.full((tex_size, tex_size), 32768, dtype=np.uint16)
    uv_r8 = np.full((tex_size, tex_size), 168, dtype=np.uint8)

    uv_col[vy, vx] = np.clip(final_rgb, 0, 255).round().astype(np.uint8)
    uv_h16[vy, vx] = np.clip(final_h16, 0, 65535).round().astype(np.uint16)
    uv_r8[vy, vx] = np.clip(final_r8, 0, 255).round().astype(np.uint8)

    # Generous 11px UV island gutter bleed prevents any black or dark seam edges across skull/jaw/ears
    inv_mask = (~valid).astype(np.uint8)
    if np.any(inv_mask):
        kernel = np.ones((11, 11), np.uint8)
        dil_col = cv2.dilate(np.where(valid[:, :, None], uv_col, 0), kernel)
        edge_zone = inv_mask & cv2.dilate(valid.astype(np.uint8), kernel)
        ez = edge_zone > 0
        uv_col[ez] = dil_col[ez]
        dil_h = cv2.dilate(np.where(valid, uv_h16, 0), kernel)
        uv_h16[ez] = np.where(dil_h[ez] > 0, dil_h[ez], 32768)
        dil_z = cv2.dilate(np.where(valid[:, :, None], uv_zones, 0), kernel)
        uv_zones[ez] = dil_z[ez]

    bc_path = os.path.join(out_dir, "uv_basecolor.png")
    h16_path = os.path.join(out_dir, "uv_height16.png")
    r8_path = os.path.join(out_dir, "uv_rough8.png")
    zones_path = os.path.join(out_dir, "uv_skin_zones.png")
    Image.fromarray(uv_col).save(bc_path)
    cv2.imwrite(h16_path, uv_h16)
    cv2.imwrite(r8_path, uv_r8)
    Image.fromarray(uv_zones).save(zones_path)
    return {"uv_basecolor": bc_path, "uv_height16": h16_path, "uv_rough8": r8_path, "uv_skin_zones": zones_path}


def fit_and_bake_face(analysis_path: str, mesh_in_path: str, mesh_out_path: str, out_dir: str) -> Dict:
    """Solve exact camera pose + 3D RBF TPS head warp + UV projected texture maps."""
    with open(analysis_path, "r", encoding="utf-8") as f:
        ana = json.load(f)
    with open(mesh_in_path, "r", encoding="utf-8") as f:
        mdata = json.load(f)

    W, H = int(ana["width"]), int(ana["height"])
    lms_norm = np.array(ana["landmarks_norm"], dtype=np.float64)
    all_verts = np.array(mdata["verts"], dtype=np.float64)
    tris_vidx = np.array(mdata["tris_vidx"], dtype=np.int32)
    tris_uv = np.array(mdata["tris_uv"], dtype=np.float32)

    obj_pts, img_pts, pnp_w, mp_z_rel, lm_vids = [], [], [], [], []
    for _name, mp_idx, vid, wt in LANDMARK_PAIRS:
        obj_pts.append(all_verts[vid])
        img_pts.append([lms_norm[mp_idx, 0] * W, lms_norm[mp_idx, 1] * H])
        mp_z_rel.append(lms_norm[mp_idx, 2])
        pnp_w.append(wt)
        lm_vids.append(vid)

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    pnp_w = np.array(pnp_w, dtype=np.float64)
    mp_z_rel = np.array(mp_z_rel, dtype=np.float64)

    # 1. Recover pinhole focal length & extrinsics
    focal_mm, K, R, t, err_px = _solve_camera_focal_sweep(obj_pts, img_pts, pnp_w, W, H)

    # 2. Compute camera-ray target 3D positions + cranium/neck anchors for smooth RBF warp
    cam_pos = -R.T @ t
    cam_fwd = R.T[:, 2]
    src_pts, dst_deltas = [], []

    # Scale MediaPipe canonical relative z cleanly to world meters (~face width ~ 0.14m)
    mp_span_x = float(lms_norm[:, 0].max() - lms_norm[:, 0].min())
    mp_scale_m = 0.145 / max(0.08, mp_span_x)
    # Zero-mean relative depth alignment (`never biases cheeks backward relative to nose tip!`)
    z_all_cv = np.array([float((R @ obj_pts[k] + t)[2]) for k in range(len(LANDMARK_PAIRS))], dtype=np.float64)
    z_mp_m = mp_z_rel * mp_scale_m
    z_shift_0 = float(np.median(z_all_cv - z_mp_m))

    # Fit RBF strictly on reliable facial anatomy with Bilateral 3D Symmetry Regularization (`true 3D nose & rounded cheeks in ±34°!`)
    skip_rbf = {"forehead_top", "forehead_L", "forehead_R", "temple_L", "temple_R"}
    soft_rbf = {"cheek_L", "cheek_R", "cheek_lower_L", "cheek_lower_R", "zygoma_L", "zygoma_R", "jaw_L", "jaw_R"}
    mid_nose = {"nose_tip", "nose_dorsum", "nose_bridge_top", "nose_subnasale"}
    name_to_idx = {LANDMARK_PAIRS[k][0]: k for k in range(len(LANDMARK_PAIRS))}
    raw_deltas: Dict[str, np.ndarray] = {}
    for i in range(len(LANDMARK_PAIRS)):
        lm_name = LANDMARK_PAIRS[i][0]
        if lm_name in skip_rbf:
            continue
        p_cur = obj_pts[i]
        z_cur = z_all_cv[i]
        # Compare centered MediaPipe depth vs MPFB2 rest depth (`eliminates artificial nose stretching & cheek flattening!`)
        z_res = (z_mp_m[i] + z_shift_0) - z_cur
        z_tar = z_cur + float(np.clip(z_res * 0.22, -0.0045, 0.0045))
        ray_cv = np.array([
            (img_pts[i, 0] - K[0, 2]) / K[0, 0],
            (img_pts[i, 1] - K[1, 2]) / K[1, 1],
            1.0,
        ], dtype=np.float64)
        ray_world = R.T @ ray_cv
        p_target = cam_pos + ray_world * z_tar
        delta = p_target - p_cur
        if lm_name in soft_rbf:
            delta = delta * 0.42
        if lm_name in mid_nose:
            # Dampen sideways camera-angle twist on midline nose (`keeps authentic symmetric 3D nose bridge in ±34°!`)
            delta = np.array([delta[0] * 0.48, np.clip(delta[1], -0.006, 0.006), delta[2]], dtype=np.float64)
        dlen = float(np.linalg.norm(delta))
        if dlen > 0.011:
            delta = delta * (0.011 / dlen)
        raw_deltas[lm_name] = delta

    # Symmetrize bilateral 3D facial pairs (`72% symmetric / 28% photo asymmetry`) so far oblique cheek/nose never flattens!
    bilat_pairs = [
        ("nose_ala_L", "nose_ala_R"),
        ("eye_L_outer", "eye_R_outer"),
        ("eye_L_inner", "eye_R_inner"),
        ("mouth_corner_L", "mouth_corner_R"),
        ("cheek_L", "cheek_R"),
        ("cheek_lower_L", "cheek_lower_R"),
        ("zygoma_L", "zygoma_R"),
        ("jaw_L", "jaw_R"),
    ]
    for l_nm, r_nm in bilat_pairs:
        if l_nm in raw_deltas and r_nm in raw_deltas:
            dl, dr = raw_deltas[l_nm], raw_deltas[r_nm]
            sym_l = np.array([0.5 * (dl[0] - dr[0]), 0.5 * (dl[1] + dr[1]), 0.5 * (dl[2] + dr[2])], dtype=np.float64)
            sym_r = np.array([-sym_l[0], sym_l[1], sym_l[2]], dtype=np.float64)
            raw_deltas[l_nm] = 0.72 * sym_l + 0.28 * dl
            raw_deltas[r_nm] = 0.72 * sym_r + 0.28 * dr

    for lm_name, delta in raw_deltas.items():
        i = name_to_idx[lm_name]
        src_pts.append(obj_pts[i])
        dst_deltas.append(delta)

    # Add rigid back-of-skull, crown, and neck-collar zero anchors so RBF tapers gracefully
    for a in mdata.get("anchors", []):
        src_pts.append(np.array(a, dtype=np.float64))
        dst_deltas.append(np.zeros(3, dtype=np.float64))

    src_arr = np.array(src_pts, dtype=np.float64)
    dst_arr = np.array(dst_deltas, dtype=np.float64)
    # Regularized smoothing prevents localized landmark pin-spikes on nose wings & cheeks in 3/4 turnarounds
    rbf = RBFInterpolator(src_arr, dst_arr, kernel="thin_plate_spline", smoothing=0.0036)

    # Smooth spatial mask: taper lower throat gracefully down to clavicles (`zero cheek/jaw discontinuities`)
    head_falloff = np.clip((all_verts[:, 2] - 1.24) / 0.14, 0.0, 1.0)[:, None]
    head_mask = all_verts[:, 2] > 1.22
    warped = all_verts.copy()
    if np.any(head_mask):
        disp = rbf(all_verts[head_mask])
        warped[head_mask] = all_verts[head_mask] + disp * head_falloff[head_mask]

    # Universal Relaxed / Gently Hooded Upper & Lower Eyelid Sculpt (`naram, thodi udaas aankhein — eliminates wide-eyed startled stare!`)
    if len(lm_vids) > 13:
        for idx_in, idx_out, idx_top, idx_bot in ((6, 7, 8, 9), (10, 11, 12, 13)):
            p_in_e = warped[lm_vids[idx_in]].copy()
            p_out_e = warped[lm_vids[idx_out]].copy()
            p_top_e = warped[lm_vids[idx_top]].copy()
            p_bot_e = warped[lm_vids[idx_bot]].copy()
            ec = 0.25 * (p_in_e + p_out_e + p_top_e + p_bot_e)
            ew = max(0.012, float(np.linalg.norm(p_out_e - p_in_e)))
            d_eye = warped - ec[None, :]
            r_ell = np.sqrt((d_eye[:, 0] / (ew * 0.72)) ** 2 + (d_eye[:, 1] / 0.018) ** 2 + (d_eye[:, 2] / 0.016) ** 2)
            in_lid = r_ell < 1.0
            if np.any(in_lid):
                w_lid = (1.0 - r_ell[in_lid] ** 2) ** 1.35
                # Outer-corner horizontal progress (0 at inner tear duct -> 1 at outer canthus for gentle sad/kind outer hood!)
                out_t = np.clip(np.abs(warped[in_lid, 0] - p_in_e[0]) / max(0.008, abs(p_out_e[0] - p_in_e[0])), 0.0, 1.0)
                is_upper = np.clip((warped[in_lid, 2] - (ec[2] - 0.001)) / 0.004, 0.0, 1.0)
                is_lower = np.clip(((ec[2] - 0.001) - warped[in_lid, 2]) / 0.004, 0.0, 1.0)
                # Gently lower upper lid (-2.2mm center, -2.9mm outer hood) and raise lower lid (+0.8mm)
                dz_lid = (-0.0021 * (0.75 + 0.45 * out_t) * is_upper + 0.0008 * is_lower) * w_lid
                warped[in_lid, 2] += dz_lid

    # 3D Anatomical Loose Submental Wattle, Platysma Cords & Horizontal Throat Folds (scaled by photo skin_texture_energy)
    tex_e_neck = float(ana.get("color_metrics", {}).get("skin_texture_energy", 0.50))
    chin_pt_w = warped[lm_vids[27]] if len(lm_vids) > 27 else np.array([0.0, -0.08, 1.48])
    cz_w = float(chin_pt_w[2])
    cx_w = float(chin_pt_w[0])
    cy_w = float(chin_pt_w[1])
    neck_zone = (warped[:, 2] > (cz_w - 0.080)) & (warped[:, 2] < (cz_w + 0.006)) & (warped[:, 1] < (cy_w + 0.095))
    if np.any(neck_zone) and tex_e_neck > 0.18:
        nz_pts = warped[neck_zone]
        dz_n = nz_pts[:, 2] - cz_w  # [-0.080 .. +0.006]
        dx_n = nz_pts[:, 0] - cx_w
        dy_n = nz_pts[:, 1] - cy_w
        # Vertical envelope peaking on upper-to-mid throat right below chin/jaw
        v_env = np.sin(np.clip((-dz_n + 0.004) / 0.082, 0.0, 1.0) * np.pi) ** 1.15
        front_throat = np.clip(1.0 - (dy_n - 0.010) / 0.068, 0.0, 1.0)
        # Paired vertical platysma muscle cords (x ≈ ±13mm) + soft submental double-chin wattle drape
        cord_l = np.exp(-(((dx_n - 0.013) / 0.0085) ** 2))
        cord_r = np.exp(-(((dx_n + 0.013) / 0.0085) ** 2))
        submental_sag = np.exp(-((dx_n / 0.032) ** 2)) * np.exp(-(((dz_n + 0.014) / 0.018) ** 2))
        # Concentric horizontal loose neck skin folds wrapping around throat
        h_folds = (np.sin(dz_n * 285.0) * 0.65 + np.cos(dz_n * 155.0) * 0.35) * np.clip(1.0 - np.abs(dx_n) / 0.065, 0.0, 1.0)
        # Push forward (-Y) along cords/wattle & radially modulate horizontal neck folds
        dy_push = -(0.0034 * (cord_l + cord_r) * v_env + 0.0038 * submental_sag + 0.0011 * h_folds * v_env) * front_throat * tex_e_neck
        dz_sag = -(0.0022 * submental_sag + 0.0014 * (cord_l + cord_r) * v_env) * front_throat * tex_e_neck
        warped[neck_zone, 1] += dy_push
        warped[neck_zone, 2] += dz_sag

    # Re-solve final exact camera on warped mesh so projection error hits near-zero!
    warped_lm_pts = warped[lm_vids]
    focal_mm2, K2, R2, t2, err_px2 = _solve_camera_focal_sweep(warped_lm_pts, img_pts, pnp_w, W, H)
    cam_mat_4x4 = _opencv_to_blender_cam_matrix(R2, t2)

    # Estimate head sagittal symmetry plane from exact L/R eye_outer (7,11), mouth_corner (25,26) & jaw (28,29) pairs
    sym_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if len(warped_lm_pts) >= 30:
        lr_vec = (warped_lm_pts[7] - warped_lm_pts[11]) + (warped_lm_pts[25] - warped_lm_pts[26]) + (warped_lm_pts[28] - warped_lm_pts[29])
        nrm_lr = float(np.linalg.norm(lr_vec))
        if nrm_lr > 1e-4:
            sym_axis = (lr_vec / nrm_lr).astype(np.float32)
    sym_mid = (0.5 * (warped_lm_pts[0] + warped_lm_pts[2])).astype(np.float32)

    # Compute smooth vertex normals on warped mesh for view-weighted + symmetric texture rasterization
    v0 = warped[tris_vidx[:, 0]]
    v1 = warped[tris_vidx[:, 1]]
    v2 = warped[tris_vidx[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.maximum(1e-9, np.linalg.norm(fn, axis=1, keepdims=True))
    vnorm = np.zeros_like(warped)
    np.add.at(vnorm, tris_vidx[:, 0], fn)
    np.add.at(vnorm, tris_vidx[:, 1], fn)
    np.add.at(vnorm, tris_vidx[:, 2], fn)
    vnorm /= np.maximum(1e-9, np.linalg.norm(vnorm, axis=1, keepdims=True))

    delit_rgb = np.array(Image.open(ana["maps"]["delit_color"]).convert("RGB"))
    height_16 = cv2.imread(ana["maps"]["height_16"], cv2.IMREAD_UNCHANGED)
    rough_8 = cv2.imread(ana["maps"]["rough_8"], cv2.IMREAD_GRAYSCALE)
    skin_mask = cv2.imread(ana["maps"]["skin_mask"], cv2.IMREAD_GRAYSCALE)
    sz_p = ana["maps"].get("skin_zones")
    skin_zones_rgb = np.array(Image.open(sz_p).convert("RGB")) if (sz_p and os.path.exists(sz_p)) else None

    halo_small = np.array(ana.get("hair_guide", {}).get("halo", []), dtype=np.float32)
    hair_halo_full = (
        cv2.resize(halo_small, (W, H), interpolation=cv2.INTER_LINEAR)
        if halo_small.ndim == 2 and halo_small.size > 0
        else None
    )
    chin_z_val = float(warped_lm_pts[27][2]) if len(warped_lm_pts) > 27 else None
    brow_z_val = float(0.5 * (warped_lm_pts[15][2] + warped_lm_pts[18][2])) if len(warped_lm_pts) > 18 else None

    uv_maps = _rasterize_head_uv_textures(
        warped.astype(np.float32),
        vnorm.astype(np.float32),
        tris_vidx,
        tris_uv,
        K2,
        R2,
        t2,
        delit_rgb,
        height_16,
        rough_8,
        skin_mask,
        out_dir,
        sym_mid_pt=sym_mid,
        sym_axis_lr=sym_axis,
        hair_halo_img=hair_halo_full,
        hair_stats=ana.get("hair_stats"),
        skin_zones_rgb=skin_zones_rgb,
        chin_z_w=chin_z_val,
        brow_z_w=brow_z_val,
        tex_size=2048,
    )
    _harmonize_uv_seams_3d(uv_maps, warped, tris_vidx, tris_uv, sym_mid)

    out_payload = {
        "warped_verts": np.round(warped, 6).tolist(),
        "focal_mm": round(float(focal_mm2), 3),
        "reproj_err_px": round(float(err_px2), 3),
        "camera_matrix_4x4": cam_mat_4x4,
        "cam_K": np.round(K2, 4).tolist(),
        "cam_R": np.round(R2, 6).tolist(),
        "cam_t": np.round(t2, 6).tolist(),
        "uv_maps": uv_maps,
    }
    with open(mesh_out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)
    print(
        f"FIT_AND_BAKE_OK: focal={focal_mm2:.1f}mm reproj_err={err_px2:.2f}px -> {mesh_out_path}",
        flush=True,
    )
    return out_payload


def _harmonize_uv_seams_3d(
    uv_maps: Dict[str, str],
    verts: np.ndarray,
    tris_vidx: np.ndarray,
    tris_uv: np.ndarray,
    sym_mid: np.ndarray,
) -> None:
    """
    Equalize albedo and micro-height smoothly across ALL 3D UV seam boundaries
    outside the central orbital/nasal/oral features (ear creases, lateral neck,
    nape, scalp) so touching UV islands never display hard seam steps.
    """
    albedo_p = uv_maps.get("uv_basecolor")
    height_p = uv_maps.get("uv_height16")
    if not albedo_p or not os.path.exists(albedo_p):
        return

    edge_map: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray]]] = {}
    for ti in range(len(tris_vidx)):
        vs = tris_vidx[ti]
        uvs = tris_uv[ti]
        for k in range(3):
            nk = (k + 1) % 3
            va, vb = int(vs[k]), int(vs[nk])
            uva, uvb = uvs[k], uvs[nk]
            if va > vb:
                va, vb = vb, va
                uva, uvb = uvb, uva
            edge_map.setdefault((va, vb), []).append((uva, uvb))

    seam_pairs = []
    for (va, vb), sides in edge_map.items():
        if len(sides) < 2:
            continue
        p_mid = 0.5 * (verts[va] + verts[vb])
        # Protect fine inner facial anatomy (eyes, nose bridge, lips) while equalizing all neck/ear/jaw/nape seams
        if (
            abs(float(p_mid[0] - sym_mid[0])) < 0.050
            and float(p_mid[1]) < float(sym_mid[1]) + 0.040
            and float(p_mid[2]) > float(sym_mid[2]) - 0.068
        ):
            continue
        u0a, u0b = sides[0]
        if any((float(np.abs(sa - u0a).sum() + np.abs(sb - u0b).sum()) > 2e-3) for sa, sb in sides[1:]):
            seam_pairs.append(sides)

    if not seam_pairs:
        return

    alb_img = cv2.cvtColor(cv2.imread(albedo_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W = alb_img.shape[:2]
    h_img = cv2.imread(height_p, cv2.IMREAD_UNCHANGED).astype(np.float32) if (height_p and os.path.exists(height_p)) else None

    gs = 256
    d_alb = np.zeros((gs, gs, 3), dtype=np.float32)
    d_h = np.zeros((gs, gs), dtype=np.float32)
    w_grid = np.zeros((gs, gs), dtype=np.float32)
    seam_bin = np.ones((gs, gs), dtype=np.uint8)

    ts = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    for sides in seam_pairs:
        for t in ts:
            uv_s = [uva * (1.0 - t) + uvb * t for uva, uvb in sides]
            cols = []
            hs = []
            pxs = []
            for u, v in uv_s:
                ix = int(np.clip(round(float(u) * (W - 1)), 0, W - 1))
                iy = int(np.clip(round((1.0 - float(v)) * (H - 1)), 0, H - 1))
                cols.append(alb_img[iy, ix])
                if h_img is not None:
                    hs.append(float(h_img[iy, ix]))
                gx = int(np.clip(round(float(u) * (gs - 1)), 0, gs - 1))
                gy = int(np.clip(round((1.0 - float(v)) * (gs - 1)), 0, gs - 1))
                pxs.append((gx, gy))
            c_mid = np.mean(cols, axis=0)
            h_mid = float(np.mean(hs)) if hs else 0.0
            for idx_s, (gx, gy) in enumerate(pxs):
                d_alb[gy, gx] += (c_mid - cols[idx_s])
                if h_img is not None:
                    d_h[gy, gx] += (h_mid - hs[idx_s])
                w_grid[gy, gx] += 1.0
                seam_bin[gy, gx] = 0

    w_blur = cv2.GaussianBlur(w_grid, (0, 0), 5.0)
    valid = w_blur > 1e-5
    if not np.any(valid):
        return

    field_alb = cv2.GaussianBlur(d_alb, (0, 0), 5.0) / np.maximum(1e-5, w_blur[..., None])
    dist_g = cv2.distanceTransform(seam_bin, cv2.DIST_L2, 5)
    alpha_g = np.clip(1.0 - dist_g / 11.0, 0.0, 1.0)
    alpha_g = (alpha_g * alpha_g * (3.0 - 2.0 * alpha_g)) * valid.astype(np.float32)

    alpha_full = cv2.resize(alpha_g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    delta_full = cv2.resize(field_alb, (W, H), interpolation=cv2.INTER_LINEAR)
    alb_out = np.clip(alb_img + delta_full * alpha_full, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(alb_out).save(albedo_p)

    if h_img is not None:
        field_h = cv2.GaussianBlur(d_h, (0, 0), 5.0) / np.maximum(1e-5, w_blur)
        delta_h_full = cv2.resize(field_h, (W, H), interpolation=cv2.INTER_LINEAR)
        h_out = np.clip(h_img + delta_h_full * alpha_full[..., 0], 0.0, 65535.0).round().astype(np.uint16)
        cv2.imwrite(height_p, h_out)


# ---------------------------------------------------------------------------
# Universal Non-Human Arbitrary Object Analyzer & Depth/Normal/Roughness Synth
# ---------------------------------------------------------------------------
def _depth_to_tangent_normal(depth_f32: np.ndarray, strength: float = 3.5) -> np.ndarray:
    gx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3) * strength
    gy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3) * strength
    nz = np.ones_like(depth_f32, dtype=np.float32)
    vec = np.stack([-gx, gy, nz], axis=-1)
    vec /= np.maximum(1e-6, np.linalg.norm(vec, axis=-1, keepdims=True))
    return ((vec * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)


def analyze_object(ref_path: str, out_dir: str, mode: str = "auto") -> Dict:
    """Analyze ANY arbitrary non-human object image using RMBG-1.4 + Depth-Anything-V2-Small."""
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(ref_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read object image: {ref_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    matte = run_rmbg14(rgb)
    depth01 = run_depth_v2_small(rgb)

    mask_raw = (matte > 0.45).astype(np.uint8)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask_raw, connectivity=8)
    mask_bin = np.zeros_like(mask_raw, dtype=bool)
    if n_lbl > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max())
        for lid in range(1, n_lbl):
            if stats[lid, cv2.CC_STAT_AREA] >= max(200, int(0.08 * max_area)):
                mask_bin |= (labels == lid)
    else:
        mask_bin = mask_raw > 0
    # Close tiny 3-6px specular dimples cleanly WITHOUT filling open handle loops!
    closed_u8 = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask_bin = closed_u8 > 0
    matte = np.maximum(matte, closed_u8.astype(np.float32) * 0.95)
    matte = np.where(cv2.dilate(mask_bin.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, matte, 0.0)
    ys, xs = np.where(mask_bin)
    if len(xs) < 100:
        raise RuntimeError(f"Foreground object mask too small in {ref_path}")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)

    # Check horizontal rotational symmetry inside object bbox -> pick lathe vs bilateral_relief
    crop_m = mask_bin[y0 : y1 + 1, x0 : x1 + 1]
    crop_flip = np.fliplr(crop_m)
    sym_iou = float(np.logical_and(crop_m, crop_flip).sum() / max(1, np.logical_or(crop_m, crop_flip).sum()))

    # Measure normalized depth purely inside object mask
    d_fg = depth01[mask_bin]
    d_lo, d_hi = float(np.percentile(d_fg, 3)), float(np.percentile(d_fg, 97))
    d_norm = np.clip((depth01 - d_lo) / max(1e-4, d_hi - d_lo), 0.0, 1.0)
    d_norm = np.where(matte > 0.25, d_norm, 0.0)
    d_norm = cv2.GaussianBlur(d_norm, (0, 0), 2.5)

    # Estimate normalized radius profile for lathe mode (64 height slices)
    lathe_profile = []
    cx_obj = 0.5 * (x0 + x1)
    for row_frac in np.linspace(0.02, 0.98, 64):
        yy = int(round(y0 + row_frac * (y1 - y0)))
        row_xs = np.where(mask_bin[yy, :])[0]
        if len(row_xs) > 1:
            half_w = 0.5 * float(row_xs.max() - row_xs.min()) / max(W, H)
        else:
            half_w = 0.002
        lathe_profile.append([round(float(half_w), 5), round(float((H - 1 - yy) / max(W, H)), 5)])

    # Check depth variation vs center to refine auto mode
    chosen_mode = mode
    if mode == "auto":
        # True lathe objects have high horizontal symmetry & taller-or-curved cylinder silhouette
        chosen_mode = "lathe" if sym_iou > 0.90 and float(bw / bh) < 1.15 else "bilateral_relief"

    # Softly de-lit Albedo & PBR Tangent Normal / Roughness Synthesis
    lin = _srgb_to_linear_np(rgb.astype(np.float32) / 255.0)
    lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    lf_obj = cv2.GaussianBlur(lum, (0, 0), max(28.0, bw * 0.14))
    t_lum = float(np.median(lf_obj[mask_bin]))
    gain = np.clip(t_lum / np.maximum(0.04, lf_obj), 0.72, 1.65)
    blend_gain = 0.55 + 0.45 * gain
    delit_lin = np.clip(lin * blend_gain[:, :, None], 0.012, 0.86)
    delit_rgb = (_linear_to_srgb_np(delit_lin) * 255.0).round().clip(0, 255).astype(np.uint8)

    # Dilate object edge colors outward so grazing UV projections never pull dark background fringes!
    fg_only = np.where(matte[:, :, None] > 0.65, delit_rgb, 0)
    dil = cv2.dilate(fg_only, np.ones((11, 11), np.uint8))
    clean_albedo = np.where(matte[:, :, None] > 0.55, delit_rgb, dil)

    normal_rgb = _depth_to_tangent_normal(d_norm + 0.06 * (lum - lf_obj), strength=3.0)
    hf_detail = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 3.0))
    rough_val = np.clip(0.36 + 1.8 * hf_detail, 0.14, 0.85)
    rough_u8 = (rough_val * 255.0).round().astype(np.uint8)

    albedo_p = os.path.join(out_dir, "obj_delit_albedo.png")
    matte_p = os.path.join(out_dir, "obj_matte.png")
    depth_p = os.path.join(out_dir, "obj_depth16.png")
    norm_p = os.path.join(out_dir, "obj_normal.png")
    rough_p = os.path.join(out_dir, "obj_rough.png")

    Image.fromarray(clean_albedo).save(albedo_p)
    cv2.imwrite(matte_p, (matte * 255.0).round().astype(np.uint8))
    cv2.imwrite(depth_p, (d_norm * 65535.0).round().astype(np.uint16))
    Image.fromarray(normal_rgb).save(norm_p)
    cv2.imwrite(rough_p, rough_u8)

    # Export downsampled normalized depth+matte grid (96x96 bbox) for Blender bilateral_relief mesh builder
    grid_res = 96
    d_crop = cv2.resize(d_norm[y0 : y1 + 1, x0 : x1 + 1], (grid_res, grid_res), interpolation=cv2.INTER_AREA)
    m_crop = cv2.resize(matte[y0 : y1 + 1, x0 : x1 + 1], (grid_res, grid_res), interpolation=cv2.INTER_AREA)

    payload = {
        "ref_path": ref_path,
        "width": W,
        "height": H,
        "bbox": [x0, y0, x1, y1],
        "aspect_wh": round(bw / bh, 4),
        "symmetry_iou": round(sym_iou, 4),
        "chosen_mode": chosen_mode,
        "cx_norm": round(float(cx_obj / W), 5),
        "cy_norm": round(float(0.5 * (y0 + y1) / H), 5),
        "lathe_profile": lathe_profile,
        "grid_res": grid_res,
        "grid_depth": np.round(d_crop, 4).tolist(),
        "grid_matte": np.round(m_crop, 4).tolist(),
        "maps": {
            "albedo": albedo_p,
            "matte": matte_p,
            "depth16": depth_p,
            "normal": norm_p,
            "rough": rough_p,
        },
    }
    json_path = os.path.join(out_dir, "object_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"OBJECT_ANALYSIS_OK: mode={chosen_mode} sym={sym_iou:.3f} -> {json_path}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser(description="Vision Solver CLI for Exact Reference-to-3D")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("analyze-portrait")
    p1.add_argument("--img", required=True)
    p1.add_argument("--out-dir", required=True)
    p1.add_argument("--colorize", action="store_true")

    p2 = sub.add_parser("fit-and-bake-face")
    p2.add_argument("--analysis", required=True)
    p2.add_argument("--mesh-in", required=True)
    p2.add_argument("--mesh-out", required=True)
    p2.add_argument("--out-dir", required=True)

    p3 = sub.add_parser("analyze-object")
    p3.add_argument("--img", required=True)
    p3.add_argument("--out-dir", required=True)
    p3.add_argument("--mode", default="auto")

    args = ap.parse_args()
    if args.cmd == "analyze-portrait":
        analyze_portrait(args.img, args.out_dir, colorize_monochrome=args.colorize)
    elif args.cmd == "fit-and-bake-face":
        fit_and_bake_face(args.analysis, args.mesh_in, args.mesh_out, args.out_dir)
    elif args.cmd == "analyze-object":
        analyze_object(args.img, args.out_dir, args.mode)


if __name__ == "__main__":
    main()
