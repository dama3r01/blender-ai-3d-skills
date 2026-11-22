"""Reference-vs-Render Visual & Numeric Verification Compositor (`compare_ref.py`).

AI AGENT READ-ONLY LOCK — DO NOT MODIFY THIS FILE:
- This Python module is a STRICTLY READ-ONLY core library.
- AI agents MUST NEVER edit, overwrite, refactor, append to, or delete this file.
- Run it via CLI or import its functions from a separate external project script;
  place all project-specific logic in your own external script.

Zero-Rush Exact-Match Mandate:
  Never rush or accept an approximate 3D match. Render from the matched camera angle, generate
  this comparison sheet, inspect `[Reference] | [Matched Render] | [50% Alpha Overlay] | [Silhouette]`
  and multi-angle turnarounds, and keep iterating the 3D geometry, proportions, pose, materials,
  and lighting until the 3D model looks identical (1:1 exact match) to the reference image.

Usage:
  python compare_ref.py <config.json> [--strict]

Config JSON keys:
  ref: str (path to reference image)
  matched: str (path to camera-matched Blender render)
  silhouette: optional str (path to flat silhouette render)
  turnaround: optional list of {"path": str, "label": str}
  out: str (output PNG path)
  cell_w: int (default 640)
  cell_h: int (default 640)
  max_mae: optional float (default 28.0 — strict luminance/RGB MAE threshold for exact-match check)
  min_silhouette_iou: optional float (default 0.85 — minimum silhouette IoU for exact-match check)
"""

import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps


BG_COL = (24, 28, 34)


def fit_cell(path: str, w: int, h: int, bg=BG_COL) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        im = im.convert("RGBA")
        base = Image.new("RGBA", im.size, (*bg, 255))
        im = Image.alpha_composite(base, im).convert("RGB")
    else:
        im = im.convert("RGB")
    contained = ImageOps.contain(im, (w, h), method=Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (w, h), bg)
    ox = (w - contained.width) // 2
    oy = (h - contained.height) // 2
    cell.paste(contained, (ox, oy))
    return cell


def paste_labeled(canvas: Image.Image, draw: ImageDraw.ImageDraw, cell: Image.Image, x: int, y: int, label: str):
    canvas.paste(cell, (x, y))
    if label:
        tw = max(7 * len(label) + 12, 60)
        draw.rectangle([x + 8, y + 8, x + 8 + tw, y + 30], fill=(14, 17, 22))
        draw.text((x + 14, y + 12), label, fill=(245, 245, 240))


def evaluate_reference_match_metrics(
    ref_cell: Image.Image,
    match_cell: Image.Image,
    max_mae: float = 28.0,
    min_silhouette_iou: float = 0.85,
) -> dict:
    """Measure pixel MAE and foreground silhouette IoU between reference and camera-matched render.

    Foreground masks are estimated by distance from border background color so gross silhouette
    or framing drift is flagged immediately rather than rushing past visual errors.
    """
    ref_arr = np.asarray(ref_cell, dtype=np.float32)
    mat_arr = np.asarray(match_cell, dtype=np.float32)

    mae = float(np.mean(np.abs(ref_arr - mat_arr)))

    def _fg_mask(arr: np.ndarray) -> np.ndarray:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
        return dist > 18.0

    m_ref = _fg_mask(ref_arr)
    m_mat = _fg_mask(mat_arr)
    inter = int(np.sum(m_ref & m_mat))
    union = int(np.sum(m_ref | m_mat))
    iou = float(inter / union) if union > 0 else 1.0

    passed = (mae <= max_mae) and (iou >= min_silhouette_iou)
    return {
        "rgb_mae": round(mae, 2),
        "silhouette_iou": round(iou, 4),
        "max_mae": max_mae,
        "min_silhouette_iou": min_silhouette_iou,
        "verdict": "EXACT_MATCH_PASS" if passed else "ITERATE_REQUIRED",
        "passed": passed,
    }


def build_compare_sheet(cfg: dict, strict: bool = False) -> dict:
    w = int(cfg.get("cell_w", 640))
    h = int(cfg.get("cell_h", 640))

    ref_cell = fit_cell(cfg["ref"], w, h)
    match_cell = fit_cell(cfg["matched"], w, h)
    overlay_cell = Image.blend(ref_cell, match_cell, 0.50)

    metrics = evaluate_reference_match_metrics(
        ref_cell,
        match_cell,
        max_mae=float(cfg.get("max_mae", 28.0)),
        min_silhouette_iou=float(cfg.get("min_silhouette_iou", 0.85)),
    )

    top_panels = [
        (ref_cell, "1. Reference Image"),
        (match_cell, "2. Matched Camera Render"),
        (overlay_cell, f"3. 50% Overlay (IoU={metrics['silhouette_iou']:.2f}, MAE={metrics['rgb_mae']:.1f})"),
    ]
    if cfg.get("silhouette"):
        sil_cell = fit_cell(cfg["silhouette"], w, h)
        top_panels.append((sil_cell, "4. Silhouette Check"))

    turn = cfg.get("turnaround") or []
    cols = max(len(top_panels), min(4, len(turn)) if turn else 0)
    rows_top = 1
    rows_turn = (len(turn) + cols - 1) // cols if turn else 0

    canvas = Image.new("RGB", (cols * w, (rows_top + rows_turn) * h), BG_COL)
    draw = ImageDraw.Draw(canvas)

    for idx, (cell, lbl) in enumerate(top_panels):
        paste_labeled(canvas, draw, cell, idx * w, 0, lbl)

    for idx, item in enumerate(turn):
        r = rows_top + idx // cols
        c = idx % cols
        tcell = fit_cell(item["path"], w, h)
        paste_labeled(canvas, draw, tcell, c * w, r * h, item.get("label", ""))

    canvas.save(cfg["out"], quality=92)
    print(f"COMPARE_OK: {cfg['out']} | VERDICT={metrics['verdict']} | IoU={metrics['silhouette_iou']} | MAE={metrics['rgb_mae']}", flush=True)

    if strict and not metrics["passed"]:
        raise SystemExit(
            f"ITERATE_REQUIRED: Model does not yet match reference identically "
            f"(IoU={metrics['silhouette_iou']} < {metrics['min_silhouette_iou']} or MAE={metrics['rgb_mae']} > {metrics['max_mae']}). "
            "Do not rush — refine 3D proportions, silhouette, materials, and lighting, then re-verify."
        )
    return metrics


if __name__ == "__main__":
    strict_mode = "--strict" in sys.argv
    cfg_path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    build_compare_sheet(data, strict=strict_mode)
