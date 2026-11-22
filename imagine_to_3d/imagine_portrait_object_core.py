"""Universal Exact Reference-to-3D & Portrait/Object Reconstruction Core (Blender 5.2 LTS).

AI AGENT READ-ONLY LOCK — DO NOT MODIFY THIS FILE:
- This Python module is a STRICTLY READ-ONLY core library.
- AI agents MUST NEVER edit, overwrite, refactor, append to, or delete this file.
- Import and call its functions from a separate external project script; place all
  project-specific helpers or overrides in your own external script.

Zero-Rush Exact-Match Mandate:
- Never rush or settle for approximate proportions, silhouettes, hair grooms, or colors.
- Iterate camera matching, 3D landmark fitting, relief displacement, hair curves, and
  lighting across all turnaround angles until the 3D result matches identically (1:1).

Uses Blender 5.2 LTS (Cycles OPTIX + AgX, open-source MPFB2 Blender Extension for
anatomical human quad basemeshes, Subsurface RANDOM_WALK_SKIN, 16-bit wrinkle bump,
native CATMULL_ROM hair curves, tailored garments, and universal arbitrary-object reconstruction).
Design invariant: Zero subject hardcoding. Works for any portrait image or arbitrary 3D object.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import bpy
import bmesh
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
VISION_SOLVER = os.path.join(HERE, "imagine_vision_solver.py")


def _resolve_vision_python() -> str:
    """Resolve Python interpreter for vision solver CLI (override via VISION_PYTHON env var)."""
    env_py = os.environ.get("VISION_PYTHON")
    if env_py and os.path.exists(env_py):
        return env_py
    for candidate in (shutil.which("python3"), shutil.which("python"), sys.executable):
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("No Python interpreter found; set the VISION_PYTHON environment variable.")


# ---------------------------------------------------------------------------
# Subprocess Bridge & Render / Scene Setup Helpers
# ---------------------------------------------------------------------------
def run_vision_cli(args: List[str]) -> str:
    """Invoke `imagine_vision_solver.py` inside the configured Python environment."""
    py_exe = _resolve_vision_python()
    cmd = [py_exe, VISION_SOLVER, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"vision_solver failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    return res.stdout


def reset_clean_scene(engine: str = "CYCLES", samples: int = 96, res_x: int = 1024, res_y: int = 1024):
    """Clear current blend scene and configure Cycles OPTIX + AgX cleanly."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.curves, bpy.data.hair_curves, bpy.data.lights, bpy.data.cameras):
        for item in list(col):
            if item.users == 0:
                col.remove(item)

    sc = bpy.context.scene
    sc.render.engine = engine
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    if hasattr(sc.render, "hair_type"):
        sc.render.hair_type = "STRIP"

    if engine == "CYCLES":
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.02
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 8
        sc.cycles.diffuse_bounces = 3
        sc.cycles.glossy_bounces = 4
        sc.cycles.transmission_bounces = 6
        sc.cycles.transparent_max_bounces = 12
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = True
            sc.cycles.device = "GPU"
        except Exception:
            sc.cycles.device = "CPU"

    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium Contrast"
    except Exception:
        pass
    sc.view_settings.exposure = 0.0


def ensure_mpfb_enabled():
    """Enable MPFB2 extension cleanly in current Blender session."""
    import addon_utils
    if "bl_ext.user_default.mpfb" not in bpy.context.preferences.addons:
        addon_utils.enable("bl_ext.user_default.mpfb", default_set=False, persistent=False)


def setup_studio_lighting(
    target_pt: Tuple[float, float, float] = (0.0, -0.05, 1.50),
    bg_luma: float = 0.022,
    key_energy: float = 205.0,
    fill_energy: float = 44.0,
    rim_energy: float = 215.0,
    key_side_x: float = -0.80,
    key_elev_z: float = 0.72,
    key_size: float = 1.05,
    exposure: float = 0.0,
):
    """Build universal photo-matched 3-point studio lighting (`key direction, contrast & exposure driven by photo!`)."""
    sc = bpy.context.scene
    sc.view_settings.exposure = float(exposure)
    w = sc.world or bpy.data.worlds.new("StudioWorld")
    sc.world = w
    if not w.use_nodes:
        w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    wo = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (bg_luma, bg_luma * 1.01, bg_luma * 1.04, 1.0)
    bg.inputs["Strength"].default_value = 0.75
    nt.links.new(bg.outputs["Background"], wo.inputs["Surface"])

    tgt = Vector(target_pt)

    def _add_area(name: str, pos: Tuple[float, float, float], energy: float, size: float, col: Tuple[float, float, float]):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = energy
        ld.size = size
        ld.color = col
        lo = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(lo)
        lo.location = pos
        direction = tgt - lo.location
        lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return lo

    # Place Key strictly on the photo-detected lit side (`key_side_x > 0 -> +X viewer right; < 0 -> -X viewer left!`)
    ks = max(-1.0, min(1.0, float(key_side_x)))
    k_sgn = 1.0 if ks >= 0.0 else -1.0
    k_off_x = k_sgn * (0.45 + 1.05 * abs(ks))
    k_off_y = -(1.55 - 0.35 * abs(ks))
    k_off_z = 0.45 + 0.65 * max(0.2, min(1.2, float(key_elev_z)))
    _add_area("KeyLight", (tgt.x + k_off_x, tgt.y + k_off_y, tgt.z + k_off_z), key_energy, max(0.55, float(key_size)), (1.0, 0.96, 0.90))
    _add_area("FillLight", (tgt.x - k_sgn * 1.60, tgt.y - 1.15, tgt.z + 0.12), fill_energy, 1.75, (0.88, 0.93, 1.0))
    _add_area("RimLight", (tgt.x - k_sgn * 0.88, tgt.y + 1.15, tgt.z + 0.92), rim_energy, 1.05, (0.98, 0.98, 1.0))
    _add_area("CrownHaloLight", (tgt.x + k_sgn * 0.68, tgt.y + 1.08, tgt.z + 1.12), rim_energy * 0.78, 1.00, (1.0, 0.99, 0.96))


# ---------------------------------------------------------------------------
# Human Portrait Reconstruction (`MPFB2 + TargetService + 3D RBF + UV Bake`)
# ---------------------------------------------------------------------------
def _apply_safe_mpfb_targets(basemesh, target_weights: Dict[str, float]):
    """Load and set MPFB2 facial/age shape targets safely without aborting on missing aliases."""
    from bl_ext.user_default.mpfb.services.targetservice import TargetService

    for tname, val in target_weights.items():
        if abs(val) < 1e-4:
            continue
        tpath = TargetService.target_full_path(tname)
        if tpath:
            TargetService.load_target(basemesh, tpath, weight=float(val))


def _bake_and_export_mpfb_mesh(basemesh) -> Dict:
    """Evaluate deformed MPFB2 body mesh and export verts, tris, UVs, and skull/neck anchors."""
    from bl_ext.user_default.mpfb.services.targetservice import TargetService

    if basemesh.data.shape_keys:
        TargetService.bake_targets(basemesh)

    mesh = basemesh.data
    verts = [[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices]

    gi_body = basemesh.vertex_groups["body"].index
    body_idx = {v.index for v in mesh.vertices if any(g.group == gi_body and g.weight > 0.5 for g in v.groups)}

    uv_layer = mesh.uv_layers["UVMap"].data
    tris_vidx, tris_uv = [], []
    for poly in mesh.polygons:
        if not all(vi in body_idx for vi in poly.vertices):
            continue
        ls = list(poly.loop_indices)
        vs = [poly.vertices[i] for i in range(len(ls))]
        uvs = [[float(uv_layer[li].uv[0]), float(uv_layer[li].uv[1])] for li in ls]
        if len(vs) == 3:
            tris_vidx.append(vs)
            tris_uv.append(uvs)
        elif len(vs) == 4:
            tris_vidx.append([vs[0], vs[1], vs[2]])
            tris_uv.append([uvs[0], uvs[1], uvs[2]])
            tris_vidx.append([vs[0], vs[2], vs[3]])
            tris_uv.append([uvs[0], uvs[2], uvs[3]])

    # Rigid anchors: posterior skull, nape, crown back, ear backs, and shoulder ring
    anchors = []
    for v in mesh.vertices:
        if v.index not in body_idx:
            continue
        co = v.co
        if (co.y > -0.015 and co.z > 1.44) or (co.z < 1.37 and co.z > 1.24):
            if len(anchors) < 250 and (v.index % 7 == 0):
                anchors.append([float(co.x), float(co.y), float(co.z)])

    return {
        "verts": verts,
        "tris_vidx": tris_vidx,
        "tris_uv": tris_uv,
        "anchors": anchors,
    }


def _strip_helper_verts_and_lower_body(
    basemesh,
    collar_cut_z: float = 1.385,
    neck_cx: float = 0.0,
    neck_trim_half_w: float = 0.062,
):
    """Remove HelperGeometry/JointCubes + shoulder blades outside cylindrical neck core cleanly."""
    for mod in list(basemesh.modifiers):
        if "Hide helpers" in mod.name or mod.type == "MASK":
            basemesh.modifiers.remove(mod)

    rm_groups = [basemesh.vertex_groups[g].index for g in ("HelperGeometry", "JointCubes") if g in basemesh.vertex_groups]
    bm = bmesh.new()
    bm.from_mesh(basemesh.data)
    bm.verts.ensure_lookup_table()
    dlayer = basemesh.data.vertices

    # Estimate throat centerline Y & median cylindrical neck radius so trapezius flare tucks cleanly inside collar!
    neck_pts_mid = [
        bv.co for bv in bm.verts
        if (collar_cut_z + 0.032) < bv.co.z < (collar_cut_z + 0.075) and abs(bv.co.x - neck_cx) < 0.068
        and not any(g.group in rm_groups and g.weight > 0.5 for g in dlayer[bv.index].groups)
    ]
    neck_cy = (sum(p.y for p in neck_pts_mid) / len(neck_pts_mid)) if neck_pts_mid else 0.04
    if neck_pts_mid:
        rs_mid = sorted(math.hypot(p.x - neck_cx, (p.y - neck_cy) * 0.92) for p in neck_pts_mid)
        r_neck_core = min(neck_trim_half_w - 0.008, max(0.046, rs_mid[int(0.65 * len(rs_mid))]))
    else:
        r_neck_core = min(0.054, neck_trim_half_w - 0.008)

    to_del = []
    tuck_top_z = collar_cut_z + 0.078
    for bv in bm.verts:
        orig_v = dlayer[bv.index]
        is_helper = any(g.group in rm_groups and g.weight > 0.5 for g in orig_v.groups)
        dx, dy = bv.co.x - neck_cx, (bv.co.y - neck_cy) * 0.92
        r_neck = math.hypot(dx, dy)
        if is_helper or bv.co.z < collar_cut_z:
            to_del.append(bv)
        elif bv.co.z < tuck_top_z and r_neck > r_neck_core * 0.85:
            # Keep upper throat loose & full above collar while smoothly tucking lower clavicle/shoulder flares inside collar!
            tuck_w = max(0.0, min(1.0, (tuck_top_z - bv.co.z) / 0.056)) ** 0.85
            target_r = r_neck * (1.0 - tuck_w) + (r_neck_core * 0.80) * tuck_w
            sc_r = target_r / max(1e-5, r_neck)
            bv.co.x = neck_cx + (bv.co.x - neck_cx) * sc_r
            bv.co.y = neck_cy + (bv.co.y - neck_cy) * sc_r

    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="VERTS")
    bm.normal_update()
    bm.to_mesh(basemesh.data)
    bm.free()

    for poly in basemesh.data.polygons:
        poly.use_smooth = True


def _build_skin_pbr_material(
    name: str,
    uv_basecolor_path: str,
    uv_height16_path: str,
    uv_rough8_path: str,
    bump_distance: float = 0.00145,
    bump_strength: float = 0.85,
    is_monochrome: bool = False,
    colorize_monochrome: bool = False,
    skin_texture_energy: float = 0.55,
    uv_skin_zones_path: Optional[str] = None,
):
    """
    Build ultra-realistic Blender 5.2 Cycles RANDOM_WALK_SKIN shader driven by UV photo maps +
    3-channel anatomical skin zones (`R=nose/malar pores, G=forehead/neck horizontal micro-lines, B=delicate eyes/lips`) +
    anatomical T-zone sebum roughness/coat + cavity crease occlusion (`ZERO uniform sandpaper noise!`).
    """
    mat = bpy.data.materials.new(name)
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    tex_e = max(0.10, min(1.0, float(skin_texture_energy)))
    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.subsurface_method = "RANDOM_WALK_SKIN"
    if is_monochrome and not colorize_monochrome:
        # Clean, soft, living dermal translucency (`natural softness without dirty soot or waxy glare!`)
        sss_base_w = 0.52
        bsdf.inputs["Subsurface Radius"].default_value = (0.78, 0.58, 0.42)
        bsdf.inputs["Subsurface Scale"].default_value = 0.0024
        bsdf.inputs["Sheen Tint"].default_value = (0.90, 0.88, 0.85, 1.0)
    else:
        sss_base_w = 0.68
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.34, 0.15)
        bsdf.inputs["Subsurface Scale"].default_value = 0.0028
        bsdf.inputs["Sheen Tint"].default_value = (0.95, 0.87, 0.80, 1.0)
    bsdf.inputs["Subsurface IOR"].default_value = 1.38
    bsdf.inputs["Subsurface Anisotropy"].default_value = 0.84
    bsdf.inputs["IOR"].default_value = 1.40
    bsdf.inputs["Specular IOR Level"].default_value = 0.32
    bsdf.inputs["Coat Roughness"].default_value = 0.32
    bsdf.inputs["Sheen Weight"].default_value = 0.16
    bsdf.inputs["Sheen Roughness"].default_value = 0.58
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = n.new("ShaderNodeTexCoord")

    img_col = bpy.data.images.load(uv_basecolor_path, check_existing=True)
    tex_col = n.new("ShaderNodeTexImage")
    tex_col.image = img_col
    L(tc.outputs["UV"], tex_col.inputs["Vector"])

    img_r = bpy.data.images.load(uv_rough8_path, check_existing=True)
    img_r.colorspace_settings.name = "Non-Color"
    tex_r = n.new("ShaderNodeTexImage")
    tex_r.image = img_r
    L(tc.outputs["UV"], tex_r.inputs["Vector"])

    img_h = bpy.data.images.load(uv_height16_path, check_existing=True)
    img_h.colorspace_settings.name = "Non-Color"
    tex_h = n.new("ShaderNodeTexImage")
    tex_h.image = img_h
    L(tc.outputs["UV"], tex_h.inputs["Vector"])

    # 1. Gentle Cavity Crease Shading (`soft 16-22% crease depth, ZERO dirty burnt black soot!`)
    cav_ao = n.new("ShaderNodeMapRange")
    cav_ao.clamp = True
    cav_ao.inputs["From Min"].default_value = 0.16
    cav_ao.inputs["From Max"].default_value = 0.49
    cav_ao.inputs["To Min"].default_value = 0.78 - 0.08 * tex_e
    cav_ao.inputs["To Max"].default_value = 1.0
    L(tex_h.outputs["Color"], cav_ao.inputs["Value"])

    col_ao = n.new("ShaderNodeMix")
    col_ao.data_type = "RGBA"
    col_ao.blend_type = "MULTIPLY"
    col_ao.inputs[0].default_value = 1.0
    L(tex_col.outputs["Color"], col_ao.inputs[6])
    L(cav_ao.outputs["Result"], col_ao.inputs[7])
    L(col_ao.outputs[2], bsdf.inputs["Base Color"])

    # Modulate SSS slightly lower inside deep wrinkle troughs so Random Walk Skin NEVER waxes out sharp creases!
    sss_mul = n.new("ShaderNodeMath")
    sss_mul.operation = "MULTIPLY"
    sss_mul.inputs[1].default_value = sss_base_w
    L(cav_ao.outputs["Result"], sss_mul.inputs[0])
    L(sss_mul.outputs["Value"], bsdf.inputs["Subsurface Weight"])

    # 2. Region-Specific Anatomical Micro-Structure (`Nose/Malar Pores vs Forehead/Neck Lines vs Cheek Facets vs Smooth Eyes/Lips!`)
    has_zones = bool(uv_skin_zones_path and os.path.exists(uv_skin_zones_path))
    if has_zones:
        img_z = bpy.data.images.load(uv_skin_zones_path, check_existing=True)
        img_z.colorspace_settings.name = "Non-Color"
        tex_z = n.new("ShaderNodeTexImage")
        tex_z.image = img_z
        L(tc.outputs["UV"], tex_z.inputs["Vector"])
        sep_z = n.new("ShaderNodeSeparateColor")
        L(tex_z.outputs["Color"], sep_z.inputs["Color"])
        z_nose_out = sep_z.outputs["Red"]
        z_fore_out = sep_z.outputs["Green"]
        z_delic_out = sep_z.outputs["Blue"]
    else:
        z_nose_out = z_fore_out = z_delic_out = None

    # Zone A (`R`): Round Sebaceous Pores strictly on Nose & Inner Malar Cheeks
    vor_pore = n.new("ShaderNodeTexVoronoi")
    vor_pore.feature = "F1"
    vor_pore.inputs["Scale"].default_value = 1050.0
    L(tc.outputs["Object"], vor_pore.inputs["Vector"])
    pore_mr = n.new("ShaderNodeMapRange")
    pore_mr.clamp = True
    pore_mr.inputs["From Min"].default_value = 0.0; pore_mr.inputs["From Max"].default_value = 0.24
    pore_mr.inputs["To Min"].default_value = -0.060 * (0.55 + 0.55 * tex_e); pore_mr.inputs["To Max"].default_value = 0.0
    L(vor_pore.outputs["Distance"], pore_mr.inputs["Value"])

    # Zone B (`G`): Anisotropic Stretched Horizontal Micro-Lines for Forehead & Neck (`zero isotropic gravel!`)
    map_horiz = n.new("ShaderNodeMapping")
    map_horiz.inputs["Scale"].default_value = (0.45, 0.45, 2.8)
    L(tc.outputs["Object"], map_horiz.inputs["Vector"])
    ns_horiz = n.new("ShaderNodeTexNoise")
    ns_horiz.inputs["Scale"].default_value = 460.0
    ns_horiz.inputs["Detail"].default_value = 5.0
    ns_horiz.inputs["Roughness"].default_value = 0.55
    L(map_horiz.outputs["Vector"], ns_horiz.inputs["Vector"])
    horiz_mr = n.new("ShaderNodeMapRange")
    horiz_mr.inputs["From Min"].default_value = 0.25; horiz_mr.inputs["From Max"].default_value = 0.75
    horiz_mr.inputs["To Min"].default_value = -0.026 * (0.5 + 0.5 * tex_e); horiz_mr.inputs["To Max"].default_value = 0.026 * (0.5 + 0.5 * tex_e)
    L(ns_horiz.outputs["Fac"], horiz_mr.inputs["Value"])

    # Zone C (`Cheeks/Jaw`): Ultra-delicate Polygonal Dermal Micro-Facets (`subtle, never harsh sand noise!`)
    vor_cell = n.new("ShaderNodeTexVoronoi")
    vor_cell.feature = "DISTANCE_TO_EDGE"
    vor_cell.inputs["Scale"].default_value = 640.0
    L(tc.outputs["Object"], vor_cell.inputs["Vector"])
    cell_mr = n.new("ShaderNodeMapRange")
    cell_mr.clamp = True
    cell_mr.inputs["From Min"].default_value = 0.0; cell_mr.inputs["From Max"].default_value = 0.16
    cell_mr.inputs["To Min"].default_value = -0.032 * (0.45 + 0.55 * tex_e); cell_mr.inputs["To Max"].default_value = 0.0
    L(vor_cell.outputs["Distance"], cell_mr.inputs["Value"])

    ns_soft = n.new("ShaderNodeTexNoise")
    ns_soft.inputs["Scale"].default_value = 210.0
    ns_soft.inputs["Detail"].default_value = 5.0
    ns_soft.inputs["Roughness"].default_value = 0.55
    L(tc.outputs["Object"], ns_soft.inputs["Vector"])

    if has_zones:
        # Gate Nose pores by R, Forehead/Neck lines by G, Cheek facets by (1 - 0.7*G), and dampen Eyes/Lips by (1 - 0.85*B)
        p_w = n.new("ShaderNodeMath"); p_w.operation = "MULTIPLY"
        L(pore_mr.outputs["Result"], p_w.inputs[0]); L(z_nose_out, p_w.inputs[1])

        h_w = n.new("ShaderNodeMath"); h_w.operation = "MULTIPLY"
        L(horiz_mr.outputs["Result"], h_w.inputs[0]); L(z_fore_out, h_w.inputs[1])

        inv_g = n.new("ShaderNodeMapRange"); inv_g.clamp = True
        inv_g.inputs["From Min"].default_value = 0.0; inv_g.inputs["From Max"].default_value = 1.0
        inv_g.inputs["To Min"].default_value = 1.0; inv_g.inputs["To Max"].default_value = 0.25
        L(z_fore_out, inv_g.inputs["Value"])
        c_w = n.new("ShaderNodeMath"); c_w.operation = "MULTIPLY"
        L(cell_mr.outputs["Result"], c_w.inputs[0]); L(inv_g.outputs["Result"], c_w.inputs[1])

        s_ph = n.new("ShaderNodeMath"); s_ph.operation = "ADD"; L(p_w.outputs["Value"], s_ph.inputs[0]); L(h_w.outputs["Value"], s_ph.inputs[1])
        s_all = n.new("ShaderNodeMath"); s_all.operation = "ADD"; L(s_ph.outputs["Value"], s_all.inputs[0]); L(c_w.outputs["Value"], s_all.inputs[1])

        delic_att = n.new("ShaderNodeMapRange"); delic_att.clamp = True
        delic_att.inputs["From Min"].default_value = 0.0; delic_att.inputs["From Max"].default_value = 1.0
        delic_att.inputs["To Min"].default_value = 1.0; delic_att.inputs["To Max"].default_value = 0.14
        L(z_delic_out, delic_att.inputs["Value"])
        h_micro = n.new("ShaderNodeMath"); h_micro.operation = "MULTIPLY"
        L(s_all.outputs["Value"], h_micro.inputs[0]); L(delic_att.outputs["Result"], h_micro.inputs[1])
    else:
        h_micro = n.new("ShaderNodeMath"); h_micro.operation = "ADD"
        L(cell_mr.outputs["Result"], h_micro.inputs[0]); L(pore_mr.outputs["Result"], h_micro.inputs[1])

    # 3. Anatomical Matte Skin + Subtle Natural T-Zone Sebum Oiliness (`ONLY nose, forehead ridge, lips & lid rim!`)
    r_grain = n.new("ShaderNodeMapRange")
    r_grain.inputs["From Min"].default_value = 0.25; r_grain.inputs["From Max"].default_value = 0.75
    r_grain.inputs["To Min"].default_value = -0.03; r_grain.inputs["To Max"].default_value = 0.05
    L(ns_soft.outputs["Fac"], r_grain.inputs["Value"])
    r_add = n.new("ShaderNodeMath"); r_add.operation = "ADD"; r_add.use_clamp = True
    L(tex_r.outputs["Color"], r_add.inputs[0])
    L(r_grain.outputs["Result"], r_add.inputs[1])
    r_clamp = n.new("ShaderNodeMapRange"); r_clamp.clamp = True
    r_clamp.inputs["From Min"].default_value = 0.0; r_clamp.inputs["From Max"].default_value = 1.0
    r_clamp.inputs["To Min"].default_value = 0.34; r_clamp.inputs["To Max"].default_value = 0.90
    L(r_add.outputs["Value"], r_clamp.inputs["Value"])
    L(r_clamp.outputs["Result"], bsdf.inputs["Roughness"])

    # Drive micro-sebum Coat strictly on glossy T-zone (`uv_rough8 < 0.52`), ZERO plastic Coat on matte cheeks/jaw/neck!
    sebum_coat = n.new("ShaderNodeMapRange")
    sebum_coat.clamp = True
    sebum_coat.inputs["From Min"].default_value = 0.34
    sebum_coat.inputs["From Max"].default_value = 0.52
    sebum_coat.inputs["To Min"].default_value = 0.14
    sebum_coat.inputs["To Max"].default_value = 0.0
    L(tex_r.outputs["Color"], sebum_coat.inputs["Value"])
    L(sebum_coat.outputs["Result"], bsdf.inputs["Coat Weight"])

    # 4. Chained Dual-Scale Normal/Bump (`Region-Aware Micro-Skin -> Smooth Natural Photo Wrinkles!`)
    bmp_micro = n.new("ShaderNodeBump")
    bmp_micro.inputs["Strength"].default_value = 0.34 + 0.16 * tex_e
    bmp_micro.inputs["Distance"].default_value = 0.00046
    L(h_micro.outputs["Value"], bmp_micro.inputs["Height"])

    bmp_photo = n.new("ShaderNodeBump")
    bmp_photo.inputs["Strength"].default_value = min(0.62, max(bump_strength * 0.68, 0.52))
    bmp_photo.inputs["Distance"].default_value = min(0.00098, max(bump_distance * 0.58, 0.00082))
    L(tex_h.outputs["Color"], bmp_photo.inputs["Height"])
    L(bmp_micro.outputs["Normal"], bmp_photo.inputs["Normal"])
    L(bmp_photo.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _build_realistic_eye_shader(name: str, iris_rgb_srgb: Tuple[float, float, float]):
    """Build soulful wet spherical eyeball shader: soft SSS sclera + rich warm-dark iris + upper-lid shadow + corneal catchlight."""
    mat = bpy.data.materials.new(name)
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.025
    bsdf.inputs["IOR"].default_value = 1.44
    bsdf.inputs["Specular IOR Level"].default_value = 0.82
    bsdf.inputs["Coat Weight"].default_value = 0.88
    bsdf.inputs["Coat Roughness"].default_value = 0.010
    bsdf.inputs["Subsurface Weight"].default_value = 0.22
    bsdf.inputs["Subsurface Scale"].default_value = 0.0015
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    L(tc.outputs["UV"], sep.inputs["Vector"])

    dx = n.new("ShaderNodeMath"); dx.operation = "SUBTRACT"; dx.inputs[1].default_value = 0.5
    L(sep.outputs["X"], dx.inputs[0])
    dy = n.new("ShaderNodeMath"); dy.operation = "SUBTRACT"; dy.inputs[1].default_value = 0.5
    L(sep.outputs["Y"], dy.inputs[0])
    dx2 = n.new("ShaderNodeMath"); dx2.operation = "MULTIPLY"; L(dx.outputs["Value"], dx2.inputs[0]); L(dx.outputs["Value"], dx2.inputs[1])
    dy2 = n.new("ShaderNodeMath"); dy2.operation = "MULTIPLY"; L(dy.outputs["Value"], dy2.inputs[0]); L(dy.outputs["Value"], dy2.inputs[1])
    r2 = n.new("ShaderNodeMath"); r2.operation = "ADD"; L(dx2.outputs["Value"], r2.inputs[0]); L(dy2.outputs["Value"], r2.inputs[1])
    rad = n.new("ShaderNodeMath"); rad.operation = "SQRT"; L(r2.outputs["Value"], rad.inputs[0])

    ns = n.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 54.0
    ns.inputs["Detail"].default_value = 5.0
    L(tc.outputs["UV"], ns.inputs["Vector"])

    # Rich, warm-dark soulful iris and soft ivory sclera (`gentle, slightly sad/kind portrait eyes, never bright startled white marbles!`)
    ir_lin = tuple(max(0.024, min(0.14, (c / 1.055) ** 2.2 * 1.35 + 0.018)) for c in iris_rgb_srgb)
    cr = n.new("ShaderNodeValToRGB")
    elems = cr.color_ramp.elements
    elems[0].position = 0.098; elems[0].color = (0.002, 0.002, 0.003, 1.0)
    elems[1].position = 0.258; elems[1].color = (0.46, 0.44, 0.41, 1.0)
    e_ir = cr.color_ramp.elements.new(0.118)
    e_ir.color = (ir_lin[0] * 1.15, ir_lin[1] * 1.08, ir_lin[2] * 1.02, 1.0)
    e_limb = cr.color_ramp.elements.new(0.232)
    e_limb.color = (ir_lin[0] * 0.32, ir_lin[1] * 0.32, ir_lin[2] * 0.32, 1.0)
    L(rad.outputs["Value"], cr.inputs["Fac"])

    mix_col = n.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.blend_type = "MULTIPLY"
    mix_col.inputs[0].default_value = 0.20
    L(cr.outputs["Color"], mix_col.inputs[6])
    L(ns.outputs["Color"], mix_col.inputs[7])

    # Upper-eyelid ambient shadow gradient along V (`shades upper sclera & top of iris softly under the relaxed upper lid!`)
    lid_sh = n.new("ShaderNodeMapRange")
    lid_sh.clamp = True
    lid_sh.inputs["From Min"].default_value = 0.48
    lid_sh.inputs["From Max"].default_value = 0.72
    lid_sh.inputs["To Min"].default_value = 1.0
    lid_sh.inputs["To Max"].default_value = 0.44
    L(sep.outputs["Y"], lid_sh.inputs["Value"])

    sh_mul = n.new("ShaderNodeMix")
    sh_mul.data_type = "RGBA"
    sh_mul.blend_type = "MULTIPLY"
    sh_mul.inputs[0].default_value = 1.0
    L(mix_col.outputs[2], sh_mul.inputs[6])
    L(lid_sh.outputs["Result"], sh_mul.inputs[7])
    L(sh_mul.outputs[2], bsdf.inputs["Base Color"])
    return mat


def _attach_and_aim_eyes(basemesh, iris_rgb_srgb: Tuple[float, float, float], gaze_yaw_ratio: float):
    """Attach MPFB2 high-poly eyes, map crisp front-pole iris UVs, and orient gaze smoothly."""
    from bl_ext.user_default.mpfb.services.assetservice import AssetService
    from bl_ext.user_default.mpfb.services.humanservice import HumanService

    eye_mhclo = str(AssetService.list_mhclo_assets("eyes")[0])
    eyes = HumanService.add_mhclo_asset(
        eye_mhclo, basemesh, asset_type="Eyes", subdiv_levels=1, material_type="MAKESKIN", set_up_rigging=False
    )
    eye_mat = _build_realistic_eye_shader("RealisticEyeShader", iris_rgb_srgb)
    eyes.data.materials.clear()
    eyes.data.materials.append(eye_mat)

    if not eyes.data.uv_layers:
        eyes.data.uv_layers.new(name="UVMap")
    uv_data = eyes.data.uv_layers["UVMap"].data

    # Compute exact left & right sphere centers and assign front-pole iris UV BEFORE gaze rotation
    eye_centers = {}
    eye_radii = {}
    for side_key, side_fn in (("L", lambda x: x > 0.0), ("R", lambda x: x < 0.0)):
        vs = [v for v in eyes.data.vertices if side_fn(v.co.x)]
        if vs:
            c = sum((v.co for v in vs), Vector()) / len(vs)
            r = max(0.010, max((v.co - c).length for v in vs))
            eye_centers[side_key] = c
            eye_radii[side_key] = r

    for poly in eyes.data.polygons:
        pc = poly.center
        sk = "L" if pc.x > 0.0 else "R"
        c = eye_centers.get(sk)
        r = eye_radii.get(sk, 0.0125)
        if not c:
            continue
        for li in poly.loop_indices:
            co = eyes.data.vertices[eyes.data.loops[li].vertex_index].co
            if co.y <= c.y:
                u = 0.5 + (co.x - c.x) / (2.0 * r)
                v_uv = 0.5 + (co.z - c.z) / (2.0 * r)
                uv_data[li].uv = (u, v_uv)
            else:
                uv_data[li].uv = (0.95, 0.95)

    # Seat eyeballs naturally inside the gently hooded eyelids (-0.3mm in Y, 1.005x scale) and aim gaze smoothly (`zero bulging stare!`)
    yaw_rad = max(-0.35, min(0.35, gaze_yaw_ratio * 0.55))
    rot = Matrix.Rotation(yaw_rad, 3, "Z") if abs(yaw_rad) > 0.01 else Matrix.Identity(3)
    for sk, side_fn in (("L", lambda x: x > 0.0), ("R", lambda x: x < 0.0)):
        c = eye_centers.get(sk)
        if c is None:
            continue
        for v in eyes.data.vertices:
            if side_fn(v.co.x):
                v.co = c + (rot @ (v.co - c)) * 1.005 + Vector((0.0, -0.0003, 0.0))
    eyes.data.update()

    for poly in eyes.data.polygons:
        poly.use_smooth = True
    return eyes


def _make_hair_curves_mat(
    name: str,
    use_point_color: bool = True,
    fallback_rgb: Tuple[float, float, float] = (0.75, 0.74, 0.72),
    hair_mix: float = 0.22,
    root_min: float = 0.24,
):
    """
    Build a glare-free matte-mixed Cycles hair shader driven by projected `curve_color` (`POINT` FLOAT_COLOR)
    and light-adaptive `HairInfo.Intercept` (`high root_min on silver/white hair avoids dark forehead bands`).
    """
    mat = bpy.data.materials.new(name)
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    out = n.new("ShaderNodeOutputMaterial")
    bsdf_p = n.new("ShaderNodeBsdfPrincipled")
    bsdf_h = n.new("ShaderNodeBsdfHairPrincipled")
    bsdf_h.parametrization = "COLOR"

    hi = n.new("ShaderNodeHairInfo")
    root_fac = n.new("ShaderNodeMapRange")
    root_fac.clamp = True
    root_fac.inputs["From Min"].default_value = 0.0
    root_fac.inputs["From Max"].default_value = 0.50
    root_fac.inputs["To Min"].default_value = float(root_min)
    root_fac.inputs["To Max"].default_value = 0.96
    L(hi.outputs["Intercept"], root_fac.inputs["Value"])

    if use_point_color:
        attr = n.new("ShaderNodeAttribute")
        attr.attribute_name = "curve_color"
        mul_c = n.new("ShaderNodeMix")
        mul_c.data_type = "RGBA"
        mul_c.blend_type = "MULTIPLY"
        mul_c.inputs[0].default_value = 1.0
        L(attr.outputs["Color"], mul_c.inputs[6])
        L(root_fac.outputs["Result"], mul_c.inputs[7])
        col_out = mul_c.outputs[2]
    else:
        lin = tuple(max(0.015, (c / 1.055) ** 2.2) for c in fallback_rgb)
        cr = n.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].position = 0.05
        cr.color_ramp.elements[0].color = (lin[0] * 0.24, lin[1] * 0.24, lin[2] * 0.24, 1.0)
        cr.color_ramp.elements[1].position = 0.90
        cr.color_ramp.elements[1].color = (min(0.90, lin[0] * 0.98), min(0.90, lin[1] * 0.98), min(0.90, lin[2] * 0.98), 1.0)
        L(hi.outputs["Intercept"], cr.inputs["Fac"])
        col_out = cr.outputs["Color"]

    # Soft matte Principled fiber base prevents Cycles multiple-scattering whiteout on light/silver strands
    L(col_out, bsdf_p.inputs["Base Color"])
    if "Roughness" in bsdf_p.inputs:
        bsdf_p.inputs["Roughness"].default_value = 0.84
    if "Coat Weight" in bsdf_p.inputs:
        bsdf_p.inputs["Coat Weight"].default_value = 0.0
    if "Coat" in bsdf_p.inputs:
        bsdf_p.inputs["Coat"].default_value = 0.0
    if "Subsurface Weight" in bsdf_p.inputs:
        bsdf_p.inputs["Subsurface Weight"].default_value = 0.05
    if "Specular IOR Level" in bsdf_p.inputs:
        bsdf_p.inputs["Specular IOR Level"].default_value = 0.06

    if "Color" in bsdf_h.inputs:
        L(col_out, bsdf_h.inputs["Color"])
    if "Roughness" in bsdf_h.inputs:
        bsdf_h.inputs["Roughness"].default_value = 0.70
    if "Radial Roughness" in bsdf_h.inputs:
        bsdf_h.inputs["Radial Roughness"].default_value = 0.72
    if "Coat" in bsdf_h.inputs:
        bsdf_h.inputs["Coat"].default_value = 0.0
    if "IOR" in bsdf_h.inputs:
        bsdf_h.inputs["IOR"].default_value = 1.55
    if "Random Roughness" in bsdf_h.inputs:
        bsdf_h.inputs["Random Roughness"].default_value = 0.35

    mix_sh = n.new("ShaderNodeMixShader")
    mix_sh.inputs["Fac"].default_value = float(max(0.0, min(1.0, hair_mix)))
    L(bsdf_p.outputs["BSDF"], mix_sh.inputs[1])
    L(bsdf_h.outputs["BSDF"], mix_sh.inputs[2])
    L(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _build_cycles_hair_curves_obj(
    name: str,
    strands_pts: List[List[Tuple[float, float, float]]],
    strands_radii: List[List[float]],
    material,
    strands_colors: Optional[List[List[Tuple[float, float, float, float]]]] = None,
):
    """Create a fast Blender 5.2 Cycles `CURVES` groom object via bulk `foreach_set` + optional point colors."""
    hc = bpy.data.hair_curves.new(name)
    hc.add_curves([len(p) for p in strands_pts])

    flat_pos, flat_rad = [], []
    for pts, rads in zip(strands_pts, strands_radii):
        for p, r in zip(pts, rads):
            flat_pos.extend((float(p[0]), float(p[1]), float(p[2])))
            flat_rad.append(float(r))

    hc.points.foreach_set("position", flat_pos)
    hc.points.foreach_set("radius", flat_rad)

    # Render as smooth Catmull-Rom curves (`0=CATMULL_ROM`) so locks have ZERO polygonal bends or straight kinks!
    try:
        ct_attr = hc.attributes.get("curve_type") or hc.attributes.new("curve_type", "INT8", "CURVE")
        ct_attr.data.foreach_set("value", [0] * len(strands_pts))
        res_attr = hc.attributes.get("resolution") or hc.attributes.new("resolution", "INT", "CURVE")
        res_attr.data.foreach_set("value", [6] * len(strands_pts))
    except Exception:
        pass

    if strands_colors is not None:
        attr = hc.attributes.new("curve_color", "FLOAT_COLOR", "POINT")
        flat_col = [c for cols in strands_colors for rgba in cols for c in rgba]
        attr.data.foreach_set("color", flat_col)

    ob = bpy.data.objects.new(name, hc)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


def _sample_photo_grid(grid: List[List[float]], u_norm: float, v_norm: float) -> float:
    """Fast bilinear grid lookup for hair halo silhouette or photo luminance."""
    res = len(grid)
    x = max(0.0, min(res - 1.001, u_norm * (res - 1)))
    y = max(0.0, min(res - 1.001, v_norm * (res - 1)))
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    v00 = grid[y0][x0]
    v10 = grid[y0][x0 + 1]
    v01 = grid[y0 + 1][x0]
    v11 = grid[y0 + 1][x0 + 1]
    return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy


def _build_scalp_undercoat_mesh(
    basemesh,
    analysis: Dict,
    fit_cam: Optional[Dict] = None,
):
    """
    Dark-root flush posterior undercoat ONLY for dark hair (`med_luma <= 0.42`).
    Skipped cleanly on bright/silver hair (`dense multi-layered 3D crown hair_curves handle full coverage
    without ever exposing a synthetic striped helmet edge in Clay, Matched, or Side views!`).
    """
    if "scalp" not in basemesh.vertex_groups:
        return None
    hstats = analysis.get("hair_stats", {})
    if float(hstats.get("med_luma", 0.72)) > 0.42:
        return None
    scalp_idx = basemesh.vertex_groups["scalp"].index
    weights = {}
    for v in basemesh.data.vertices:
        w = next((g.weight for g in v.groups if g.group == scalp_idx), 0.0)
        if w > 0.25:
            weights[v.index] = float(w)
    if len(weights) < 40:
        return None

    bm = bmesh.new()
    v_map = {}
    for p in basemesh.data.polygons:
        if not all(vi in weights for vi in p.vertices):
            continue
        p_cos = [basemesh.data.vertices[vi].co for vi in p.vertices]
        if any(co.y < -0.005 for co in p_cos):
            continue
        f_verts = []
        for vi in p.vertices:
            if vi not in v_map:
                bv = basemesh.data.vertices[vi]
                w = weights[vi]
                edge_t = max(0.0, min(1.0, (w - 0.55) / 0.35))
                back_t = max(0.0, min(1.0, (bv.co.y - 0.002) / 0.030))
                off_n = -0.008 * (1.0 - edge_t * back_t) + 0.0008 * (edge_t * back_t)
                v_map[vi] = bm.verts.new(bv.co + bv.normal * off_n)
            f_verts.append(v_map[vi])
        try:
            bm.faces.new(f_verts)
        except ValueError:
            pass

    if not bm.faces:
        bm.free()
        return None

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    smesh = bpy.data.meshes.new("ScalpRootShell")
    bm.to_mesh(smesh)
    bm.free()

    sob = bpy.data.objects.new("ScalpRootShell", smesh)
    bpy.context.collection.objects.link(sob)
    for p in sob.data.polygons:
        p.use_smooth = True
    sob.visible_shadow = False

    sm = sob.modifiers.new("ScalpSmooth", type="SMOOTH")
    sm.factor = 0.35
    sm.iterations = 2
    sub = sob.modifiers.new("ScalpSubsurf", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 1

    hair_luma = float(hstats.get("med_luma", 0.20))
    med_srgb = hstats.get("med_rgb_srgb", [0.18, 0.14, 0.11])
    lin_h = tuple(max(0.012, (float(c) / 1.055) ** 2.2) for c in med_srgb)

    mat = bpy.data.materials.new("ScalpRootUndercoatMat")
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.90
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.05
    tc = n.new("ShaderNodeTexCoord")
    ns = n.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 145.0; ns.inputs["Detail"].default_value = 8.0
    L(tc.outputs["Object"], ns.inputs["Vector"])
    cr = n.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.25; cr.color_ramp.elements[0].color = (lin_h[0] * 0.12, lin_h[1] * 0.12, lin_h[2] * 0.12, 1.0)
    cr.color_ramp.elements[1].position = 0.85; cr.color_ramp.elements[1].color = (lin_h[0] * 0.46, lin_h[1] * 0.46, lin_h[2] * 0.46, 1.0)
    L(ns.outputs["Fac"], cr.inputs["Fac"])
    L(cr.outputs["Color"], bsdf.inputs["Base Color"])
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])
    sob.data.materials.append(mat)
    return sob


def build_curly_hair_and_facial_hair(
    basemesh,
    analysis: Dict,
    fit_cam: Optional[Dict] = None,
    cam_obj=None,
    seed: int = 42,
):
    """
    Build Dual-Layer Universal Portrait Hair + Photo-Preserved Facial Hair:
      1. `PhotoHairVolume`: Depth-sculpted volumetric hair mesh wrapped cleanly behind skull.
      2. `PortraitCurlyHair`: ~950 fine matte-mixed perimeter halo curls & loose crown tendrils.
      3. `PortraitMustache`: ~340 ultra-fine crisscrossed whiskers layered over preserved photo lip texture.
      4. `PortraitEyebrows`: ~125 fine brow hairs per side blended into photo brow creases.
    """
    rng = random.Random(seed)
    hstats = analysis["hair_stats"]
    hguide = analysis.get("hair_guide")
    is_mono = bool(analysis.get("color_metrics", {}).get("is_monochrome", False))
    colorize = bool(analysis.get("color_metrics", {}).get("colorize_monochrome", False))
    crown_lift = float(hstats.get("crown_lift_ratio", 0.35))
    flare_L = float(hstats.get("side_L_flare_ratio", 0.35))
    flare_R = float(hstats.get("side_R_flare_ratio", 0.35))
    hair_luma = float(hstats.get("med_luma", 0.72))
    med_srgb = tuple(hstats.get("med_rgb_srgb", [0.76, 0.75, 0.73] if hair_luma > 0.50 else [0.18, 0.14, 0.11]))
    med_lin = tuple(max(0.015, (float(c) / 1.055) ** 2.2) for c in med_srgb)
    med_lin_lum = max(1e-4, 0.2126 * med_lin[0] + 0.7152 * med_lin[1] + 0.0722 * med_lin[2])
    chroma_ratio = (
        (1.0, 1.0, 0.98)
        if (is_mono and not colorize)
        else tuple(max(0.45, min(1.55, c / med_lin_lum)) for c in med_lin)
    )

    use_pcol = bool(hguide and fit_cam and "cam_K" in fit_cam)
    W_img = float(analysis.get("width", 1024))
    H_img = float(analysis.get("height", 1024))
    if use_pcol:
        K = fit_cam["cam_K"]
        Rm = Matrix(fit_cam["cam_R"])
        tv = Vector(fit_cam["cam_t"])
        halo_grid = hguide["halo"]
        luma_grid = hguide["luma"]

        def _proj_uv(pt: Vector) -> Tuple[float, float]:
            xcv = Rm @ pt + tv
            zc = max(0.05, float(xcv.z))
            u = (K[0][0] * xcv.x / zc + K[0][2]) / W_img
            v = (K[1][1] * xcv.y / zc + K[1][2]) / H_img
            return u, v
    else:
        def _proj_uv(pt: Vector) -> Tuple[float, float]:
            return 0.5, 0.5

    def _pt_color(
        pt: Vector,
        default_srgb: Tuple[float, float, float],
        bias: float = 0.94,
        strand_mul: float = 1.0,
        gamma_pow: float = 1.95,
    ) -> Tuple[float, float, float, float]:
        if use_pcol:
            u, v = _proj_uv(pt)
            lum_s = _sample_photo_grid(luma_grid, u, v)
            val = max(0.030, min(0.95, ((lum_s * bias) ** gamma_pow) * strand_mul))
            return (
                max(0.025, min(0.95, val * chroma_ratio[0])),
                max(0.025, min(0.95, val * chroma_ratio[1])),
                max(0.025, min(0.95, val * chroma_ratio[2])),
                1.0,
            )
        lin = tuple(max(0.015, (c / 1.055) ** 2.2) for c in default_srgb)
        return (lin[0], lin[1], lin[2], 1.0)

    def _ray_march_halo_len(root_pt: Vector, sweep_dir: Vector, side_w: float, base_fallback: float, droop_scale: float) -> float:
        """March along candidate clump growth arc to hit exact 1:1 photo `hair_halo` outer boundary!"""
        if not use_pcol:
            return base_fallback
        u0, v0 = _proj_uv(root_pt)
        u_test, v_test = _proj_uv(root_pt + sweep_dir * 0.08)
        uv_drift = math.hypot(u_test - u0, v_test - v0)
        out_r = (root_pt - head_center).normalized()
        # Only skip back-center occiput points facing away with zero silhouette leverage
        if uv_drift < 0.006 or (out_r.y > 0.35 and abs(out_r.x) < 0.045):
            return base_fallback
        s_best = 0.0
        for step in range(1, 27):
            s = step * 0.014  # up to 37.8 cm for soaring curly crown & temple locks
            probe = root_pt + sweep_dir * s + Vector((0.0, 0.0, -droop_scale * ((s / 0.22) ** 2) * (0.45 + 0.75 * side_w)))
            up, vp = _proj_uv(probe)
            if not (0.0 <= up <= 1.0 and 0.0 <= vp <= 1.0):
                # Allow top/side wispy curls right to top frame border cleanly
                if step >= 4:
                    s_best = max(s_best, s * 0.96)
                break
            if _sample_photo_grid(halo_grid, up, vp) >= 0.09:
                s_best = s
        if s_best > 0.025:
            return max(0.048, min(0.35, s_best * rng.uniform(0.90, 1.08)))
        return base_fallback * rng.uniform(0.68, 0.95)

    def _clamp_in_halo(pt: Vector, prev_pt: Vector, t: float, root_y: float = 0.0) -> Vector:
        # Smooth organic tip-taper at photo halo boundary (`stops naturally as soft curly tip, ZERO bounce-back grid!`)
        if not use_pcol or t < 0.08 or root_y > 0.055:
            return pt
        u, v = _proj_uv(pt)
        delta = pt - prev_pt
        if not (0.004 <= u <= 0.996 and 0.004 <= v <= 0.996):
            return prev_pt + delta * 0.12
        hval = _sample_photo_grid(halo_grid, u, v)
        if hval < 0.16:
            keep = max(0.02, (hval / 0.16) ** 1.45)
            # Cleanly arrest outward growth right at photo contour (`ZERO straight vertical hanging tassels below ear!`)
            return prev_pt + delta * keep
        return pt

    # Layer 1: Optional flush posterior dark-hair root shell (skipped automatically for bright/silver hair)
    vol_hair = _build_scalp_undercoat_mesh(basemesh, analysis, fit_cam)

    # Authentic grey-silver depth (`soft silver hairline roots + deep slate-grey internal locks -> zero chalk white!`)
    head_rmin = 0.50 if hair_luma > 0.48 else 0.24
    hair_def = (0.72, 0.71, 0.69) if hair_luma > 0.50 else med_srgb
    mat_hair = _make_hair_curves_mat("PortraitHairMat", use_point_color=use_pcol, fallback_rgb=hair_def, hair_mix=0.28, root_min=head_rmin)
    mat_brow = _make_hair_curves_mat("PortraitBrowMat", use_point_color=use_pcol, fallback_rgb=(0.42, 0.40, 0.38), hair_mix=0.22, root_min=0.30)
    mat_must = _make_hair_curves_mat("PortraitMustacheMat", use_point_color=use_pcol, fallback_rgb=(0.76, 0.75, 0.73), hair_mix=0.26, root_min=0.56)

    gi_scalp = basemesh.vertex_groups["scalp"].index
    scalp_verts = [v for v in basemesh.data.vertices if any(g.group == gi_scalp and g.weight > 0.18 for g in v.groups)]
    if not scalp_verts:
        scalp_verts = [v for v in basemesh.data.vertices if v.co.z > 1.54]

    head_center = sum((v.co for v in scalp_verts), Vector()) / max(1, len(scalp_verts))
    head_center.z -= 0.032

    # Universal cranial scalp domain: clean cheekbone anteriorly, full crown + posterior/below-ear nape!
    zyg_l_z = float(basemesh.data.vertices[11794].co.z) if len(basemesh.data.vertices) > 11794 else float(head_center.z - 0.075)
    zyg_y_ref = float(basemesh.data.vertices[11794].co.y) if len(basemesh.data.vertices) > 11794 else -0.075
    brow_z_ref = float(basemesh.data.vertices[6979].co.z) if len(basemesh.data.vertices) > 6979 else float(head_center.z - 0.035)
    ear_y_ref = float(head_center.y) - 0.008

    scalp_wmap = {}
    for v in basemesh.data.vertices:
        w = next((g.weight for g in v.groups if g.group == gi_scalp), 0.0)
        is_post_skull = v.co.y > (ear_y_ref + 0.008)
        min_z_lat = (zyg_l_z - 0.006) if is_post_skull else (zyg_l_z + 0.014 if v.co.y < -0.022 else zyg_l_z + 0.004)
        in_temple_ear = (v.co.z > min_z_lat) and (v.co.y > zyg_y_ref - 0.002) and (abs(v.co.x - head_center.x) > 0.050 or is_post_skull)
        in_upper_crown = (v.co.z > brow_z_ref + 0.008) and (v.co.y > -0.114)
        if (w > 0.02 and v.co.z > zyg_l_z - 0.010) or in_temple_ear or in_upper_crown:
            scalp_wmap[v.index] = max(float(w), 0.78 if in_temple_ear else 0.52)
    scalp_tris, tri_areas = [], []
    for p in basemesh.data.polygons:
        if not all(scalp_wmap.get(vi, 0.0) > 0.15 for vi in p.vertices):
            continue
        if any(basemesh.data.vertices[vi].co.z < zyg_l_z - 0.012 for vi in p.vertices):
            continue
        pvs = [basemesh.data.vertices[vi] for vi in p.vertices]
        for k in range(1, len(pvs) - 1):
            a_tri = 0.5 * ((pvs[k].co - pvs[0].co).cross(pvs[k + 1].co - pvs[0].co)).length
            if a_tri > 1e-8:
                scalp_tris.append((pvs[0], pvs[k], pvs[k + 1]))
                tri_areas.append(a_tri)

    flare_avg = 0.5 * (flare_L + flare_R)
    puff_fac = min(1.0, max(0.0, (max(crown_lift, flare_avg) - 0.12) / 0.26))
    len_crown = 0.072 + 0.105 * crown_lift
    len_side_L = 0.072 + 0.120 * flare_L
    len_side_R = 0.072 + 0.120 * flare_R

    droop_hair = 0.024 if puff_fac > 0.5 else (0.048 - 0.024 * puff_fac)
    core_roots = []
    feather_roots = []
    post_ear_roots = []
    flyaway_roots = []
    crown_cover_roots = []
    n_cand = 5200
    if scalp_tris:
        chosen_tris = rng.choices(scalp_tris, weights=tri_areas, k=n_cand)
        for va, vb, vc in chosen_tris:
            r1 = math.sqrt(rng.random())
            r2 = rng.random()
            p_s = va.co * (1.0 - r1) + vb.co * (r1 * (1.0 - r2)) + vc.co * (r1 * r2)
            n_s = (va.normal * (1.0 - r1) + vb.normal * (r1 * (1.0 - r2)) + vc.normal * (r1 * r2)).normalized()
            is_front_forehead = (n_s.y < -0.45) and (abs(p_s.x - head_center.x) < 0.056) and (p_s.y < head_center.y - 0.015)
            is_post_loc = p_s.y > (ear_y_ref - 0.004)
            z_lat_min = (zyg_l_z - 0.042) if is_post_loc else (zyg_l_z + 0.014 if p_s.y < -0.022 else zyg_l_z + 0.002)
            is_lateral_temple = (abs(n_s.x) > 0.34 or abs(p_s.x - head_center.x) >= 0.050 or is_post_loc) and (p_s.y > zyg_y_ref - 0.002) and (p_s.z > z_lat_min)
            if use_pcol and is_front_forehead:
                z_wave = 0.008 * math.sin(p_s.x * 68.0) + 0.006 * math.cos(p_s.x * 155.0 + 1.2) + rng.uniform(-0.010, 0.010)
                uh, vh = _proj_uv(p_s + Vector((0.0, 0.0, z_wave)))
                hv = _sample_photo_grid(halo_grid, uh, vh)
                uh0, vh0 = _proj_uv(p_s)
                hv_true = _sample_photo_grid(halo_grid, uh0, vh0)
                hv_mix = 0.55 * hv + 0.45 * hv_true + rng.uniform(-0.07, 0.07)
                if hv_mix < 0.06:
                    continue
                p_keep = min(1.0, max(0.0, (hv_mix - 0.06) / 0.44)) ** 1.05
                if rng.random() > p_keep:
                    continue
                if hv_mix < 0.16:
                    feather_roots.append((p_s, n_s, hv_mix))
                else:
                    core_roots.append((p_s, n_s, hv_mix))
                    crown_cover_roots.append((p_s, n_s, hv_mix))
            elif is_lateral_temple:
                uh, vh = _proj_uv(p_s)
                hv_lat = _sample_photo_grid(halo_grid, uh, vh) if use_pcol else 0.78
                keep_lat = (puff_fac > 0.25) or (hv_lat > 0.16) or (p_s.y > ear_y_ref - 0.012)
                if not keep_lat:
                    continue
                core_roots.append((p_s, n_s, max(0.68, hv_lat)))
                crown_cover_roots.append((p_s, n_s, 0.88))
                if is_post_loc:
                    post_ear_roots.append((p_s, n_s, 0.88))
            else:
                core_roots.append((p_s, n_s, 0.90))
                if p_s.z > head_center.z - 0.022:
                    crown_cover_roots.append((p_s, n_s, 0.90))
                if is_post_loc and (abs(p_s.x - head_center.x) > 0.032 or p_s.y > head_center.y + 0.015) and (p_s.z > zyg_l_z - 0.040):
                    post_ear_roots.append((p_s, n_s, 0.90))
                if (p_s.z > head_center.z - 0.008) and (abs(p_s.x - head_center.x) > 0.044 or p_s.z > head_center.z + 0.052):
                    flyaway_roots.append((p_s, n_s, 0.90))
    if not core_roots:
        core_roots = [(v.co.copy(), v.normal.copy(), 0.75) for v in scalp_verts]
    if not crown_cover_roots:
        crown_cover_roots = core_roots
    if not post_ear_roots:
        post_ear_roots = core_roots
    if not flyaway_roots:
        flyaway_roots = core_roots

    # Prioritize anterior crown + above-ear temples + rich bilateral side volume (`dense photo-accurate side fluff!`)
    crown_front_roots = [r for r in crown_cover_roots if (r[0].y < head_center.y + 0.045 and r[0].z > brow_z_ref + 0.006)]
    temple_ear_roots = [r for r in crown_cover_roots if (abs(r[0].x - head_center.x) > 0.046 and r[0].z < head_center.z + 0.055)]
    side_vol_roots = [r for r in core_roots if abs(r[0].x - head_center.x) > 0.042]
    if side_vol_roots:
        core_roots = core_roots + side_vol_roots
    if crown_front_roots:
        crown_cover_roots = crown_cover_roots + crown_front_roots * 3
    if temple_ear_roots:
        crown_cover_roots = crown_cover_roots + temple_ear_roots * 3

    hair_pts, hair_rads, hair_cols = [], [], []
    n_major_clumps = 365
    n_wisp_clumps = 135
    n_post_ear = 95
    n_flyaway_clumps = 40
    n_crown_cover = 860
    n_feather_clumps = 68
    n_ctrl = 27

    c_pe_start = n_major_clumps + n_wisp_clumps
    c_fly_start = c_pe_start + n_post_ear
    c_cc_start = c_fly_start + n_flyaway_clumps
    c_fea_start = c_cc_start + n_crown_cover
    total_clumps = c_fea_start + (n_feather_clumps if feather_roots else 0)
    for c_idx in range(total_clumps):
        is_wisp = n_major_clumps <= c_idx < c_pe_start
        is_post_ear = c_pe_start <= c_idx < c_fly_start
        is_flyaway = c_fly_start <= c_idx < c_cc_start
        is_crown_cover = c_cc_start <= c_idx < c_fea_start
        is_feather = c_idx >= c_fea_start
        if is_feather:
            sv_co, sv_n, hv_r = rng.choice(feather_roots)
        elif is_crown_cover:
            sv_co, sv_n, hv_r = rng.choice(crown_cover_roots)
        elif is_post_ear:
            sv_co, sv_n, hv_r = rng.choice(post_ear_roots)
        elif is_flyaway:
            sv_co, sv_n, hv_r = rng.choice(flyaway_roots)
        else:
            sv_co, sv_n, hv_r = rng.choice(core_roots)

        # Keep crown blanket & major curls cleanly seated right on/above scalp surface (`never sunken inside dome`)
        n_off = 0.0002 if is_feather else (0.0012 if (sv_n.y < -0.45 and abs(sv_co.x - head_center.x) < 0.065) else 0.0022)
        g_root = sv_co + sv_n * n_off + Vector((rng.uniform(-0.0025, 0.0025), rng.uniform(-0.0025, 0.0025), rng.uniform(-0.0025, 0.0025)))
        r_scalp_local = max(0.065, float((sv_co - head_center).length))
        out_dir = (g_root - head_center).normalized()
        back_w = max(0.0, min(1.0, (out_dir.y - 0.05) / 0.85))
        side_w = min(1.0, max(0.0, (abs(out_dir.x) - 0.20) / 0.52))
        is_temple_ear = (abs(g_root.x - head_center.x) > 0.048) and (g_root.z < head_center.z + 0.048)
        is_post_nape = is_post_ear or ((g_root.y > ear_y_ref - 0.012) and (g_root.z < head_center.z + 0.045) and (abs(g_root.x - head_center.x) > 0.034))
        side_len = len_side_L if g_root.x > head_center.x else len_side_R
        base_slen = (len_crown * (1.0 - side_w) + side_len * side_w) * rng.uniform(0.92, 1.28)

        tang_side = Vector((-out_dir.y, out_dir.x, 0.0))
        if tang_side.length > 1e-4:
            tang_side.normalize()
        tang_sign = math.copysign(1.0, g_root.x - head_center.x) if abs(g_root.x - head_center.x) > 0.01 else rng.choice((-1.0, 1.0))

        is_fg = (sv_n.y < -0.45) and (abs(g_root.x - head_center.x) < 0.062)
        hv_t = min(1.0, max(0.0, (hv_r - 0.12) / 0.66)) if is_fg else 1.0

        if is_feather:
            # Fine soft foreground hairline wisps arch cleanly UP and back into crown blanket (NO flat skin kinks)
            flow_f = tang_side * (rng.uniform(-0.45, 0.45) * tang_sign)
            cur_dir = Vector((
                out_dir.x * 0.24 + flow_f.x + rng.uniform(-0.18, 0.18),
                0.42 + max(0.0, out_dir.y) * 0.22 + rng.uniform(-0.06, 0.14),
                0.76 + rng.uniform(-0.10, 0.20),
            )).normalized()
            slen = base_slen * rng.uniform(0.42, 0.72)
            is_shadow_layer = False  # never draw black single wire hairs on exposed forehead skin!
        elif is_crown_cover:
            # Crisscrossed tangential crown & temple wave locks sweep diagonally ACROSS skull dome (ZERO bald khopdi or temple band!)
            cross_dir = rng.choice((-1.0, 1.0)) * rng.uniform(0.55, 1.15)
            flow_swirl = tang_side * cross_dir
            y_c_arch = max(0.04, out_dir.y * 0.24 + flow_swirl.y * 0.35 + 0.16 + rng.uniform(-0.08, 0.12))
            z_c_arch = (0.28 * (1.0 - back_w) - 0.32 * back_w - 0.22 * float(is_post_nape)) + rng.uniform(-0.16, 0.24)
            cur_dir = (
                Vector((out_dir.x * 0.22 + flow_swirl.x, y_c_arch, z_c_arch)).normalized()
                + out_dir * rng.uniform(0.14, 0.30)
            ).normalized()
            slen = base_slen * rng.uniform(0.52, 0.84)
            is_shadow_layer = (not is_fg) and (rng.random() < 0.38)
        else:
            # Cohesive wavy/curly outer volume locks (`smooth C2 waves & gentle S-bends, ZERO radial bristle spokes!`)
            swirl_w = rng.uniform(-0.62, 0.62) * tang_sign
            y_fore = (out_dir.y * 0.08 + 0.28 * (1.0 - 0.45 * hv_t)) if out_dir.y < 0.0 else (out_dir.y * 0.34)
            y_comp = y_fore * (1.0 - 0.46 * back_w) + 0.16 * back_w + (0.14 if is_post_nape else 0.0)
            z_comp = (
                0.78 * (1.0 - side_w) * (0.68 + 0.32 * hv_t)
                + (0.34 if is_temple_ear else 0.26) * side_w
                - (0.38 + 0.56 * back_w) * (0.72 - 0.28 * puff_fac)
                - (0.24 if is_post_nape else 0.0)
                + rng.uniform(-0.18, 0.28)
            )
            cur_dir = Vector((
                out_dir.x * (0.56 + 0.42 * side_w + 0.20 * float(is_temple_ear)) + tang_side.x * swirl_w + rng.uniform(-0.18, 0.18),
                y_comp + tang_side.y * swirl_w * 0.50 + rng.uniform(-0.10, 0.12),
                z_comp,
            )).normalized()
            is_shadow_layer = (not is_fg and not is_wisp and not is_flyaway) and (rng.random() < 0.32)
            s_target = _ray_march_halo_len(g_root, cur_dir, side_w, base_slen, droop_hair)
            back_scale = 1.0 - 0.40 * back_w
            if is_flyaway:
                slen = s_target * rng.uniform(0.92, 1.12) * back_scale
            else:
                slen = s_target * back_scale * (rng.uniform(0.64, 0.86) if is_shadow_layer else (rng.uniform(0.92, 1.12) if is_wisp else rng.uniform(0.86, 1.08)))

        ortho1 = cur_dir.cross(Vector((0.0, 0.0, 1.0)))
        ortho1 = ortho1.normalized() if ortho1.length > 1e-4 else Vector((1.0, 0.0, 0.0))
        ortho2 = cur_dir.cross(ortho1).normalized()

        curl_fg_scale = (0.48 + 0.44 * hv_t) if is_fg else 1.0
        # Smooth C2 Flowing S-Wave & Progressive Mid-to-Tip Curl Integrator (`soft curling wavy tips — ZERO straight brush bristles!`)
        curl_cycles = rng.uniform(1.45, 2.75 + 0.75 * puff_fac)
        curl_turn_rad = (0.068 + 0.118 * puff_fac) * curl_fg_scale * (0.55 if is_feather else 1.0)
        wave_lat_rad = rng.uniform(0.048, 0.095 + 0.055 * puff_fac) * curl_fg_scale
        grav_mul = (0.14 if (is_fg or is_feather) else (0.28 if is_crown_cover else 1.0))
        grav_step = (0.048 + 0.072 * side_w + 0.042 * back_w) * grav_mul / max(1, n_ctrl - 1)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        phase2 = phase + rng.uniform(0.7, 2.3)
        ds = slen / max(1, n_ctrl - 1)

        guide_pts = [g_root]
        tang = cur_dir.copy()
        max_arc_lift = 0.012 if is_crown_cover else 0.018
        for j in range(1, n_ctrl):
            t = j / (n_ctrl - 1)
            # Progressive tip-curl ramp (`curls curl & wave more softly toward mid/tips instead of fanning straight!`)
            curl_env = 0.55 + 1.15 * t
            spin_a = phase + t * curl_cycles * math.pi * 1.35
            bend_vec = ortho1 * math.sin(spin_a) + ortho2 * math.cos(spin_a * 0.85 + phase2)
            secondary_vec = ortho2 * math.sin(t * curl_cycles * math.pi * 1.75 + phase2)
            gz = -grav_step * (0.30 + 1.45 * t) if not ((is_fg or is_feather) and t < 0.72) else 0.0
            tang = (tang + bend_vec * (curl_turn_rad * curl_env) + secondary_vec * (wave_lat_rad * curl_env) + Vector((0.0, 0.0, gz))).normalized()
            # Gentle tangent-space steering (`never snap raw coordinates into L-brackets!`)
            if is_fg or is_feather:
                if tang.y < 0.02:
                    tang.y = 0.02
                if tang.z < 0.05 and t < 0.55:
                    tang.z = 0.05
                tang.normalize()
            raw_pt = guide_pts[-1] + tang * ds
            # Keep surface-blanketing crown/temple curls OUTSIDE MBLab skull dome (`eliminates bald khopdi!`)
            if is_crown_cover or is_post_ear:
                r_vec = raw_pt - head_center
                r_cur = r_vec.length
                r_min = min(0.145, max(r_scalp_local, 0.084) + (0.0030 + max_arc_lift * math.sin(math.pi * min(1.0, t))))
                if r_cur < r_min and r_cur > 1e-4:
                    raw_pt = head_center + r_vec * (r_min / r_cur)
            # Keep scalp strands cleanly behind anterior face/nose coronal plane (`zero floating nose wisps in side view!`)
            if raw_pt.y < (zyg_y_ref - 0.006) and raw_pt.z < (brow_z_ref + 0.024):
                raw_pt.y = zyg_y_ref - 0.006 + 0.002 * t
            clamped_pt = _clamp_in_halo(raw_pt, guide_pts[-1], t, g_root.y)
            fly_relax = 0.22 if is_flyaway else 0.20
            nxt_pt = clamped_pt + (raw_pt - clamped_pt) * (fly_relax * (1.0 - 0.40 * t))
            if (nxt_pt - head_center).length > 0.34:
                nxt_pt = head_center + (nxt_pt - head_center).normalized() * 0.34
            guide_pts.append(nxt_pt)

        # Authentic grey-silver lock hierarchy (`soft silver hairline + rich dark-slate/mid-grey/silver-white hair mass`)
        if is_shadow_layer:
            layer_shade = rng.uniform(0.24, 0.48)
        elif is_fg or is_feather:
            layer_shade = rng.uniform(0.74, 1.00)
        elif is_wisp or is_flyaway:
            layer_shade = rng.uniform(0.76, 1.04)
        elif rng.random() < 0.46:
            layer_shade = rng.uniform(0.42, 0.68)
        else:
            layer_shade = rng.uniform(0.78, 1.02)

        if is_feather:
            f_t = max(0.0, min(1.0, (hv_r - 0.10) / 0.25))
            n_ch = 2 + int(round(2.0 * f_t))
        elif is_crown_cover:
            n_ch = 7
        elif is_post_ear:
            n_ch = 7
        elif is_flyaway:
            n_ch = 2
        elif is_wisp:
            n_ch = 5 if side_w > 0.35 else 4
        else:
            n_ch = max(4, int(round(4 + 10 * hv_t))) + (2 if side_w > 0.35 else 0)

        for _ in range(n_ch):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            is_cohesive_lock = (not is_wisp and not is_feather and not is_flyaway)
            rr = rng.uniform(0.0006, 0.0040) if is_cohesive_lock else rng.uniform(0.0008, 0.0056)
            off_base = ortho1 * (math.cos(ang) * rr) + ortho2 * (math.sin(ang) * rr)
            ch_phase = rng.uniform(0.0, 2.0 * math.pi)
            ch_phase2 = rng.uniform(0.0, 2.0 * math.pi)
            ch_wfreq1 = rng.uniform(1.6, 2.7) * math.pi
            ch_wfreq2 = rng.uniform(2.6, 3.8) * math.pi
            ch_wamp = (0.0020 + 0.0034 * puff_fac) * (0.68 if is_cohesive_lock else 1.15) * rng.uniform(0.60, 1.30)
            len_mul = rng.uniform(0.58, 1.06) if (is_flyaway or is_wisp or is_feather) else rng.uniform(0.74, 1.05)
            tip_flare = rng.uniform(0.35, 0.90) if is_cohesive_lock else rng.uniform(0.80, 1.50)
            if is_feather:
                f_t = max(0.0, min(1.0, (hv_r - 0.10) / 0.25))
                r_root = rng.uniform(0.000038, 0.000060 + 0.000020 * f_t)
            elif is_flyaway:
                r_root = rng.uniform(0.000042, 0.000068)
            elif is_crown_cover:
                r_root = rng.uniform(0.000088, 0.000138)
            elif is_wisp:
                r_root = rng.uniform(0.000054, 0.000088)
            else:
                r_root = rng.uniform(0.000084, 0.000134)
            strand_shade = max(0.28, min(1.14, layer_shade * rng.uniform(0.76, 1.14)))
            pts_c, rads_c, cols_c = [], [], []
            prev_p = g_root
            for j in range(n_ctrl):
                t = j / (n_ctrl - 1)
                gp = guide_pts[0] if j == 0 else g_root + (guide_pts[j] - g_root) * len_mul
                env_c = min(1.0, t * 2.0) * (t ** 0.78)
                friz = (
                    ortho1 * ((math.sin(ch_wfreq1 * t + ch_phase) * 0.68 + math.sin(ch_wfreq2 * t + ch_phase2) * 0.32) * ch_wamp * env_c)
                    + ortho2 * ((math.cos(ch_wfreq1 * 0.90 * t + ch_phase2) * 0.62 + math.cos(ch_wfreq2 * 1.08 * t + ch_phase) * 0.38) * ch_wamp * env_c)
                )
                # Taper lock tips gracefully (`peak spread mid-lock, curling softly together at tip — zero broom ends!`)
                flare_env = 0.42 + tip_flare * math.sin(math.pi * 0.78 * t) if is_cohesive_lock else (0.45 + tip_flare * t)
                pos = gp + off_base * flare_env + friz
                if j > 0:
                    if pos.y < (zyg_y_ref - 0.006) and pos.z < (brow_z_ref + 0.024):
                        pos.y = zyg_y_ref - 0.006 + 0.002 * t
                    c_pos = _clamp_in_halo(pos, prev_p, t, g_root.y)
                    fly_relax = 0.20 if is_flyaway else 0.16
                    pos = c_pos + (pos - c_pos) * (fly_relax * (1.0 - 0.40 * t))
                    if (pos - head_center).length > 0.34:
                        pos = head_center + (pos - head_center).normalized() * 0.34
                prev_p = pos
                pts_c.append((pos.x, pos.y, pos.z))
                rads_c.append(r_root * (1.0 - 0.74 * (t ** 1.25)))
                root_fade = (0.90 + 0.10 * t) if (is_feather or is_fg) else (0.76 + 0.24 * t)
                cols_c.append(_pt_color(pos, hair_def, bias=0.86, strand_mul=strand_shade * root_fade, gamma_pow=1.92))
            hair_pts.append(pts_c)
            hair_rads.append(rads_c)
            hair_cols.append(cols_c)

    hair_ob = _build_cycles_hair_curves_obj(
        "PortraitCurlyHair", hair_pts, hair_rads, mat_hair, strands_colors=hair_cols if use_pcol else None
    )

    # 2. Universal Surface-Riding Exterior 3D Mustache (`strictly ABOVE mouth line & IN FRONT OF upper lip -> mouth & lower lip 100% visible!`)
    fh = analysis.get("facial_hair", {})
    m_dens = float(fh.get("mustache_density", 0.78))
    m_span_r = float(fh.get("mustache_span_ratio", 0.76))
    m_droop_r = float(fh.get("mustache_droop_ratio", 0.88))

    subn = basemesh.data.vertices[317].co.copy()
    ulip_c = basemesh.data.vertices[509].co.copy()
    mc_l = basemesh.data.vertices[10372].co.copy()
    mc_r = basemesh.data.vertices[3704].co.copy()
    mc_mid_y = 0.5 * (mc_l.y + mc_r.y)
    mc_mid_z = 0.5 * (mc_l.z + mc_r.z)
    # Photo-matched mustache width (`extends ~4-5mm past each mouth corner, never oversized handlebars`)
    half_span_x = max(0.023, min(0.76, m_span_r) * abs(mc_l.x - mc_r.x) * 0.76)
    lip_h = max(0.011, float(subn.z - ulip_c.z - 0.0010))
    dy_lat = max(0.004, float(mc_mid_y - ulip_c.y)) * 0.90
    dz_lat = max(0.0020, float(ulip_c.z - mc_mid_z) + 0.0014)

    def _lip_surface_pt(u_val: float, s_v: float) -> Vector:
        """
        Evaluate true 3D upper-lip cutaneous surface for u_val in [-1, 1] and s_v in [0, 1]
        (s_v=0 is the upper-lip vermilion margin ABOVE the mouth cleft; s_v=1 is right under the nostrils).
        """
        s_cl = max(-0.04, min(1.0, s_v))
        rx = ulip_c.x + u_val * half_span_x
        # Keep top philtrum root plane aligned with protruding upper lip vermilion so hairs NEVER dive backward (+Y) into mouth!
        y_c = min(ulip_c.y, subn.y) + 0.0012 * max(0.0, s_cl)
        ry = y_c + dy_lat * (u_val ** 2)
        rz = (ulip_c.z + 0.0014) + s_cl * lip_h - dz_lat * (u_val ** 2)
        return Vector((rx, ry, rz))

    must_pts, must_rads, must_cols = [], [], []
    if m_dens >= 0.14:
        m_lo = float(fh.get("luma_lo", 0.22))
        m_hi = float(fh.get("luma_hi", 0.65))
        c_span = max(0.0, min(1.0, (m_hi - m_lo) / max(0.08, m_hi) - 0.08))
        # Dense 4-tier volumetric brush (`~390` tufts x `9-10` Catmull-Rom strands = `~3,700` soft fluffy whiskers!)
        n_tufts = int(155 + 255 * m_dens)

        for t_idx in range(n_tufts):
            u = rng.uniform(-0.96, 0.96)
            # Roots sit strictly on the upper lip cutaneous skin (`s_v in [0.22, 0.96]`, NEVER inside the mouth opening!)
            s_v_root = rng.uniform(0.22, 0.96)
            tier = t_idx % 5
            is_dark_core = (tier == 0)
            is_inner_pillow = (tier == 1)
            is_bright_guard = (tier in (3, 4))

            if is_dark_core:
                tuft_shade = rng.uniform(0.32, 0.56) * (0.55 + 0.45 * c_span)
            elif is_inner_pillow:
                tuft_shade = rng.uniform(0.68, 0.90)
            elif is_bright_guard:
                tuft_shade = rng.uniform(1.10, 1.42) * (0.75 + 0.25 * c_span)
            else:
                tuft_shade = rng.uniform(0.90, 1.12)

            # Soft outward chevron splay (`no stiff vertical comb teeth!`)
            part_u = (0.10 * math.copysign(1.0, u) * (0.30 + 0.70 * abs(u)) if abs(u) > 0.03 else rng.uniform(-0.06, 0.06))
            ph_t = rng.uniform(0.0, 6.28)
            n_ch_m = 10 if (is_inner_pillow or is_dark_core) else 9
            for _ in range(n_ch_m):
                u0 = max(-1.05, min(1.05, u + rng.uniform(-0.055, 0.055)))
                sv0 = max(0.18, min(0.98, s_v_root + rng.uniform(-0.06, 0.06)))
                # Tips terminate right at the upper-lip vermilion margin (`sv_end in [-0.03, 0.22]`), leaving mouth cleft & lower lip 100% visible!
                corner_extra = 0.06 * (abs(u0) ** 1.4) * m_droop_r
                sv_end = max(-0.04 - corner_extra, min(0.24, -0.02 + 0.20 * (sv0 ** 1.3) + rng.uniform(-0.04, 0.05) - corner_extra))
                u_splay = part_u + u0 * 0.14 + rng.uniform(-0.14, 0.14)

                br = rng.uniform(0.00014, 0.00020) if (is_dark_core or is_inner_pillow) else rng.uniform(0.00012, 0.00017)
                ph_m = ph_t + rng.uniform(-0.85, 0.85)
                sp_w = tuft_shade * rng.uniform(0.86, 1.12)
                puff_out = (0.0024 if is_inner_pillow else 0.0038) * rng.uniform(0.82, 1.18)
                jx = rng.uniform(-0.0008, 0.0008)
                jz = rng.uniform(-0.0006, 0.0006)

                pts_m, rads_m, cols_m = [], [], []
                for j in range(9):
                    t = j / 8.0
                    u_curr = u0 + u_splay * (t ** 1.12)
                    sv_curr = sv0 * (1.0 - (t ** 1.05)) + sv_end * (t ** 1.05)
                    surf_pt = _lip_surface_pt(u_curr, sv_curr)
                    # Strictly exterior (-Y) rounded puff cushion: ALWAYS in front of upper lip, NEVER goes backward (+Y) into mouth!
                    delta_y_out = -(0.0009 + puff_out * math.sin(math.pi * (0.10 + 0.78 * t)))
                    # Organic 3D curl & wave breaks up parallel lines into soft, fluffy intertwined tufts
                    w_env = math.sin(math.pi * (0.15 + 0.75 * t))
                    curl_j = Vector((
                        0.0018 * math.sin(t * 5.2 + ph_m) + jx,
                        -0.0006 * abs(math.cos(t * 4.4 + ph_m)),
                        0.0009 * math.sin(t * 4.8 + ph_m * 1.2) + jz,
                    )) * w_env
                    pos = surf_pt + Vector((0.0, delta_y_out, 0.0)) + curl_j
                    # Hard anatomical floor above lower lip & mouth cleft so mouth is ALWAYS clearly visible!
                    mouth_floor_z = (ulip_c.z + 0.0005) - (dz_lat + 0.0010) * (u_curr ** 2)
                    if pos.z < mouth_floor_z:
                        pos.z = mouth_floor_z
                    pts_m.append((pos.x, pos.y, pos.z))
                    rads_m.append(br * (1.0 - 0.54 * (t ** 1.20)))
                    root_fade = 0.84 + 0.16 * t
                    cols_m.append(_pt_color(pos, (0.80, 0.79, 0.77), bias=0.35, strand_mul=sp_w * root_fade, gamma_pow=1.45))
                must_pts.append(pts_m)
                must_rads.append(rads_m)
                must_cols.append(cols_m)
    must_ob = _build_cycles_hair_curves_obj(
        "PortraitMustache", must_pts, must_rads, mat_must, strands_colors=must_cols if use_pcol else None
    )

    # 3. Fine Layered Eyebrows (~125 hairs per side blended cleanly into photo brow shadow)
    brow_pts, brow_rads, brow_cols = [], [], []
    for inner_vid, mid_vid, outer_vid, side_s in ((6957, 6979, 7024, 1.0), (181, 204, 255, -1.0)):
        p_in = basemesh.data.vertices[inner_vid].co.copy()
        p_mid = basemesh.data.vertices[mid_vid].co.copy()
        p_out = basemesh.data.vertices[outer_vid].co.copy()
        for _ in range(125):
            s = rng.uniform(0.0, 1.0)
            base_p = ((1 - s) ** 2) * p_in + 2 * (1 - s) * s * p_mid + (s ** 2) * p_out
            root_p = base_p + Vector((rng.uniform(-0.002, 0.002), -0.0015, rng.uniform(-0.0026, 0.0026)))
            blen = rng.uniform(0.007, 0.013)
            bdir = Vector((side_s * rng.uniform(0.55, 0.90), -0.20, rng.uniform(0.16, 0.52))).normalized()
            br = rng.uniform(0.00008, 0.00014)
            brow_sp = rng.uniform(0.45, 0.95)
            pts_b, rads_b, cols_b = [], [], []
            for j in range(5):
                t = j / 4.0
                pos = root_p + bdir * (blen * t) + Vector((0.0, 0.0, -0.002 * t * t))
                pts_b.append((pos.x, pos.y, pos.z))
                rads_b.append(br * (1.0 - 0.78 * t))
                cols_b.append(_pt_color(pos, (0.42, 0.40, 0.38), bias=0.78, strand_mul=brow_sp, gamma_pow=2.2))
            brow_pts.append(pts_b)
            brow_rads.append(rads_b)
            brow_cols.append(cols_b)
    brow_ob = _build_cycles_hair_curves_obj(
        "PortraitEyebrows", brow_pts, brow_rads, mat_brow, strands_colors=brow_cols if use_pcol else None
    )

    out_objs = [hair_ob, must_ob, brow_ob]
    if vol_hair is not None:
        out_objs.insert(0, vol_hair)
    return out_objs


def build_tailored_collar_bust(
    basemesh,
    analysis: Dict,
    fit_cam: Optional[Dict] = None,
    cam_obj=None,
):
    """
    Build a TRUE 3D rounded anatomical tailored coat/shirt bust + folded 3D collar + wide 3D lapels
    + 3D horn button (`zero staircase cuts or flat slabs on +-34° / side turnarounds!`), photo-fitted
    from `torso_coat` width, shoulder slope, & V-neck dimensions, with SUBSURF -> Camera UV Project
    on front and seamless procedural wool weave wrapping 360° around the shoulders and back.
    """
    tcoat = analysis.get("torso_coat")
    cstats = analysis.get("collar_stats", {})
    c_luma = float(cstats.get("luma", 0.28))
    c_rgb = cstats.get("rgb_srgb", [c_luma, c_luma, c_luma])

    gi_body = basemesh.vertex_groups["body"].index if "body" in basemesh.vertex_groups else None
    chin_co = basemesh.data.vertices[5225].co.copy()
    chin_z = float(chin_co.z)
    zyg_l = basemesh.data.vertices[11794].co.copy()
    zyg_r = basemesh.data.vertices[5182].co.copy()
    w_face_m = max(0.11, float((zyg_l - zyg_r).length))

    neck_pts = [
        v.co for v in basemesh.data.vertices
        if (chin_z - 0.088) < v.co.z < (chin_z - 0.012) and abs(v.co.x - chin_co.x) < 0.080
        and (gi_body is None or any(g.group == gi_body and g.weight > 0.5 for g in v.groups))
    ]
    if neck_pts:
        cx = 0.5 * (min(p.x for p in neck_pts) + max(p.x for p in neck_pts))
        cy = 0.5 * (min(p.y for p in neck_pts) + max(p.y for p in neck_pts))
        neck_rx = min(0.068, max(0.052, 0.5 * (max(p.x for p in neck_pts) - min(p.x for p in neck_pts))))
        neck_ry = min(0.072, max(0.054, 0.5 * (max(p.y for p in neck_pts) - min(p.y for p in neck_pts))))
    else:
        cx, cy, neck_rx, neck_ry = float(chin_co.x), float(chin_co.y) + 0.082, 0.058, 0.060
    # Seat collar height directly from photo collar-to-chin distance so neck is NEVER artificially long!
    prof_pre = tcoat.get("shoulder_top_y_rel", []) if tcoat else []
    mid_col_rel = min(prof_pre[len(prof_pre) // 3 : 2 * len(prof_pre) // 3]) if len(prof_pre) >= 16 else 0.12
    neck_z = chin_z + max(-0.012, min(0.004, 0.008 - mid_col_rel * 0.075))

    # Derive universal 3D bust proportions, wide tailored shoulders & front placket/button center from reference
    half_bust_w = 0.34
    bust_drop = 0.285
    v_open_hw = 0.036
    v_drop = 0.068
    sh_tilt_z = 0.0
    sh_drop_L = 0.048
    sh_drop_R = 0.048
    placket_x = cx + 0.012
    btn_x_target = placket_x
    btn_z_target = neck_z - v_drop * 0.85 - 0.010

    def _ray_to_chest_plane(ux: float, vy: float, y_plane: float) -> Optional[Vector]:
        if not (fit_cam and "cam_K" in fit_cam):
            return None
        K_m = fit_cam["cam_K"]
        Rm_m = Matrix(fit_cam["cam_R"])
        tv_m = Vector(fit_cam["cam_t"])
        W_i = float(analysis.get("width", 1024))
        H_i = float(analysis.get("height", 1024))
        Rm_tr = Rm_m.transposed()
        cam_p = -(Rm_tr @ tv_m)
        ray_cv = Vector(((ux * W_i - K_m[0][2]) / K_m[0][0], (vy * H_i - K_m[1][2]) / K_m[1][1], 1.0))
        ray_w = (Rm_tr @ ray_cv).normalized()
        if abs(ray_w.y) < 1e-4:
            return None
        s_hit = (y_plane - cam_p.y) / ray_w.y
        return cam_p + ray_w * s_hit if s_hit > 0 else None

    if tcoat:
        cw_r = float(tcoat.get("coat_width_ratio", 3.4))
        cd_r = float(tcoat.get("coat_drop_ratio", 2.0))
        v_hw_r = float(tcoat.get("v_open_hw_ratio", 0.24))
        v_dr_r = float(tcoat.get("v_drop_ratio", 0.36))
        half_bust_w = min(0.258, max(0.214, 0.50 * cw_r * w_face_m * 0.73))
        bust_drop = min(0.32, max(0.24, cd_r * w_face_m * 1.10))
        v_open_hw = min(0.064, max(0.028, v_hw_r * w_face_m * 0.92))
        v_drop = min(0.090, max(0.046, v_dr_r * w_face_m * 1.05))
        prof = tcoat.get("shoulder_top_y_rel", [])
        if len(prof) >= 16:
            mid_collar_y = min(prof[len(prof) // 3 : 2 * len(prof) // 3])
            r_side_y = sum(prof[1:5]) / 4.0
            l_side_y = sum(prof[-5:-1]) / 4.0
            sh_drop_R = min(0.115, max(0.068, 0.056 + (r_side_y - mid_collar_y) * w_face_m * 0.82))
            sh_drop_L = min(0.115, max(0.068, 0.056 + (l_side_y - mid_collar_y) * w_face_m * 0.82))
            sh_tilt_z = 0.5 * (sh_drop_R - sh_drop_L)

        # Ray-align front placket & coat button strictly to reference photo when detected
        y_chest_approx = cy - neck_ry - 0.072
        v_cx_norm = tcoat.get("v_open_cx_norm")
        if v_cx_norm is not None:
            p_v = _ray_to_chest_plane(float(v_cx_norm), float(analysis["landmarks_norm"][152][1]) + 0.08, y_chest_approx)
            if p_v is not None:
                placket_x = max(cx - 0.040, min(cx + 0.040, float(p_v.x)))
                btn_x_target = placket_x
        b_norm = tcoat.get("button_norm")
        if b_norm and len(b_norm) == 2:
            p_b = _ray_to_chest_plane(float(b_norm[0]), float(b_norm[1]), y_chest_approx)
            if p_b is not None:
                btn_x_target = max(cx - 0.042, min(cx + 0.042, float(p_b.x)))
                btn_z_target = max(neck_z - bust_drop * 0.62, min(neck_z - 0.042, float(p_b.z)))
                placket_x = btn_x_target

    bm = bmesh.new()
    # Parameterized in polar angle where ang=0 is TRUE FRONT (-Y), ang=+-pi is BACK (+Y)
    # Natural sloped anatomical coat shoulders & lean torso taper (zero balloon bloat!)
    rings = [
        # (dz_center, rx_m, ry_m, y_shift, v_dip_scale, slope_scale) — slim tailored knit coat profile (zero barrel bulk!)
        (-0.016, max(0.040, neck_rx - 0.006), max(0.040, neck_ry - 0.006), -0.002, 0.48, 0.00),  # tucked strictly inside lower neck/collar
        (-0.012, neck_rx + 0.005, neck_ry + 0.004, -0.004, 0.78, 0.16),                           # snug collar seat hugging throat cleanly
        (-0.024, half_bust_w * 0.68, 0.078, -0.008, 0.50, 0.54),                                  # sloped upper trapezius/shoulder shelf
        (-0.066, half_bust_w * 0.93, 0.092, -0.010, 0.16, 0.92),                                  # sleek deltoid shoulder cap
        (-0.122, half_bust_w * 0.95, 0.098, -0.012, 0.00, 0.86),                                  # slim tailored chest/armhole
        (-0.182, half_bust_w * 0.92, 0.098, -0.012, 0.00, 0.50),                                  # tapered mid torso
        (-0.240, half_bust_w * 0.87, 0.093, -0.012, 0.00, 0.22),                                  # lower torso taper
        (-bust_drop, half_bust_w * 0.81, 0.088, -0.010, 0.00, 0.00),                              # clean bottom bust tuck
    ]
    n_seg = 64
    ring_loops = []
    for r_idx, (dz, rx, ry, dy, v_sc, sl_sc) in enumerate(rings):
        row = []
        for i in range(n_seg):
            ang = 2.0 * math.pi * i / n_seg
            sa, ca = math.sin(ang), math.cos(ang)  # sa>0 is +X (character left), ca>0 is -Y (front)
            side_sh_dz = (-sh_drop_L if sa > 0 else -sh_drop_R) * abs(sa) * sl_sc + sh_tilt_z * sa * sl_sc
            chest_bulge = 0.003 * (ca ** 2) if (ca > 0 and r_idx >= 3) else 0.0
            back_round = 0.004 * (ca ** 2) if (ca < 0 and r_idx >= 2) else 0.0
            x = cx + sa * rx
            # Crisp overlapping coat placket fold step along center front chest below collar V
            placket_step = 0.0
            if ca > 0.78 and r_idx >= 2:
                d_pl = x - (placket_x + 0.004 * max(0, r_idx - 2))
                placket_step = 0.0062 if d_pl >= 0.0 else max(0.0, 0.0062 * (1.0 + d_pl / 0.008))
            y = cy + dy - ca * ry - chest_bulge + back_round - placket_step
            z = neck_z + dz + side_sh_dz
            if v_sc > 0.0 and ca > 0.58:
                notch = ((ca - 0.58) / 0.42) ** 1.4
                z -= v_drop * notch * v_sc
            if r_idx == 1 and ca > 0.42:
                flare = 0.014 * (ca - 0.42)
                x += math.copysign(flare * 0.45, sa)
                y -= flare * 0.5
            row.append(bm.verts.new((x, y, z)))
        ring_loops.append(row)

    # Explicitly OUTWARD winding: (r+1[i], r+1[ni], r[ni], r[i]) gives -Y outward normal on front chest!
    for r in range(len(ring_loops) - 1):
        for i in range(n_seg):
            ni = (i + 1) % n_seg
            bm.faces.new((ring_loops[r + 1][i], ring_loops[r + 1][ni], ring_loops[r][ni], ring_loops[r][i]))

    # Close BOTH top neck hub (+Z outward inside throat) & bottom hub (-Z outward) so torso is 100% closed manifold!
    top_hub = bm.verts.new((cx, cy - 0.002, neck_z - 0.022))
    first_r = ring_loops[0]
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        bm.faces.new((first_r[i], first_r[ni], top_hub))

    bot_hub = bm.verts.new((cx, cy - 0.020, neck_z - bust_drop - 0.018))
    last_r = ring_loops[-1]
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        bm.faces.new((last_r[ni], last_r[i], bot_hub))

    def _torso_surface_xy(sa_v: float, ca_v: float, z_q: float) -> Tuple[float, float]:
        """Exact radial torso shell (x, y) at angle (sa, ca) & height z_q so collar NEVER intersects chest!"""
        z_nodes = [neck_z + rg[0] for rg in rings]
        if z_q >= z_nodes[0]:
            k0, k1, ft = 0, 0, 0.0
        elif z_q <= z_nodes[-1]:
            k0, k1, ft = len(rings) - 1, len(rings) - 1, 0.0
        else:
            k0 = 0
            for idx_r in range(len(rings) - 1):
                if z_nodes[idx_r] >= z_q >= z_nodes[idx_r + 1]:
                    k0 = idx_r
                    break
            k1 = min(len(rings) - 1, k0 + 1)
            dz_sp = max(1e-5, z_nodes[k0] - z_nodes[k1])
            ft = max(0.0, min(1.0, (z_nodes[k0] - z_q) / dz_sp))
        rx_q = rings[k0][1] * (1.0 - ft) + rings[k1][1] * ft
        ry_q = rings[k0][2] * (1.0 - ft) + rings[k1][2] * ft
        dy_q = rings[k0][3] * (1.0 - ft) + rings[k1][3] * ft
        c_bulge = (0.005 * (ca_v ** 2)) if (ca_v > 0 and k1 >= 3) else 0.0
        b_round = (0.004 * (ca_v ** 2)) if (ca_v < 0 and k1 >= 2) else 0.0
        tx = cx + sa_v * rx_q
        ty = cy + dy_q - ca_v * ry_q - c_bulge + b_round - (0.0062 if ca_v > 0.78 else 0.0)
        return tx, ty

    # Crisp Two-Piece Tailored Coat Collar + Sleek Shoulder-Hugging Lapels Wrapping 360° (`zero neck gaps or chest clip!`)
    n_col_st = 45
    col_rings = []
    for k in range(n_col_st):
        u = -1.0 + 2.0 * (k / (n_col_st - 1))  # -1 (right lapel) -> 0 (back nape) -> +1 (left lapel)
        ang_c = math.pi * (1.0 - u)
        sa = math.sin(ang_c)
        ca = math.cos(ang_c)

        front_fac = max(0.0, (ca + 0.16) / 1.16)
        front_open = max(0.0, (ca - 0.38) / 0.62)
        # Sharp classic triangular coat/shirt collar peak pointing down-and-inward toward Button 1 (`zero flat chest bib!`)
        lapel_wing = math.exp(-(((abs(u) - 0.74) / 0.12) ** 2))
        lapel_tip = max(0.0, (abs(u) - 0.82) / 0.18)
        back_nape = max(0.0, (-ca - 0.25) / 0.75)

        # Embed inner collar fold 3mm inside neck back/sides so ZERO black corner gap or rim EVER appears!
        in_x = cx + sa * (neck_rx - 0.003 + 0.009 * front_open)
        in_x = in_x * (1.0 - 0.86 * lapel_tip) + (placket_x + math.copysign(0.003, u)) * (0.86 * lapel_tip)
        in_y = cy - 0.002 - ca * (neck_ry - 0.003 + 0.016 * front_open + 0.014 * lapel_tip)
        in_z = neck_z + 0.005 * (1.0 - 0.82 * front_fac) - 0.004 * back_nape + 0.42 * sh_tilt_z * sa * (1.0 - 0.4 * front_fac) - v_drop * (0.48 * front_fac + 0.78 * lapel_tip)

        # Sleek folded jacket collar blade: narrow rolled collar on nape/sides tapering diagonally down into pointed collar tips!
        rx_s = (neck_rx + 0.044) * (1.0 - 0.40 * front_fac) + (half_bust_w * 0.34 + 0.022 * lapel_wing) * (0.40 * front_fac)
        ry_s = (neck_ry + 0.032) * (1.0 - 0.42 * front_fac) + (0.090 + 0.014 * lapel_wing) * (0.42 * front_fac)
        w_shrink = 1.0 - 0.86 * (lapel_tip ** 1.05)
        tip_tuck = 1.0 - 0.88 * (lapel_tip ** 1.15)
        out_x = cx + sa * rx_s * w_shrink + (math.copysign(0.024 * lapel_wing * w_shrink, u) if abs(u) > 0.05 else 0.0)
        out_x = out_x * (1.0 - 0.72 * lapel_tip) + (placket_x + math.copysign(0.004, u)) * (0.72 * lapel_tip)
        out_y_raw = cy - 0.010 - ca * (ry_s + 0.006 * front_fac) - (0.008 * (ca ** 2) * tip_tuck if ca > 0 else 0.0)
        out_y = out_y_raw * (1.0 - 0.64 * (lapel_tip ** 1.35)) + min(out_y_raw, in_y - 0.002) * (0.64 * (lapel_tip ** 1.35))
        slope_s = (-sh_drop_L if out_x > cx else -sh_drop_R) * min(1.0, abs(out_x - cx) / max(0.05, half_bust_w)) * 0.38
        # Collar edge slopes sharply diagonally to pointed peak at lapel_wing, then angles cleanly UP to Button 1 V-notch!
        out_z = neck_z - 0.020 - 0.028 * front_fac - 0.046 * lapel_wing + 0.012 * lapel_tip + slope_s + sh_tilt_z * sa * 0.30

        # Radial torso clearance clamp: push p_out slightly OUTSIDE chest shell so lapels never pierce torso!
        tx_s, ty_s = _torso_surface_xy(sa, ca, out_z)
        if ca > 0.05:
            out_y = min(out_y, ty_s - (0.003 + 0.005 * tip_tuck))
        p_in = Vector((in_x, in_y, in_z))
        p_out = Vector((out_x, out_y, out_z))
        p_mid = (p_in + p_out) * 0.5 + Vector((sa * 0.002 * tip_tuck, -ca * 0.0025 * tip_tuck, 0.0028 * (1.0 - 0.60 * front_fac) * tip_tuck))
        thick_v = Vector((-sa * 0.001, -ca * 0.0012, -0.0065)) * tip_tuck  # sleek tailored wool collar thickness!

        p_out_bot = p_out + thick_v * 0.75
        if ca > 0.05:
            p_out_bot.y = min(p_out_bot.y, ty_s - 0.0015)

        pts_sec = [
            p_in + Vector((0, 0, 0.0018 * tip_tuck)),
            p_mid + Vector((0, 0, 0.0024 * tip_tuck)),
            p_out,
            p_out_bot,
            p_mid + thick_v,
            p_in + thick_v * 0.85,
        ]
        col_rings.append([bm.verts.new(p) for p in pts_sec])

    for k in range(len(col_rings) - 1):
        r0, r1 = col_rings[k], col_rings[k + 1]
        for s in range(6):
            ns = (s + 1) % 6
            bm.faces.new((r0[s], r0[ns], r1[ns], r1[s]))
    bm.faces.new(col_rings[0][::-1])
    bm.faces.new(col_rings[-1])

    # Remove disconnected pre-Subsurf button; snap dedicated PhotoCoatButton onto EVALUATED chest surface!
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    cmesh = bpy.data.meshes.new("PhotoTorsoCoatBust")
    bm.to_mesh(cmesh)
    bm.free()

    cob = bpy.data.objects.new("PhotoTorsoCoatBust", cmesh)
    bpy.context.collection.objects.link(cob)
    for p in cob.data.polygons:
        p.use_smooth = True
    if not cob.data.uv_layers:
        cob.data.uv_layers.new(name="UVMap")

    sm = cob.modifiers.new("CoatSmooth", type="SMOOTH")
    sm.factor = 0.09
    sm.iterations = 1
    sub = cob.modifiers.new("CoatSubsurf", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 2

    def _finish_coat_mats(m_coat):
        cob.data.materials.append(m_coat)
        mat_btn = bpy.data.materials.new("DarkHornButtonMat")
        mat_btn.use_nodes = True
        b_bsdf = mat_btn.node_tree.nodes.get("Principled BSDF")
        if b_bsdf:
            b_bsdf.inputs["Base Color"].default_value = (0.016, 0.015, 0.014, 1.0)
            b_bsdf.inputs["Roughness"].default_value = 0.86
            if "Specular IOR Level" in b_bsdf.inputs:
                b_bsdf.inputs["Specular IOR Level"].default_value = 0.06
        # Raycast onto post-Smooth/Subsurf evaluated chest surface so all placket buttons sit flush across front & side!
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        ev_cob = cob.evaluated_get(deps)
        btn_z_top = max(neck_z - bust_drop * 0.44, min(btn_z_target, neck_z - v_drop - 0.014))
        btn_step = max(0.056, bust_drop * 0.22)
        bm_b = bmesh.new()
        for btn_k in range(3):
            bz = btn_z_top - btn_k * btn_step
            if bz < (neck_z - bust_drop + 0.032):
                break
            bx = btn_x_target + 0.0028 * btn_k
            probe = Vector((bx, cy - neck_ry - 0.26, bz))
            ok_b, hit_loc, hit_nrm, _ = ev_cob.closest_point_on_mesh(probe)
            if not ok_b:
                hit_loc = Vector((bx, cy - neck_ry - 0.095, bz))
                hit_nrm = Vector((0.0, -1.0, 0.0))
            else:
                ok_r, r_loc, r_nrm, _ = ev_cob.ray_cast(probe, Vector((0.0, 1.0, 0.0)))
                if ok_r:
                    hit_loc, hit_nrm = r_loc, r_nrm
            rot_q = Vector((0.0, 0.0, 1.0)).rotation_difference(hit_nrm.normalized()).to_matrix().to_4x4()
            b_pos = hit_loc + hit_nrm.normalized() * 0.0008
            bmesh.ops.create_icosphere(
                bm_b,
                subdivisions=3,
                radius=0.0076,
                matrix=Matrix.Translation(b_pos) @ rot_q @ Matrix.Scale(0.24, 4, (0, 0, 1)),
            )
        bmesh.ops.recalc_face_normals(bm_b, faces=bm_b.faces)
        bmesh_d = bpy.data.meshes.new("PhotoCoatButtonsMesh")
        bm_b.to_mesh(bmesh_d)
        bm_b.free()
        bob = bpy.data.objects.new("PhotoCoatButtons", bmesh_d)
        bpy.context.collection.objects.link(bob)
        for p in bob.data.polygons:
            p.use_smooth = True
        bob.data.materials.append(mat_btn)
        return cob

    if tcoat and fit_cam and cam_obj is not None and "cam_K" in fit_cam:
        W_img = float(analysis["width"])
        H_img = float(analysis["height"])
        Rm = Matrix(fit_cam["cam_R"])
        uvp = cob.modifiers.new("CoatUVProject", type="UV_PROJECT")
        uvp.uv_layer = "UVMap"
        uvp.projectors[0].object = cam_obj
        uvp.aspect_x = int(W_img)
        uvp.aspect_y = int(H_img)

        # Dual front camera-projected photo tweed shader + seamless normalized 3D salt-and-pepper wool wrapping 360°
        mat = bpy.data.materials.new("PhotoCoatTweedMat")
        if not mat.use_nodes:
            mat.use_nodes = True
        nt = mat.node_tree
        n, L = nt.nodes, nt.links.new
        n.clear()
        out = n.new("ShaderNodeOutputMaterial")
        bsdf = n.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Sheen Weight"].default_value = 0.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.006
        L(bsdf.outputs["BSDF"], out.inputs["Surface"])

        tc = n.new("ShaderNodeTexCoord")
        im_c = bpy.data.images.load(tcoat["albedo"], check_existing=True)
        tx_c = n.new("ShaderNodeTexImage"); tx_c.image = im_c; tx_c.extension = "EXTEND"
        L(tc.outputs["UV"], tx_c.inputs["Vector"])
        gam_c = n.new("ShaderNodeGamma")
        gam_c.inputs["Gamma"].default_value = 1.00
        L(tx_c.outputs["Color"], gam_c.inputs["Color"])
        bc_c = n.new("ShaderNodeBrightContrast")
        bc_c.inputs["Bright"].default_value = -0.01
        bc_c.inputs["Contrast"].default_value = -0.02
        L(gam_c.outputs["Color"], bc_c.inputs["Color"])

        # Smooth wide UV feather around ALL edges so bottom chest & shoulders NEVER show a hard horizontal line!
        sep_uv = n.new("ShaderNodeSeparateXYZ")
        L(tc.outputs["UV"], sep_uv.inputs["Vector"])
        u_lo = n.new("ShaderNodeMapRange"); u_lo.clamp = True; u_lo.inputs["From Min"].default_value = 0.010; u_lo.inputs["From Max"].default_value = 0.110
        u_hi = n.new("ShaderNodeMapRange"); u_hi.clamp = True; u_hi.inputs["From Min"].default_value = 0.890; u_hi.inputs["From Max"].default_value = 0.990; u_hi.inputs["To Min"].default_value = 1.0; u_hi.inputs["To Max"].default_value = 0.0
        v_lo = n.new("ShaderNodeMapRange"); v_lo.clamp = True; v_lo.inputs["From Min"].default_value = 0.010; v_lo.inputs["From Max"].default_value = 0.165
        v_hi = n.new("ShaderNodeMapRange"); v_hi.clamp = True; v_hi.inputs["From Min"].default_value = 0.880; v_hi.inputs["From Max"].default_value = 0.985; v_hi.inputs["To Min"].default_value = 1.0; v_hi.inputs["To Max"].default_value = 0.0
        L(sep_uv.outputs["X"], u_lo.inputs["Value"]); L(sep_uv.outputs["X"], u_hi.inputs["Value"])
        L(sep_uv.outputs["Y"], v_lo.inputs["Value"]); L(sep_uv.outputs["Y"], v_hi.inputs["Value"])
        uv_m1 = n.new("ShaderNodeMath"); uv_m1.operation = "MULTIPLY"; L(u_lo.outputs["Result"], uv_m1.inputs[0]); L(u_hi.outputs["Result"], uv_m1.inputs[1])
        uv_m2 = n.new("ShaderNodeMath"); uv_m2.operation = "MULTIPLY"; L(v_lo.outputs["Result"], uv_m2.inputs[0]); L(v_hi.outputs["Result"], uv_m2.inputs[1])
        uv_box = n.new("ShaderNodeMath"); uv_box.operation = "MULTIPLY"; L(uv_m1.outputs["Value"], uv_box.inputs[0]); L(uv_m2.outputs["Value"], uv_box.inputs[1])

        geom = n.new("ShaderNodeNewGeometry")
        # Use PURE camera forward axis so grazing upward collar/shoulder surfaces smoothly switch to crisp 3D tweed (no smear!)
        cam_fwd_to_cam = (cam_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
        vdot = n.new("ShaderNodeVectorMath")
        vdot.operation = "DOT_PRODUCT"
        vdot.inputs[1].default_value = (cam_fwd_to_cam.x, cam_fwd_to_cam.y, cam_fwd_to_cam.z)
        L(geom.outputs["Normal"], vdot.inputs[0])
        f_mask = n.new("ShaderNodeMapRange")
        f_mask.clamp = True
        f_mask.inputs["From Min"].default_value = 0.30
        f_mask.inputs["From Max"].default_value = 0.60
        L(vdot.outputs["Value"], f_mask.inputs["Value"])
        pm_1 = n.new("ShaderNodeMath"); pm_1.operation = "MULTIPLY"
        L(f_mask.outputs["Result"], pm_1.inputs[0])
        L(uv_box.outputs["Value"], pm_1.inputs[1])
        proj_mask = n.new("ShaderNodeMath"); proj_mask.operation = "MULTIPLY"
        L(pm_1.outputs["Value"], proj_mask.inputs[0])
        L(tx_c.outputs["Alpha"], proj_mask.inputs[1])

        # Fine Soft Knitted Wool Tweed (`2x finer tight stitches + softer charcoal-grey contrast matching real knit!`)
        ns_warp = n.new("ShaderNodeTexNoise")
        ns_warp.inputs["Scale"].default_value = 340.0
        ns_warp.inputs["Detail"].default_value = 3.0
        L(tc.outputs["Object"], ns_warp.inputs["Vector"])
        warp_vec = n.new("ShaderNodeMix")
        warp_vec.data_type = "VECTOR"
        warp_vec.inputs[0].default_value = 0.005
        L(tc.outputs["Object"], warp_vec.inputs[4])
        L(ns_warp.outputs["Color"], warp_vec.inputs[5])

        wv_warp = n.new("ShaderNodeTexWave")
        wv_warp.wave_type = "BANDS"; wv_warp.bands_direction = "DIAGONAL"
        wv_warp.inputs["Scale"].default_value = 235.0; wv_warp.inputs["Distortion"].default_value = 0.50
        L(warp_vec.outputs[1], wv_warp.inputs["Vector"])

        wv_cross = n.new("ShaderNodeTexWave")
        wv_cross.wave_type = "BANDS"; wv_cross.bands_direction = "X"
        wv_cross.inputs["Scale"].default_value = 235.0; wv_cross.inputs["Distortion"].default_value = 0.42
        L(warp_vec.outputs[1], wv_cross.inputs["Vector"])

        vor_yarn = n.new("ShaderNodeTexVoronoi")
        vor_yarn.inputs["Scale"].default_value = 265.0
        L(warp_vec.outputs[1], vor_yarn.inputs["Vector"])

        ns_slub = n.new("ShaderNodeTexNoise")
        ns_slub.inputs["Scale"].default_value = 360.0
        ns_slub.inputs["Detail"].default_value = 5.0
        ns_slub.inputs["Roughness"].default_value = 0.65
        L(warp_vec.outputs[1], ns_slub.inputs["Vector"])

        w_w1 = n.new("ShaderNodeMath"); w_w1.operation = "MULTIPLY"; w_w1.inputs[1].default_value = 0.34; L(wv_warp.outputs["Fac"], w_w1.inputs[0])
        w_wc = n.new("ShaderNodeMath"); w_wc.operation = "MULTIPLY"; w_wc.inputs[1].default_value = 0.26; L(wv_cross.outputs["Fac"], w_wc.inputs[0])
        w_vr = n.new("ShaderNodeMath"); w_vr.operation = "MULTIPLY"; w_vr.inputs[1].default_value = 0.18; L(vor_yarn.outputs["Distance"], w_vr.inputs[0])
        w_sl = n.new("ShaderNodeMath"); w_sl.operation = "MULTIPLY"; w_sl.inputs[1].default_value = 0.22; L(ns_slub.outputs["Fac"], w_sl.inputs[0])
        s1 = n.new("ShaderNodeMath"); s1.operation = "ADD"; L(w_w1.outputs["Value"], s1.inputs[0]); L(w_wc.outputs["Value"], s1.inputs[1])
        s2 = n.new("ShaderNodeMath"); s2.operation = "ADD"; L(w_vr.outputs["Value"], s2.inputs[0]); L(w_sl.outputs["Value"], s2.inputs[1])
        mix_t = n.new("ShaderNodeMath"); mix_t.operation = "ADD"; L(s1.outputs["Value"], mix_t.inputs[0]); L(s2.outputs["Value"], mix_t.inputs[1])

        lin_mid = max(0.008, (c_luma / 1.055) ** 2.2)
        c_lin_raw = tuple(max(1e-4, (float(c) / 1.055) ** 2.2) for c in c_rgb)
        lum_c_raw = max(1e-4, 0.2126 * c_lin_raw[0] + 0.7152 * c_lin_raw[1] + 0.0722 * c_lin_raw[2])
        c_tint = tuple(max(0.45, min(1.65, c / lum_c_raw)) for c in c_lin_raw)
        c_std_val = float(cstats.get("std", 0.14))
        tweed_fac = min(1.0, max(0.0, (c_std_val - 0.06) / 0.16))
        # Softer muted wool contrast matched directly to photo knit (`zero giant stark white/black pebbles!`)
        c_lo = max(0.004, float(cstats.get("p15_lin", lin_mid * (1.0 - 0.85 * tweed_fac))) * 0.40)
        c_md = max(0.012, min(0.24, float(cstats.get("p50_lin", lin_mid * 0.72)) * 0.62))
        c_hi = min(0.32, max(c_md * 1.55, float(cstats.get("p85_lin", lin_mid * (1.10 + 1.35 * tweed_fac))) * 0.70))

        cr_back = n.new("ShaderNodeValToRGB")
        cr_back.color_ramp.elements[0].position = 0.24
        cr_back.color_ramp.elements[0].color = (c_lo * c_tint[0], c_lo * c_tint[1], c_lo * c_tint[2], 1.0)
        cr_back.color_ramp.elements[1].position = 0.78
        cr_back.color_ramp.elements[1].color = (min(0.42, c_hi * c_tint[0]), min(0.42, c_hi * c_tint[1]), min(0.42, c_hi * c_tint[2]), 1.0)
        el_m = cr_back.color_ramp.elements.new(0.51)
        el_m.color = (c_md * c_tint[0], c_md * c_tint[1], c_md * c_tint[2], 1.0)
        L(mix_t.outputs["Value"], cr_back.inputs["Fac"])

        mix_cb = n.new("ShaderNodeMix")
        mix_cb.data_type = "RGBA"
        L(proj_mask.outputs["Value"], mix_cb.inputs[0])
        L(cr_back.outputs["Color"], mix_cb.inputs[6])
        L(bc_c.outputs["Color"], mix_cb.inputs[7])
        L(mix_cb.outputs[2], bsdf.inputs["Base Color"])

        im_r = bpy.data.images.load(tcoat["rough"], check_existing=True)
        im_r.colorspace_settings.name = "Non-Color"
        tx_r = n.new("ShaderNodeTexImage"); tx_r.image = im_r; tx_r.extension = "EXTEND"
        L(tc.outputs["UV"], tx_r.inputs["Vector"])
        r_clamp = n.new("ShaderNodeMapRange")
        r_clamp.clamp = True
        r_clamp.inputs["From Min"].default_value = 0.0
        r_clamp.inputs["From Max"].default_value = 1.0
        r_clamp.inputs["To Min"].default_value = 0.88
        r_clamp.inputs["To Max"].default_value = 0.98
        L(tx_r.outputs["Color"], r_clamp.inputs["Value"])
        mix_rf = n.new("ShaderNodeMix")
        mix_rf.data_type = "FLOAT"
        L(proj_mask.outputs["Value"], mix_rf.inputs[0])
        mix_rf.inputs[2].default_value = 0.96
        L(r_clamp.outputs["Result"], mix_rf.inputs[3])
        L(mix_rf.outputs[0], bsdf.inputs["Roughness"])

        im_n = bpy.data.images.load(tcoat["normal"], check_existing=True)
        im_n.colorspace_settings.name = "Non-Color"
        tx_n = n.new("ShaderNodeTexImage"); tx_n.image = im_n; tx_n.extension = "EXTEND"
        nmap = n.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = 0.34
        L(tc.outputs["UV"], tx_n.inputs["Vector"])
        L(tx_n.outputs["Color"], nmap.inputs["Color"])

        bmp_back = n.new("ShaderNodeBump")
        bmp_back.inputs["Strength"].default_value = 0.26
        bmp_back.inputs["Distance"].default_value = 0.0010
        L(mix_t.outputs["Value"], bmp_back.inputs["Height"])

        mix_nm = n.new("ShaderNodeMix")
        mix_nm.data_type = "VECTOR"
        L(proj_mask.outputs["Value"], mix_nm.inputs[0])
        L(bmp_back.outputs["Normal"], mix_nm.inputs[4])
        L(nmap.outputs["Normal"], mix_nm.inputs[5])
        L(mix_nm.outputs[1], bsdf.inputs["Normal"])

        return _finish_coat_mats(mat)

    # Procedural Herringbone Wool fallback when no coat texture/camera projection is provided
    mat = bpy.data.materials.new("HerringboneTweedMat")
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.90
    bsdf.inputs["Sheen Weight"].default_value = 0.16
    bsdf.inputs["Sheen Roughness"].default_value = 0.65
    bsdf.inputs["Sheen Tint"].default_value = (0.18, 0.16, 0.14, 1.0)
    bsdf.inputs["Specular IOR Level"].default_value = 0.14
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = n.new("ShaderNodeTexCoord")
    vor_f = n.new("ShaderNodeTexVoronoi")
    vor_f.inputs["Scale"].default_value = 280.0
    L(tc.outputs["Object"], vor_f.inputs["Vector"])

    wv = n.new("ShaderNodeTexWave")
    wv.wave_type = "BANDS"
    wv.bands_direction = "DIAGONAL"
    wv.inputs["Scale"].default_value = 180.0
    wv.inputs["Distortion"].default_value = 1.2
    L(tc.outputs["Object"], wv.inputs["Vector"])

    w_wv = n.new("ShaderNodeMath"); w_wv.operation = "MULTIPLY"; w_wv.inputs[1].default_value = 0.45; L(wv.outputs["Fac"], w_wv.inputs[0])
    w_vr = n.new("ShaderNodeMath"); w_vr.operation = "MULTIPLY"; w_vr.inputs[1].default_value = 0.55; L(vor_f.outputs["Distance"], w_vr.inputs[0])
    mix_t = n.new("ShaderNodeMath"); mix_t.operation = "ADD"; L(w_wv.outputs["Value"], mix_t.inputs[0]); L(w_vr.outputs["Value"], mix_t.inputs[1])

    cr = n.new("ShaderNodeValToRGB")
    lo_c = 0.007 if c_luma < 0.45 else 0.020
    hi_c = 0.036 if c_luma < 0.45 else 0.095
    cr.color_ramp.elements[0].position = 0.22; cr.color_ramp.elements[0].color = (lo_c, lo_c * 0.94, lo_c * 0.88, 1.0)
    cr.color_ramp.elements[1].position = 0.85; cr.color_ramp.elements[1].color = (hi_c, hi_c * 0.95, hi_c * 0.88, 1.0)
    L(mix_t.outputs["Value"], cr.inputs["Fac"])
    L(cr.outputs["Color"], bsdf.inputs["Base Color"])

    bmp = n.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.55
    bmp.inputs["Distance"].default_value = 0.0018
    L(mix_t.outputs["Value"], bmp.inputs["Height"])
    L(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    return _finish_coat_mats(mat)


def _create_matched_camera(cam_matrix_4x4: List[List[float]], focal_mm: float, name: str = "MatchedRefCam"):
    cd = bpy.data.cameras.new(name)
    cd.lens = focal_mm
    cd.sensor_width = 36.0
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start = 0.01
    cd.clip_end = 50.0
    co = bpy.data.objects.new(name, cd)
    bpy.context.collection.objects.link(co)
    co.matrix_world = Matrix(cam_matrix_4x4)
    return co


def _render_portrait_qa_set(
    out_dir: str,
    matched_cam,
    center_pt: Tuple[float, float, float],
    focal_mm: float,
    res_x: int,
    res_y: int,
    render_samples: int = 72,
) -> Dict[str, str]:
    """Render matched camera angle, silhouette, clay relief, and -35 / +35 turnaround views."""
    sc = bpy.context.scene
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.cycles.samples = render_samples

    paths = {}
    # 1. Matched Reference Camera Render
    sc.camera = matched_cam
    p_matched = os.path.join(out_dir, "render_matched.png")
    sc.render.filepath = p_matched
    bpy.ops.render.render(write_still=True)
    paths["matched"] = p_matched

    # 2. Orbit Turnarounds (-34 deg, +34 deg, +82 deg true profile) around head center
    tgt = Vector(center_pt)
    cam_pos = matched_cam.matrix_world.translation.copy()
    for tag, deg in (("turn_L", -34.0), ("turn_R", 34.0), ("turn_side", 82.0)):
        rot = Matrix.Rotation(math.radians(deg), 3, "Z")
        new_pos = tgt + rot @ (cam_pos - tgt)
        oc = bpy.data.cameras.new(f"Cam_{tag}")
        oc.lens = focal_mm
        oc.sensor_width = 36.0
        oo = bpy.data.objects.new(f"Cam_{tag}", oc)
        bpy.context.collection.objects.link(oo)
        oo.location = new_pos
        oo.rotation_euler = (tgt - new_pos).to_track_quat("-Z", "Y").to_euler()
        sc.camera = oo
        p_t = os.path.join(out_dir, f"render_{tag}.png")
        sc.render.filepath = p_t
        bpy.ops.render.render(write_still=True)
        paths[tag] = p_t

    # 3. Clay Relief Diagnostic Render (inspect sculpted geometry, wrinkles & hair silhouette cleanly)
    clay_mat = bpy.data.materials.new("DiagClayMat")
    if not clay_mat.use_nodes:
        clay_mat.use_nodes = True
    cbs = clay_mat.node_tree.nodes.get("Principled BSDF")
    cbs.inputs["Base Color"].default_value = (0.58, 0.55, 0.52, 1.0)
    cbs.inputs["Roughness"].default_value = 0.42
    sc.view_layers[0].material_override = clay_mat
    sc.camera = matched_cam
    p_clay = os.path.join(out_dir, "render_clay.png")
    sc.render.filepath = p_clay
    bpy.ops.render.render(write_still=True)
    sc.view_layers[0].material_override = None
    paths["clay"] = p_clay
    return paths


def build_exact_portrait_from_ref(
    ref_image_path: str,
    out_dir: str,
    gender: float = 1.0,
    age: float = 0.90,
    race: Optional[Dict[str, float]] = None,
    target_overrides: Optional[Dict[str, float]] = None,
    colorize_monochrome: bool = False,
    render_samples: int = 84,
    samples_preview_res: int = 800,
) -> Dict:
    """
    End-to-End Universal Human Portrait Reconstruction in Blender 5.2 LTS:
      1. Run ONNX + MediaPipe portrait analyzer -> landmarks, blendshapes, de-lit skin/coat/hair grids
      2. Spawn MPFB2 quad basemesh + anatomical age/gender/eye-bag targets
      3. Run Pinhole `solvePnP` focal sweep + anchored `3D RBF Thin-Plate Spline` + 2048x2048 UV texture bake
      4. Refit & aim spherical `high-poly` eyeballs, apply skin PBR + 3D `DISPLACE` facial wrinkles
      5. Build photo-silhouette curly hair, mustache, eyebrows, & photo-sculpted 3D coat bust
      6. Render matched view, turnarounds, & clay relief
    """
    os.makedirs(out_dir, exist_ok=True)
    analysis_json = os.path.join(out_dir, "portrait_analysis.json")
    cli_args = ["analyze-portrait", "--img", ref_image_path, "--out-dir", out_dir]
    if colorize_monochrome:
        cli_args.append("--colorize")
    run_vision_cli(cli_args)
    with open(analysis_json, "r", encoding="utf-8") as f:
        ana = json.load(f)

    reset_clean_scene(engine="CYCLES", samples=render_samples)
    ensure_mpfb_enabled()

    from bl_ext.user_default.mpfb.services.humanservice import HumanService
    from bl_ext.user_default.mpfb.services.targetservice import TargetService

    macro = TargetService.get_default_macro_info_dict()
    macro["gender"] = float(gender)
    macro["age"] = float(age)
    macro["weight"] = 0.56
    macro["muscle"] = 0.45
    macro["race"] = race or {"caucasian": 1.0, "asian": 0.0, "african": 0.0}
    basemesh = HumanService.create_human(mask_helpers=True, macro_detail_dict=macro)

    # Universal anatomical skin-age & loose-skin micro-targets driven automatically by photo texture energy + age
    bs = ana.get("blendshapes", {})
    skin_tex_pre = float(ana.get("color_metrics", {}).get("skin_texture_energy", 0.45))
    age_drive = max(float(age), min(1.0, 0.25 + 0.85 * skin_tex_pre))
    default_t = {}
    if age_drive > 0.55:
        ag_s = max(0.0, min(1.0, (age_drive - 0.50) / 0.45))
        default_t.update({
            "head-age-incr": 0.35 + 0.30 * ag_s,
            "l-eye-bag-incr": 0.36 + 0.34 * ag_s,
            "r-eye-bag-incr": 0.36 + 0.34 * ag_s,
            "l-eye-bag-height-incr": 0.25 + 0.22 * ag_s,
            "r-eye-bag-height-incr": 0.25 + 0.22 * ag_s,
            "mouth-laugh-lines-in": 0.34 + 0.32 * ag_s,
            "l-cheek-volume-decr": 0.04 + 0.05 * ag_s,
            "r-cheek-volume-decr": 0.04 + 0.05 * ag_s,
            "eyebrows-angle-down": 0.22 * ag_s,
        })
    if bs.get("eyeSquintLeft", 0) > 0.25 or bs.get("eyeSquintRight", 0) > 0.25:
        default_t["eyebrows-angle-down"] = 0.25
    if target_overrides:
        default_t.update(target_overrides)
    _apply_safe_mpfb_targets(basemesh, default_t)

    mesh_in_p = os.path.join(out_dir, "mpfb_rest.json")
    mesh_out_p = os.path.join(out_dir, "mpfb_warped.json")
    with open(mesh_in_p, "w", encoding="utf-8") as f:
        json.dump(_bake_and_export_mpfb_mesh(basemesh), f)

    run_vision_cli([
        "fit-and-bake-face",
        "--analysis", analysis_json,
        "--mesh-in", mesh_in_p,
        "--mesh-out", mesh_out_p,
        "--out-dir", out_dir,
    ])
    with open(mesh_out_p, "r", encoding="utf-8") as f:
        fit = json.load(f)

    # Apply 3D RBF warped vertices onto basemesh
    warped_v = fit["warped_verts"]
    for i, co in enumerate(warped_v):
        basemesh.data.vertices[i].co = Vector(co)
    basemesh.data.update()

    # Attach spherical high-poly eyes fitted cleanly to deformed sockets BEFORE stripping helpers!
    iris_rgb = tuple(ana.get("iris_rgb_srgb", [0.06, 0.04, 0.03]))
    gaze_yaw = float(ana.get("gaze_yaw_ratio", 0.0))
    eyes = _attach_and_aim_eyes(basemesh, iris_rgb, gaze_yaw)

    # Apply calibrated skin PBR + anatomical zone micro-detail + high-density 3D Displace facial carving
    uv_maps = fit["uv_maps"]
    cmet = ana.get("color_metrics", {})
    is_mono = bool(cmet.get("is_monochrome", False))
    skin_tex_e = float(cmet.get("skin_texture_energy", 0.55))
    skin_mat = _build_skin_pbr_material(
        "PortraitSkinMat",
        uv_maps["uv_basecolor"],
        uv_maps["uv_height16"],
        uv_maps["uv_rough8"],
        is_monochrome=is_mono,
        colorize_monochrome=colorize_monochrome,
        skin_texture_energy=skin_tex_e,
        uv_skin_zones_path=uv_maps.get("uv_skin_zones"),
    )
    basemesh.data.materials.clear()
    basemesh.data.materials.append(skin_mat)

    cam = _create_matched_camera(fit["camera_matrix_4x4"], fit["focal_mm"])

    # Build hair, mustache, eyebrows, coat bust & camera aim while canonical vertex IDs are intact!
    hair_objs = build_curly_hair_and_facial_hair(basemesh, ana, fit_cam=fit, cam_obj=cam)
    chin_co = basemesh.data.vertices[5225].co.copy()
    collar_obj = build_tailored_collar_bust(basemesh, ana, fit_cam=fit, cam_obj=cam)
    head_c = Vector(basemesh.data.vertices[294].co) * 0.4 + Vector(basemesh.data.vertices[133].co) * 0.6

    # Keep generous cylindrical neck intact inside tucked Ring 0 collar seat (`zero black ragged seam, zero shoulder horns!`)
    _strip_helper_verts_and_lower_body(
        basemesh, collar_cut_z=float(chin_co.z) - 0.076, neck_cx=float(chin_co.x), neck_trim_half_w=0.068
    )

    # Subdivide (render_levels=3) + Physically Carve Dual-Band 16-bit Forehead Wrinkles, Eye Bags & Nasolabial Folds!
    sub = basemesh.modifiers.new("HeadSmoothSubsurf", type="SUBSURF")
    sub.levels = 2
    sub.render_levels = 3

    disp_tex = bpy.data.textures.new("FacialWrinkleDispTex", type="IMAGE")
    im_disp = bpy.data.images.load(uv_maps["uv_height16"], check_existing=True)
    im_disp.colorspace_settings.name = "Non-Color"
    disp_tex.image = im_disp
    disp = basemesh.modifiers.new("FacialMicroDisplace", type="DISPLACE")
    disp.texture = disp_tex
    disp.texture_coords = "UV"
    disp.uv_layer = "UVMap"
    disp.mid_level = 0.5
    # Balanced fleshy 3D wrinkle displacement (`2.0..2.8 mm`, soft natural folds across front, side & clay views!)
    disp.strength = 0.0020 + 0.0008 * skin_tex_e

    # Universal Photo-Driven Soft Studio Lighting: matches Key side & soft Rembrandt shading without black eye pits!
    p_key_x = float(cmet.get("key_side_x", 0.65)) * 0.78
    p_key_z = float(cmet.get("key_elev_z", 0.72))
    p_cr = float(cmet.get("photo_contrast_ratio", 4.5))
    cr_t = max(0.0, min(1.0, (p_cr - 2.2) / 5.5))
    k_size = 1.24 - 0.24 * cr_t
    k_eng = 168.0
    f_eng = max(30.0, min(48.0, 44.0 - 14.0 * cr_t))
    setup_studio_lighting(
        target_pt=(head_c.x, head_c.y + 0.04, head_c.z - 0.02),
        bg_luma=0.015,
        key_energy=k_eng,
        fill_energy=f_eng,
        rim_energy=158.0,
        key_side_x=p_key_x,
        key_elev_z=p_key_z,
        key_size=k_size,
        exposure=0.0,
    )

    # Preserve input aspect ratio cleanly on rendered preview
    W0, H0 = int(ana["width"]), int(ana["height"])
    rw = samples_preview_res
    rh = int(round(samples_preview_res * H0 / max(1, W0)))
    render_paths = _render_portrait_qa_set(
        out_dir, cam, (0.0, -0.05, 1.50), fit["focal_mm"], rw, rh, render_samples=render_samples
    )

    return {
        "analysis": ana,
        "focal_mm": fit["focal_mm"],
        "reproj_err_px": fit["reproj_err_px"],
        "uv_maps": uv_maps,
        "renders": render_paths,
        "objects": [basemesh.name, eyes.name, *[o.name for o in hair_objs], collar_obj.name],
    }


# ---------------------------------------------------------------------------
# Universal Non-Human Arbitrary Object Reconstruction (`lathe`, `bilateral_relief`, `box_subsurf`)
# ---------------------------------------------------------------------------
def _build_lathe_object_mesh(profile_rz: List[List[float]], target_height: float = 1.0, segs: int = 48):
    """Build clean manifold quad lathe object from 2D silhouette radius/height profile."""
    bm = bmesh.new()
    rings = []
    min_h = min(p[1] for p in profile_rz)
    max_h = max(p[1] for p in profile_rz)
    span_h = max(1e-4, max_h - min_h)
    scale_h = target_height / span_h

    for r_norm, z_norm in profile_rz:
        r = max(0.004, r_norm * scale_h)
        z = (z_norm - min_h) * scale_h
        row = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs - 0.5 * math.pi
            row.append(bm.verts.new((r * math.sin(ang), -r * math.cos(ang), z)))
        rings.append(row)

    for k in range(len(rings) - 1):
        for s in range(segs):
            ns = (s + 1) % segs
            bm.faces.new((rings[k][s], rings[k][ns], rings[k + 1][ns], rings[k + 1][s]))

    # Cap top & bottom cleanly with pole vertex
    bot_c = bm.verts.new((0.0, 0.0, (profile_rz[0][1] - min_h) * scale_h))
    top_c = bm.verts.new((0.0, 0.0, (profile_rz[-1][1] - min_h) * scale_h))
    for s in range(segs):
        ns = (s + 1) % segs
        bm.faces.new((bot_c, rings[0][ns], rings[0][s]))
        bm.faces.new((top_c, rings[-1][s], rings[-1][ns]))

    mesh = bpy.data.meshes.new("LatheObjectMesh")
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new("ReconstructedObject", mesh)
    bpy.context.collection.objects.link(ob)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def _build_bilateral_relief_mesh(
    grid_depth: List[List[float]],
    grid_matte: List[List[float]],
    aspect_wh: float,
    target_height: float = 1.0,
    thickness: float = 0.34,
):
    """
    Build watertight manifold double-sided quad relief from Depth-Anything-V2 + RMBG-1.4 grids.
    Every interior mask quad gets matching front (+depth) & back (-rounded depth) faces and
    clean boundary bridge quads so turnaround views show realistic 3D solidity!
    """
    N = len(grid_depth)
    w_m = target_height * aspect_wh
    bm = bmesh.new()

    active = [[grid_matte[r][c] > 0.45 for c in range(N - 1)] for r in range(N - 1)]
    # Mark active grid vertices needed by active cells
    v_used = [[False] * N for _ in range(N)]
    for r in range(N - 1):
        for c in range(N - 1):
            if active[r][c]:
                v_used[r][c] = v_used[r + 1][c] = v_used[r][c + 1] = v_used[r + 1][c + 1] = True

    f_verts = [[None] * N for _ in range(N)]
    b_verts = [[None] * N for _ in range(N)]

    for r in range(N):
        for c in range(N):
            if not v_used[r][c]:
                continue
            u = c / (N - 1)
            v = 1.0 - r / (N - 1)
            x = (u - 0.5) * w_m
            z = v * target_height
            # Soft rounded sqrt edge taper so bilateral relief rim forms smooth curved shoulder!
            edge_w = math.sqrt(min(1.0, max(0.02, (grid_matte[r][c] - 0.35) / 0.55)))
            d_val = float(grid_depth[r][c])
            y_front = -(0.04 + d_val * 0.55 * thickness) * edge_w
            y_back = +(0.04 + (0.35 + 0.45 * d_val) * 0.48 * thickness) * edge_w
            f_verts[r][c] = bm.verts.new((x, y_front, z))
            b_verts[r][c] = bm.verts.new((x, y_back, z))

    for r in range(N - 1):
        for c in range(N - 1):
            if not active[r][c]:
                continue
            bm.faces.new((f_verts[r + 1][c], f_verts[r + 1][c + 1], f_verts[r][c + 1], f_verts[r][c]))
            bm.faces.new((b_verts[r][c], b_verts[r][c + 1], b_verts[r + 1][c + 1], b_verts[r + 1][c]))

    # Close silhouette boundary perimeter between front and back shells cleanly
    def _is_act(rr, cc):
        return 0 <= rr < N - 1 and 0 <= cc < N - 1 and active[rr][cc]

    for r in range(N - 1):
        for c in range(N - 1):
            if not active[r][c]:
                continue
            if not _is_act(r - 1, c):
                bm.faces.new((f_verts[r][c], f_verts[r][c + 1], b_verts[r][c + 1], b_verts[r][c]))
            if not _is_act(r + 1, c):
                bm.faces.new((f_verts[r + 1][c + 1], f_verts[r + 1][c], b_verts[r + 1][c], b_verts[r + 1][c + 1]))
            if not _is_act(r, c - 1):
                bm.faces.new((f_verts[r + 1][c], f_verts[r][c], b_verts[r][c], b_verts[r + 1][c]))
            if not _is_act(r, c + 1):
                bm.faces.new((f_verts[r][c + 1], f_verts[r + 1][c + 1], b_verts[r + 1][c + 1], b_verts[r][c + 1]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("BilateralReliefMesh")
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new("ReconstructedObject", mesh)
    bpy.context.collection.objects.link(ob)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def _build_object_pbr_shader(name: str, maps: Dict[str, str], is_metallic: float = 0.0):
    """Build PBR material using projected de-lit albedo + tangent normal + roughness maps."""
    mat = bpy.data.materials.new(name)
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = is_metallic
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = n.new("ShaderNodeTexCoord")

    im_col = bpy.data.images.load(maps["albedo"], check_existing=True)
    tx_col = n.new("ShaderNodeTexImage"); tx_col.image = im_col; tx_col.extension = "EXTEND"
    L(tc.outputs["UV"], tx_col.inputs["Vector"])
    L(tx_col.outputs["Color"], bsdf.inputs["Base Color"])

    im_r = bpy.data.images.load(maps["rough"], check_existing=True)
    im_r.colorspace_settings.name = "Non-Color"
    tx_r = n.new("ShaderNodeTexImage"); tx_r.image = im_r; tx_r.extension = "EXTEND"
    L(tc.outputs["UV"], tx_r.inputs["Vector"])
    L(tx_r.outputs["Color"], bsdf.inputs["Roughness"])

    im_n = bpy.data.images.load(maps["normal"], check_existing=True)
    im_n.colorspace_settings.name = "Non-Color"
    tx_n = n.new("ShaderNodeTexImage"); tx_n.image = im_n; tx_n.extension = "EXTEND"
    nmap = n.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = 0.65
    L(tc.outputs["UV"], tx_n.inputs["Vector"])
    L(tx_n.outputs["Color"], nmap.inputs["Color"])
    L(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_exact_object_from_ref(
    ref_image_path: str,
    out_dir: str,
    topology_mode: str = "auto",
    depth_thickness: float = 0.36,
    is_metallic: float = 0.0,
    render_samples: int = 72,
    preview_res: int = 800,
) -> Dict:
    """
    Universal Arbitrary Object Image-to-3D Reconstructor in Blender 5.2 LTS.
    Supports `"auto"`, `"lathe"`, and `"bilateral_relief"` with camera-matched UV Project
    stacked cleanly ABOVE Subdivision Surface + synthesized PBR Normal/Roughness maps.
    """
    os.makedirs(out_dir, exist_ok=True)
    run_vision_cli(["analyze-object", "--img", ref_image_path, "--out-dir", out_dir, "--mode", topology_mode])
    with open(os.path.join(out_dir, "object_analysis.json"), "r", encoding="utf-8") as f:
        ana = json.load(f)

    reset_clean_scene(engine="CYCLES", samples=render_samples)
    mode = ana["chosen_mode"]
    if mode == "lathe":
        ob = _build_lathe_object_mesh(ana["lathe_profile"], target_height=1.0, segs=52)
    else:
        ob = _build_bilateral_relief_mesh(
            ana["grid_depth"], ana["grid_matte"], float(ana["aspect_wh"]), target_height=1.0, thickness=depth_thickness
        )

    if not ob.data.uv_layers:
        ob.data.uv_layers.new(name="UVMap")

    # Compute exact matched front camera framing object bounding height + center offset
    x0, y0, x1, y1 = ana["bbox"]
    W, H = float(ana["width"]), float(ana["height"])
    bh_frac = max(0.15, (y1 - y0) / H)
    lens_mm = 70.0
    # Match vertical height in normalized camera coordinates so bbox fills exact pixel fraction!
    sensor_h = 36.0 * H / W
    dist = 1.0 * (lens_mm / sensor_h) / bh_frac
    cx_shift = (ana["cx_norm"] - 0.5) * (1.0 * W / H) * (dist * sensor_h / lens_mm)
    cy_shift = (0.5 - ana["cy_norm"]) * (dist * sensor_h / lens_mm)

    cd = bpy.data.cameras.new("ObjectMatchCam")
    cd.lens = lens_mm
    cd.sensor_width = 36.0
    cd.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("ObjectMatchCam", cd)
    bpy.context.collection.objects.link(cam)
    cam.location = (-cx_shift, -dist, 0.5 - cy_shift)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    # CRITICAL STACK ORDER: SMOOTH -> SUBSURF FIRST -> UV_PROJECT SECOND eliminates stair-step teeth & UV distortion!
    if mode == "bilateral_relief":
        sm = ob.modifiers.new("ContourSmooth", type="SMOOTH")
        sm.factor = 0.55
        sm.iterations = 5
    sub = ob.modifiers.new("SmoothSubsurf", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    uvp = ob.modifiers.new("CameraUVProject", type="UV_PROJECT")
    uvp.uv_layer = "UVMap"
    uvp.projectors[0].object = cam
    uvp.aspect_x = int(W)
    uvp.aspect_y = int(H)

    ob.data.materials.append(_build_object_pbr_shader("ObjectRefMat", ana["maps"], is_metallic=is_metallic))
    setup_studio_lighting(target_pt=(0.0, 0.0, 0.50), bg_luma=0.04, key_energy=280.0, fill_energy=130.0, rim_energy=220.0)

    sc = bpy.context.scene
    sc.render.resolution_x = preview_res
    sc.render.resolution_y = int(round(preview_res * H / W))

    renders = {}
    sc.camera = cam
    p_m = os.path.join(out_dir, "obj_matched.png")
    sc.render.filepath = p_m
    bpy.ops.render.render(write_still=True)
    renders["matched"] = p_m

    tgt = Vector((0.0, 0.0, 0.50))
    for tag, deg in (("turn_L", -36.0), ("turn_R", 36.0)):
        rot = Matrix.Rotation(math.radians(deg), 3, "Z")
        npos = tgt + rot @ (cam.location - tgt)
        tc = bpy.data.cameras.new(f"ObjCam_{tag}"); tc.lens = lens_mm
        to = bpy.data.objects.new(f"ObjCam_{tag}", tc)
        bpy.context.collection.objects.link(to)
        to.location = npos
        to.rotation_euler = (tgt - npos).to_track_quat("-Z", "Y").to_euler()
        sc.camera = to
        p_t = os.path.join(out_dir, f"obj_{tag}.png")
        sc.render.filepath = p_t
        bpy.ops.render.render(write_still=True)
        renders[tag] = p_t

    clay = bpy.data.materials.new("ObjClay")
    if not clay.use_nodes:
        clay.use_nodes = True
    clay.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.58, 0.56, 0.54, 1.0)
    sc.view_layers[0].material_override = clay
    sc.camera = cam
    p_c = os.path.join(out_dir, "obj_clay.png")
    sc.render.filepath = p_c
    bpy.ops.render.render(write_still=True)
    sc.view_layers[0].material_override = None
    renders["clay"] = p_c

    return {"analysis": ana, "renders": renders, "object": ob.name, "mode": mode}


# ---------------------------------------------------------------------------
# Standalone Cleanup & Retopo Bake Helper for Raw AI `.glb/.obj` Meshes
# ---------------------------------------------------------------------------
def clean_and_bake_raw_mesh(
    src_obj_or_path,
    target_faces: int = 12000,
    voxel_size: float = 0.008,
    bake_size: int = 2048,
    out_dir: Optional[str] = None,
):
    """
    Production cleanup pipeline for high-VRAM AI exports (Hunyuan3D-2 / TRELLIS / TripoSR):
      Merge by Distance -> Duplicate Clean Proxy -> Voxel Remesh -> QuadriFlow Retopo ->
      SLIM UV Unwrap -> Selected-to-Active Cycles Albedo + Normal Bake.
    """
    if isinstance(src_obj_or_path, str):
        ext = os.path.splitext(src_obj_or_path)[1].lower()
        before = set(bpy.data.objects)
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=src_obj_or_path)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=src_obj_or_path)
        added = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
        if not added:
            raise RuntimeError(f"No mesh imported from {src_obj_or_path}")
        hi_obj = added[0]
    else:
        hi_obj = src_obj_or_path

    bpy.context.view_layer.objects.active = hi_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0005)
    bpy.ops.object.mode_set(mode="OBJECT")

    lo_obj = hi_obj.copy()
    lo_obj.data = hi_obj.data.copy()
    lo_obj.name = f"{hi_obj.name}_Retopo"
    bpy.context.collection.objects.link(lo_obj)
    bpy.ops.object.select_all(action="DESELECT")
    lo_obj.select_set(True)
    bpy.context.view_layer.objects.active = lo_obj

    lo_obj.data.remesh_voxel_size = voxel_size
    bpy.ops.object.voxel_remesh()
    bpy.ops.object.quadriflow_remesh(target_faces=target_faces, use_mesh_symmetry=False, preserve_sharp=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Bake Base Color & Tangent Normal from hi_obj -> lo_obj cleanly in Cycles
        b_mat = bpy.data.materials.new(f"{lo_obj.name}_BakedMat")
        if not b_mat.use_nodes:
            b_mat.use_nodes = True
        nt = b_mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        im_diff = bpy.data.images.new("Bake_Albedo", width=bake_size, height=bake_size)
        tx_diff = nt.nodes.new("ShaderNodeTexImage"); tx_diff.image = im_diff
        lo_obj.data.materials.clear()
        lo_obj.data.materials.append(b_mat)

        bpy.ops.object.select_all(action="DESELECT")
        hi_obj.select_set(True)
        lo_obj.select_set(True)
        bpy.context.view_layer.objects.active = lo_obj
        nt.nodes.active = tx_diff

        bpy.context.scene.render.bake.use_selected_to_active = True
        bpy.context.scene.render.bake.cage_extrusion = 0.04
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
        im_diff.filepath_raw = os.path.join(out_dir, f"{lo_obj.name}_albedo.png")
        im_diff.file_format = "PNG"
        im_diff.save()
        nt.links.new(tx_diff.outputs["Color"], bsdf.inputs["Base Color"])

    return lo_obj
