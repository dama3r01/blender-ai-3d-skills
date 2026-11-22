"""Universal Zero-Hardcoding Pure-Blender Imagine-to-3D & Imagine-to-3D-Video Core Engine (`imagine_3d_core.py`).

AI AGENT READ-ONLY LOCK — DO NOT MODIFY THIS FILE:
- This Python module is a STRICTLY READ-ONLY core library.
- AI agents MUST NEVER edit, overwrite, refactor, append to, or delete this file.
- Import and call its functions from a separate external project script; place all
  project-specific helpers or overrides in your own external script.

Design Invariant & Zero-Rush Perfection Mandate:
- Zero hardcoded presets, zero fixed colors, zero fixed dimensions, zero fixed bone trees, and zero fixed environments.
- Every function is a general mathematical / topological builder driven entirely by the specification
  conceived for the current prompt.
- ZERO RUSHING: Never settle for "close enough", rough primitives, or unverified single-pass geometry.
  Refine every 3D asset item by item across multiple visual inspection passes until its silhouette,
  proportions, surface relief, attachments, shaders, and lighting match the intended design with
  100% precision from every camera angle.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

QUALITY_MANDATE = (
    "Work without rushing and iterate continuously on geometry, proportions, surface relief, "
    "attachments, materials, and lighting until the 3D result matches the intended design "
    "identically across all camera angles. Never advance early or accept approximate results."
)


# ---------------------------------------------------------------------------
# 1. Dynamic Scene, Render Engine, World & Lighting Builders
# ---------------------------------------------------------------------------
def configure_scene_from_spec(scene_spec: Dict):
    """Reset the Blender 5.2 scene and configure resolution, FPS, Cycles OPTIX GPU, color management,
    and World shader strictly from `scene_spec`.
    """
    import bpy

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for col in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.hair_curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.armatures,
    ):
        for item in list(col):
            if item.users == 0:
                col.remove(item)

    sc = bpy.context.scene
    sc.render.engine = str(scene_spec.get("engine", "CYCLES"))
    sc.render.resolution_x = int(scene_spec["res_x"])
    sc.render.resolution_y = int(scene_spec["res_y"])
    sc.render.resolution_percentage = int(scene_spec.get("resolution_percentage", 100))
    sc.render.fps = int(scene_spec.get("fps", 30))
    if hasattr(sc.render.image_settings, "media_type"):
        sc.render.image_settings.media_type = "IMAGE"
    sc.render.image_settings.file_format = "PNG"

    if sc.render.engine == "CYCLES":
        sc.cycles.samples = int(scene_spec.get("samples", 64))
        sc.cycles.use_adaptive_sampling = bool(scene_spec.get("use_adaptive_sampling", True))
        sc.cycles.adaptive_threshold = float(scene_spec.get("adaptive_threshold", 0.03))
        sc.cycles.use_denoising = bool(scene_spec.get("use_denoising", True))
        sc.cycles.max_bounces = int(scene_spec.get("max_bounces", 6))
        sc.cycles.diffuse_bounces = int(scene_spec.get("diffuse_bounces", 3))
        sc.cycles.glossy_bounces = int(scene_spec.get("glossy_bounces", 3))
        sc.cycles.transmission_bounces = int(scene_spec.get("transmission_bounces", 6))
        sc.cycles.transparent_max_bounces = int(scene_spec.get("transparent_max_bounces", 8))

        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = True
            sc.cycles.device = "GPU"
        except Exception:
            sc.cycles.device = "CPU"

    if "view_transform" in scene_spec:
        try:
            sc.view_settings.view_transform = str(scene_spec["view_transform"])
        except Exception:
            pass
    if "look" in scene_spec:
        try:
            sc.view_settings.look = str(scene_spec["look"])
        except Exception:
            pass
    sc.view_settings.exposure = float(scene_spec.get("exposure", 0.0))

    if "world_rgb" in scene_spec:
        w = sc.world or bpy.data.worlds.new("ImaginedWorld")
        sc.world = w
        w.use_nodes = True
        nt = w.node_tree
        nt.nodes.clear()
        wo = nt.nodes.new("ShaderNodeOutputWorld")
        bg = nt.nodes.new("ShaderNodeBackground")
        w_rgb = tuple(float(c) for c in scene_spec["world_rgb"])
        bg.inputs["Color"].default_value = (*w_rgb, 1.0)
        bg.inputs["Strength"].default_value = float(scene_spec.get("world_strength", 1.0))
        nt.links.new(bg.outputs["Background"], wo.inputs["Surface"])

    return sc


def configure_lighting_from_spec(lights_spec: Sequence[Dict]):
    """Build dynamic 3D lights (`AREA`, `SUN`, `POINT`, `SPOT`) strictly from `lights_spec`."""
    import bpy
    from mathutils import Vector

    created = []
    for idx, lspec in enumerate(lights_spec):
        name = str(lspec.get("name", f"Light_{idx:02d}"))
        ltype = str(lspec["type"]).upper()
        ld = bpy.data.lights.new(name, type=ltype)
        ld.energy = float(lspec["energy"])
        ld.color = tuple(float(c) for c in lspec["color"])
        if ltype == "AREA" and "size" in lspec and hasattr(ld, "size"):
            ld.size = float(lspec["size"])
        elif ltype == "SPOT" and hasattr(ld, "spot_size"):
            ld.spot_size = math.radians(float(lspec.get("spot_size_deg", 45.0)))
            ld.spot_blend = float(lspec.get("spot_blend", 0.35))
        elif ltype == "SUN" and "angle_deg" in lspec and hasattr(ld, "angle"):
            ld.angle = math.radians(float(lspec["angle_deg"]))
        elif ltype == "POINT" and "shadow_soft_size" in lspec and hasattr(ld, "shadow_soft_size"):
            ld.shadow_soft_size = float(lspec["shadow_soft_size"])

        lo = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(lo)
        lo.location = tuple(float(x) for x in lspec["location"])
        if "target" in lspec:
            tgt = Vector(tuple(float(x) for x in lspec["target"]))
            lo.rotation_euler = (tgt - lo.location).to_track_quat("-Z", "Y").to_euler()
        elif "rotation_deg" in lspec:
            lo.rotation_euler = tuple(math.radians(float(a)) for a in lspec["rotation_deg"])
        created.append(lo)
    return created


# ---------------------------------------------------------------------------
# 2. Pure Native 3D Geometry Builders (Lofted Cross-Sections, Parametric Manifolds & BMesh)
# ---------------------------------------------------------------------------
def build_lofted_cross_section_mesh(
    name: str,
    rings_spec: Sequence[Dict],
    ring_segments: int = 32,
    cap_start: bool = True,
    cap_end: bool = True,
    subsurf_levels: int = 0,
    material: Optional[object] = None,
):
    """Build a quad-topology 3D mesh by lofting a sequence of custom cross-section rings (`rings_spec`).

    Each entry in `rings_spec` defines:
      - `center`: `(x, y, z)` ring center in 3D space
      - `rx`, `ry`: ring radii along local X and Y axes
      - `rot_deg`: optional `(rx_deg, ry_deg, rz_deg)` ring rotation
      - `squareness`: optional superellipse shaping parameter (`0.0` = ellipse, `>0` = rounded box)
      - `offset_fn`: optional `fn(angle_rad) -> radial_multiplier` for custom lobes, ridges, or folds
    """
    import bpy
    import bmesh
    from mathutils import Euler, Vector

    bm = bmesh.new()
    all_rings = []
    n_seg = int(ring_segments)

    for r_dict in rings_spec:
        cx, cy, cz = (float(v) for v in r_dict["center"])
        rx = max(1e-5, float(r_dict["rx"]))
        ry = max(1e-5, float(r_dict["ry"]))
        sq = float(r_dict.get("squareness", 0.0))
        rot_deg = r_dict.get("rot_deg", (0.0, 0.0, 0.0))
        eul = Euler(tuple(math.radians(float(a)) for a in rot_deg), "XYZ")
        offset_fn = r_dict.get("offset_fn", None)

        ring_verts = []
        p_exp = max(0.20, 1.0 - 0.65 * max(-0.5, min(0.95, sq)))
        for si in range(n_seg):
            th = (2.0 * math.pi * si) / n_seg
            cs, sn = math.cos(th), math.sin(th)
            ux = math.copysign(abs(cs) ** p_exp, cs)
            uy = math.copysign(abs(sn) ** p_exp, sn)
            mult = float(offset_fn(th)) if offset_fn is not None else 1.0
            local_pt = Vector((ux * rx * mult, uy * ry * mult, 0.0))
            local_pt.rotate(eul)
            ring_verts.append(bm.verts.new(Vector((cx, cy, cz)) + local_pt))
        all_rings.append(ring_verts)

    for ri in range(len(all_rings) - 1):
        r0, r1 = all_rings[ri], all_rings[ri + 1]
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            bm.faces.new((r0[si], r0[sn], r1[sn], r1[si]))

    if cap_start and all_rings:
        r0 = all_rings[0]
        c0 = sum((v.co for v in r0), Vector()) / len(r0)
        vc0 = bm.verts.new(c0)
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            bm.faces.new((r0[sn], r0[si], vc0))

    if cap_end and all_rings:
        r_last = all_rings[-1]
        c1 = sum((v.co for v in r_last), Vector()) / len(r_last)
        vc1 = bm.verts.new(c1)
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            bm.faces.new((r_last[si], r_last[sn], vc1))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(me)
    bm.free()

    for p in me.polygons:
        p.use_smooth = True

    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)

    if int(subsurf_levels) > 0:
        mod = ob.modifiers.new(name="Subsurf", type="SUBSURF")
        mod.levels = int(subsurf_levels)
        mod.render_levels = int(subsurf_levels)

    if material is not None:
        ob.data.materials.append(material)
    return ob


def build_parametric_3d_surface(
    name: str,
    subdivisions: int,
    contour_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    relief_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
    relief_amplitude: float = 0.0,
    material: Optional[object] = None,
):
    """Construct a 360° 3D manifold from a user-supplied 3D `contour_fn(ux, uy, uz) -> (px, py, pz)`
    and optional `relief_fn(ux, uy, uz) -> displacement_field`, storing normalized relief in the
    `relief_height` vertex float attribute.
    """
    import bpy
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=int(subdivisions), radius=1.0)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    coords = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    dirs = coords / np.maximum(1e-6, norms)
    ux, uy, uz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    px, py, pz = contour_fn(ux, uy, uz)

    if relief_fn is not None:
        raw_h = np.asarray(relief_fn(ux, uy, uz), dtype=np.float32)
        p02, p98 = np.percentile(raw_h, [2.0, 98.0])
        h01 = np.clip((raw_h - p02) / max(1e-6, p98 - p02), 0.0, 1.0).astype(np.float32)
    else:
        h01 = np.full(len(bm.verts), 0.5, dtype=np.float32)

    if float(relief_amplitude) != 0.0:
        disp = (h01 - 0.5) * float(relief_amplitude)
        px = px + ux * disp
        py = py + uy * disp
        pz = pz + uz * disp

    for idx, vert in enumerate(bm.verts):
        vert.co.x = float(px[idx])
        vert.co.y = float(py[idx])
        vert.co.z = float(pz[idx])

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(me)
    bm.free()

    for p in me.polygons:
        p.use_smooth = True

    attr = me.attributes.new(name="relief_height", type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", h01.tolist())

    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    if material is not None:
        ob.data.materials.append(material)
    return ob


def build_custom_bmesh_assembly(
    name: str,
    build_fn: Callable,
    material: Optional[object] = None,
    subsurf_levels: int = 0,
    use_smooth: bool = True,
):
    """Build any custom hard-surface, architectural, or organic 3D mesh directly via a `bmesh` callback."""
    import bpy
    import bmesh

    bm = bmesh.new()
    build_fn(bm)
    bm.normal_update()
    me = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = bool(use_smooth)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)

    if int(subsurf_levels) > 0:
        mod = ob.modifiers.new(name="Subsurf", type="SUBSURF")
        mod.levels = int(subsurf_levels)
        mod.render_levels = int(subsurf_levels)

    if material is not None:
        ob.data.materials.append(material)
    return ob


# ---------------------------------------------------------------------------
# 3. Dynamic Cycles Material, Eyeball, Hair Curves & Custom Armature Builders
# ---------------------------------------------------------------------------
def build_custom_pbr_or_sss_material(name: str, mat_spec: Dict):
    """Construct a native Cycles material shader dynamically from `mat_spec`."""
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    base_rgb = tuple(float(c) for c in mat_spec["base_rgb"])
    bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = float(mat_spec.get("roughness", 0.45))
    bsdf.inputs["Metallic"].default_value = float(mat_spec.get("metallic", 0.0))
    if "specular" in mat_spec and "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(mat_spec["specular"])
    if "ior" in mat_spec and "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = float(mat_spec["ior"])
    if "transmission" in mat_spec and "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = float(mat_spec["transmission"])
    if "coat_weight" in mat_spec and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = float(mat_spec["coat_weight"])
        bsdf.inputs["Coat Roughness"].default_value = float(mat_spec.get("coat_roughness", 0.1))

    sss_w = float(mat_spec.get("subsurface_weight", 0.0))
    if sss_w > 0.0:
        bsdf.subsurface_method = str(mat_spec.get("subsurface_method", "RANDOM_WALK_SKIN"))
        bsdf.inputs["Subsurface Weight"].default_value = sss_w
        if "subsurface_radius" in mat_spec:
            bsdf.inputs["Subsurface Radius"].default_value = tuple(float(c) for c in mat_spec["subsurface_radius"])
        bsdf.inputs["Subsurface Scale"].default_value = float(mat_spec.get("subsurface_scale", 0.003))

    if "emission_rgb" in mat_spec and "Emission Color" in bsdf.inputs:
        e_rgb = tuple(float(c) for c in mat_spec["emission_rgb"])
        bsdf.inputs["Emission Color"].default_value = (*e_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = float(mat_spec.get("emission_strength", 1.0))

    if "valley_rgb" in mat_spec and "ridge_rgb" in mat_spec:
        at_node = n.new("ShaderNodeAttribute")
        at_node.attribute_name = str(mat_spec.get("relief_attribute", "relief_height"))
        mix_col = n.new("ShaderNodeMix")
        mix_col.data_type = "RGBA"
        v_rgb = tuple(float(c) for c in mat_spec["valley_rgb"])
        r_rgb = tuple(float(c) for c in mat_spec["ridge_rgb"])
        mix_col.inputs[6].default_value = (*v_rgb, 1.0)
        mix_col.inputs[7].default_value = (*r_rgb, 1.0)
        L(at_node.outputs["Fac"], mix_col.inputs[0])
        L(mix_col.outputs[2], bsdf.inputs["Base Color"])

        if float(mat_spec.get("relief_bump_strength", 0.0)) > 0.0:
            bmp = n.new("ShaderNodeBump")
            bmp.inputs["Strength"].default_value = float(mat_spec["relief_bump_strength"])
            bmp.inputs["Distance"].default_value = float(mat_spec.get("relief_bump_distance", 0.002))
            L(at_node.outputs["Fac"], bmp.inputs["Height"])
            L(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    elif float(mat_spec.get("noise_bump_strength", 0.0)) > 0.0:
        tc = n.new("ShaderNodeTexCoord")
        ns = n.new("ShaderNodeTexNoise")
        ns.inputs["Scale"].default_value = float(mat_spec.get("noise_bump_scale", 200.0))
        ns.inputs["Detail"].default_value = float(mat_spec.get("noise_bump_detail", 4.0))
        L(tc.outputs["Object"], ns.inputs["Vector"])
        bmp = n.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = float(mat_spec["noise_bump_strength"])
        bmp.inputs["Distance"].default_value = float(mat_spec.get("noise_bump_distance", 0.001))
        L(ns.outputs["Fac"], bmp.inputs["Height"])
        L(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def build_custom_3d_eyeballs(eye_spec: Dict):
    """Build custom 3D eyeballs from `eye_spec` (`centers`, `radius`, `iris_rgb`, `sclera_rgb`, `pupil_rgb`, `pupil_ratio`, `iris_ratio`)."""
    import bpy
    import bmesh

    name_prefix = str(eye_spec.get("name_prefix", "Eye"))
    mat = bpy.data.materials.new(f"{name_prefix}_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = float(eye_spec.get("roughness", 0.02))
    bsdf.inputs["IOR"].default_value = float(eye_spec.get("ior", 1.44))
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = float(eye_spec.get("coat_weight", 0.90))
        bsdf.inputs["Coat Roughness"].default_value = float(eye_spec.get("coat_roughness", 0.008))
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    L(tc.outputs["Generated"], sep.inputs["Vector"])

    dx = n.new("ShaderNodeMath"); dx.operation = "SUBTRACT"; dx.inputs[1].default_value = 0.5
    L(sep.outputs["X"], dx.inputs[0])
    dz = n.new("ShaderNodeMath"); dz.operation = "SUBTRACT"; dz.inputs[1].default_value = 0.5
    L(sep.outputs["Z"], dz.inputs[0])
    dx2 = n.new("ShaderNodeMath"); dx2.operation = "MULTIPLY"; L(dx.outputs["Value"], dx2.inputs[0]); L(dx.outputs["Value"], dx2.inputs[1])
    dz2 = n.new("ShaderNodeMath"); dz2.operation = "MULTIPLY"; L(dz.outputs["Value"], dz2.inputs[0]); L(dz.outputs["Value"], dz2.inputs[1])
    r2 = n.new("ShaderNodeMath"); r2.operation = "ADD"; L(dx2.outputs["Value"], r2.inputs[0]); L(dz2.outputs["Value"], r2.inputs[1])
    rad = n.new("ShaderNodeMath"); rad.operation = "SQRT"; L(r2.outputs["Value"], rad.inputs[0])

    pupil_r = float(eye_spec["pupil_ratio"])
    iris_r = float(eye_spec["iris_ratio"])
    pupil_rgb = tuple(float(c) for c in eye_spec.get("pupil_rgb", (0.005, 0.005, 0.008)))
    iris_rgb = tuple(float(c) for c in eye_spec["iris_rgb"])
    sclera_rgb = tuple(float(c) for c in eye_spec["sclera_rgb"])

    cr = n.new("ShaderNodeValToRGB")
    elems = cr.color_ramp.elements
    elems[0].position = pupil_r; elems[0].color = (*pupil_rgb, 1.0)
    elems[1].position = iris_r; elems[1].color = (*sclera_rgb, 1.0)
    e_ir = cr.color_ramp.elements.new(pupil_r + 0.20 * (iris_r - pupil_r))
    e_ir.color = (*iris_rgb, 1.0)
    L(rad.outputs["Value"], cr.inputs["Fac"])
    L(cr.outputs["Color"], bsdf.inputs["Base Color"])

    created = {}
    radius = float(eye_spec["radius"])
    for key, center_pt in eye_spec["centers"].items():
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=24, radius=radius)
        bm.normal_update()
        me = bpy.data.meshes.new(f"{name_prefix}_{key}_Mesh")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new(f"{name_prefix}_{key}", me)
        bpy.context.collection.objects.link(ob)
        ob.location = tuple(float(v) for v in center_pt)
        ob.data.materials.append(mat)
        created[key] = ob
    return created


def build_native_curves_groom(
    name: str,
    strands_pts: Sequence[Sequence[Tuple[float, float, float]]],
    strands_radii: Sequence[Sequence[float]],
    root_rgb: Tuple[float, float, float],
    tip_rgb: Tuple[float, float, float],
    roughness: float = 0.55,
):
    """Create a pure native Blender 5.2 `CURVES` (`CATMULL_ROM`, `resolution=6`) 3D groom object."""
    import bpy

    hc = bpy.data.hair_curves.new(name)
    hc.add_curves([len(p) for p in strands_pts])

    flat_pos, flat_rad = [], []
    for pts, rads in zip(strands_pts, strands_radii):
        for p, r in zip(pts, rads):
            flat_pos.extend((float(p[0]), float(p[1]), float(p[2])))
            flat_rad.append(float(r))

    hc.points.foreach_set("position", flat_pos)
    hc.points.foreach_set("radius", flat_rad)

    try:
        ct_attr = hc.attributes.get("curve_type") or hc.attributes.new("curve_type", "INT8", "CURVE")
        ct_attr.data.foreach_set("value", [0] * len(strands_pts))
        res_attr = hc.attributes.get("resolution") or hc.attributes.new("resolution", "INT", "CURVE")
        res_attr.data.foreach_set("value", [6] * len(strands_pts))
    except Exception:
        pass

    mat = bpy.data.materials.new(f"{name}_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    n, L = nt.nodes, nt.links.new
    n.clear()

    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Specular IOR Level"].default_value = 0.08
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.0
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])

    hi = n.new("ShaderNodeHairInfo")
    cr = n.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.0
    cr.color_ramp.elements[0].color = (*root_rgb, 1.0)
    cr.color_ramp.elements[1].position = 1.0
    cr.color_ramp.elements[1].color = (*tip_rgb, 1.0)
    L(hi.outputs["Intercept"], cr.inputs["Fac"])
    L(cr.outputs["Color"], bsdf.inputs["Base Color"])

    ob = bpy.data.objects.new(name, hc)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def build_custom_armature_from_spec(
    rig_name: str,
    bones_spec: Sequence[Dict],
    ik_chains_spec: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Build any custom native Blender `Armature` dynamically from `bones_spec` and `ik_chains_spec`.

    Each entry in `bones_spec`:
      - `name`: bone name
      - `head`: `(x, y, z)` world rest head position
      - `tail`: `(x, y, z)` world rest tail position
      - `parent`: optional parent bone name
      - `connect`: optional bool (`True` if connected to parent tail)

    Each entry in `ik_chains_spec`:
      - `name`: controller key
      - `ik_bone`: bone receiving the `IK` constraint
      - `end_bone`: optional terminal hand/foot bone receiving world-space `COPY_ROTATION`
      - `chain_count`: integer IK chain length
      - `pole_offset`: `(dx, dy, dz)` world offset for the pole target
      - `pole_angle_deg`: optional pole angle in degrees
    """
    import bpy
    from mathutils import Vector

    arm_data = bpy.data.armatures.new(f"{rig_name}_Data")
    rig = bpy.data.objects.new(rig_name, arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")

    eb = arm_data.edit_bones
    bone_rest_heads = {}
    bone_rest_tails = {}
    for bspec in bones_spec:
        bname = str(bspec["name"])
        b = eb.new(bname)
        b.head = Vector(tuple(float(v) for v in bspec["head"]))
        b.tail = Vector(tuple(float(v) for v in bspec["tail"]))
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0.0, 0.0, 0.05))
        pname = bspec.get("parent")
        if pname and pname in eb:
            b.parent = eb[pname]
            b.use_connect = bool(bspec.get("connect", False))
        bone_rest_heads[bname] = b.head.copy()
        bone_rest_tails[bname] = b.tail.copy()

    bpy.ops.object.mode_set(mode="OBJECT")

    controllers = {}
    for cspec in ik_chains_spec or []:
        ckey = str(cspec["name"])
        ik_bname = str(cspec["ik_bone"])
        end_bname = cspec.get("end_bone")

        target_pos = bone_rest_heads.get(end_bname, bone_rest_tails.get(ik_bname, Vector((0.0, 0.0, 0.0))))
        t_ob = bpy.data.objects.new(f"{rig_name}_Target_{ckey}", None)
        bpy.context.collection.objects.link(t_ob)
        t_ob.location = target_pos

        p_ob = None
        if "pole_offset" in cspec:
            p_ob = bpy.data.objects.new(f"{rig_name}_Pole_{ckey}", None)
            bpy.context.collection.objects.link(p_ob)
            p_ob.location = bone_rest_heads.get(ik_bname, target_pos) + Vector(
                tuple(float(v) for v in cspec["pole_offset"])
            )

        pb_ik = rig.pose.bones[ik_bname]
        ik = pb_ik.constraints.new("IK")
        ik.target = t_ob
        if p_ob is not None:
            ik.pole_target = p_ob
            if "pole_angle_deg" in cspec:
                ik.pole_angle = math.radians(float(cspec["pole_angle_deg"]))
        ik.chain_count = int(cspec.get("chain_count", 2))

        if end_bname and end_bname in rig.pose.bones:
            pb_end = rig.pose.bones[end_bname]
            cr = pb_end.constraints.new("COPY_ROTATION")
            cr.target = t_ob
            cr.target_space = "WORLD"
            cr.owner_space = "WORLD"

        controllers[ckey] = {"target": t_ob, "pole": p_ob}

    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"

    return {"rig": rig, "controllers": controllers}


def set_bone_world_contact_pose(
    target_empty,
    contact_pt: Tuple[float, float, float],
    forward_dir: Tuple[float, float, float],
    contact_normal: Tuple[float, float, float],
    forward_offset: float = 0.05,
    normal_offset: float = 0.012,
    invert_normal_axis: bool = False,
    pole_empty=None,
    pole_pos: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Solve an orthonormal world-space rotation matrix `[x_ax | y_ax | z_ax]` from `forward_dir` and
    `contact_normal` and position `target_empty` so a hand, paw, or end-effector contacts `contact_pt` cleanly.
    """
    from mathutils import Matrix, Vector

    y_ax = Vector(forward_dir).normalized()
    p_n = Vector(contact_normal)
    p_n = (p_n - y_ax * p_n.dot(y_ax)).normalized()
    x_ax = (-p_n) if invert_normal_axis else p_n
    z_ax = x_ax.cross(y_ax).normalized()
    x_ax = y_ax.cross(z_ax).normalized()

    rot_mat = Matrix((
        (x_ax.x, y_ax.x, z_ax.x),
        (x_ax.y, y_ax.y, z_ax.y),
        (x_ax.z, y_ax.z, z_ax.z),
    ))
    target_empty.location = Vector(contact_pt) - y_ax * float(forward_offset) - p_n * float(normal_offset)
    target_empty.rotation_euler = rot_mat.to_euler("XYZ")
    if pole_empty is not None and pole_pos is not None:
        pole_empty.location = Vector(pole_pos)


# ---------------------------------------------------------------------------
# 4. Blender 5.2 LTS Video Sequencer (VSE) H.264 + AAC Audio Muxer
# ---------------------------------------------------------------------------
def mux_frames_to_mp4_vse(
    frames_dir: str,
    output_mp4_path: str,
    res_x: int,
    res_y: int,
    fps: int = 30,
    frame_hold: int = 1,
    n_rendered_frames: int = 300,
    audio_source_path: Optional[str] = None,
) -> str:
    """Encode rendered PNG frames + optional audio stream into an H.264/AAC `.mp4` in Blender 5.2 VSE."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.use_sequencer = True
    sc.render.resolution_x = int(res_x)
    sc.render.resolution_y = int(res_y)
    sc.render.resolution_percentage = 100
    sc.render.fps = int(fps)

    if not sc.sequence_editor:
        sc.sequence_editor_create()
    seq = sc.sequence_editor
    seqs = seq.strips if hasattr(seq, "strips") else seq.sequences

    cur_f = 1
    for idx in range(int(n_rendered_frames)):
        fp = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        if not os.path.exists(fp):
            continue
        st = seqs.new_image(name=f"F_{idx:04d}", filepath=fp, channel=2, frame_start=cur_f)
        st.frame_final_duration = int(frame_hold)
        cur_f += int(frame_hold)

    sc.frame_start = 1
    sc.frame_end = max(1, cur_f - 1)

    if audio_source_path and os.path.exists(audio_source_path):
        try:
            snd = seqs.new_sound(name="AudioTrack", filepath=audio_source_path, channel=1, frame_start=1)
            snd.frame_final_duration = sc.frame_end
        except Exception as e:
            print("Warning: Audio load skipped:", e)

    if hasattr(sc.render.image_settings, "media_type"):
        sc.render.image_settings.media_type = "VIDEO"
    else:
        sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.ffmpeg.audio_codec = "AAC"
    sc.render.ffmpeg.audio_bitrate = 192
    sc.render.filepath = output_mp4_path

    bpy.ops.render.render(animation=True)
    return output_mp4_path


# ---------------------------------------------------------------------------
# 7. Zero-Rush Exact-Match Verification Gate
# ---------------------------------------------------------------------------
def assert_quality_iteration_gate(
    item_name: str,
    verified_views: Sequence[str],
    required_views: Sequence[str] = ("front", "three_quarter", "side", "detail"),
    geometry_clean: bool = False,
    exact_visual_match_confirmed: bool = False,
    measured_color_delta_e: Optional[float] = None,
    max_color_delta_e: float = 3.5,
) -> Dict[str, object]:
    """Enforce the Zero-Rush Exact-Match Perfection Gate before advancing past an asset.

    Raises `RuntimeError` if any required inspection view is missing, geometry overlaps/defects
    remain, color calibration exceeds `max_color_delta_e`, or exact visual match is not yet confirmed.
    """
    verified_set = {str(v).strip().lower() for v in verified_views}
    missing_views = [v for v in required_views if v.lower() not in verified_set]
    failures: List[str] = []

    if missing_views:
        failures.append(f"Missing required turnaround inspection views: {missing_views}")
    if not geometry_clean:
        failures.append("Geometry/attachment audit not marked clean (check overlaps, seams, and grounding)")
    if measured_color_delta_e is not None and float(measured_color_delta_e) > float(max_color_delta_e):
        failures.append(
            f"Color calibration delta_E={measured_color_delta_e:.2f} exceeds tolerance {max_color_delta_e:.2f}"
        )
    if not exact_visual_match_confirmed:
        failures.append(
            "Exact visual match not yet confirmed — do not rush; refine geometry, proportions, and materials first"
        )

    report = {
        "item": item_name,
        "passed": len(failures) == 0,
        "verified_views": sorted(verified_set),
        "missing_views": missing_views,
        "measured_color_delta_e": measured_color_delta_e,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"Quality gate blocked advancement for '{item_name}': " + "; ".join(failures))
    return report


# ---------------------------------------------------------------------------
# 8. Open-Source MPFB2 Full-Body Humanoid & Orthonormal Hand-Grip Helpers
# ---------------------------------------------------------------------------
def build_imagined_human_3d(char_spec: Dict) -> Dict:
    """Build a complete head-to-toe 3D human character dynamically from the imagined `char_spec` using the open-source MPFB2 Blender Extension.

    Enforces the 100% Collinear Zero-Spaghetti Arm Law:
    `detailed_helpers=True, feet_on_ground=False` -> `TargetService.bake_targets(human)` ->
    `HumanService.add_builtin_rig(human, "default")` at (0,0,0) -> lift `rig.location.z = -z_min_feet`.
    All phenotype traits, skin, hair, and clothing assets are driven dynamically by `char_spec`.
    """
    import bpy
    import addon_utils

    if "bl_ext.user_default.mpfb" not in bpy.context.preferences.addons:
        addon_utils.enable("bl_ext.user_default.mpfb", default_set=False, persistent=False)

    from bl_ext.user_default.mpfb.services.assetservice import AssetService
    from bl_ext.user_default.mpfb.services.humanservice import HumanService
    from bl_ext.user_default.mpfb.services.targetservice import TargetService

    char_name = char_spec.get("name", "ImaginedCharacter3D")
    macro = TargetService.get_default_macro_info_dict()
    macro.update(
        {
            "gender": float(char_spec.get("gender", 0.5)),
            "age": float(char_spec.get("age", 0.50)),
            "muscle": float(char_spec.get("muscle", 0.50)),
            "weight": float(char_spec.get("weight", 0.50)),
            "height": float(char_spec.get("height", 0.50)),
            "proportions": float(char_spec.get("proportions", 0.50)),
            "cupsize": float(char_spec.get("cupsize", 0.0 if float(char_spec.get("gender", 0.5)) >= 0.7 else 0.5)),
            "race": char_spec.get("race", {"caucasian": 0.34, "asian": 0.33, "african": 0.33}),
        }
    )

    human = HumanService.create_human(
        mask_helpers=True,
        detailed_helpers=True,
        extra_vertex_groups=True,
        feet_on_ground=False,
        scale=float(char_spec.get("scale", 0.1)),
        macro_detail_dict=macro,
    )
    human.name = f"{char_name}_Body"

    skin_search = str(char_spec.get("skin_keyword", "")).lower()
    skins = AssetService.list_mhmat_skin_assets()
    chosen_skin = None
    for s in skins:
        if skin_search and skin_search in str(s).lower():
            chosen_skin = str(s)
            break
    if not chosen_skin and skins:
        chosen_skin = str(skins[0])
    if chosen_skin:
        HumanService.set_character_skin(chosen_skin, human, skin_type="MAKESKIN")

    for tname, tval in char_spec.get("custom_targets", {}).items():
        tpath = TargetService.target_full_path(tname)
        if tpath:
            TargetService.load_target(human, tpath, weight=float(tval))

    TargetService.bake_targets(human)
    rig = HumanService.add_builtin_rig(human, "default")
    rig.name = f"{char_name}_Rig"
    human.parent = rig

    loaded_assets: Dict[str, object] = {}
    for item in char_spec.get("assets", []):
        cat = item["category"]
        kw = str(item.get("keyword", "")).lower()
        candidates = AssetService.list_mhclo_assets(cat)
        match_path = None
        for c in candidates:
            if kw in str(c).lower():
                match_path = str(c)
                break
        if not match_path and candidates:
            match_path = str(candidates[0])
        if match_path:
            asset_ob = HumanService.add_mhclo_asset(
                match_path,
                human,
                asset_type=cat.capitalize(),
                subdiv_levels=int(item.get("subdiv_levels", 0)),
                material_type="MAKESKIN",
                set_up_rigging=True,
            )
            offset_mm = float(item.get("normal_offset_mm", 2.5 if cat in ("clothes", "shoes") else 0.0))
            if offset_mm != 0.0:
                for v in asset_ob.data.vertices:
                    v.co += v.normal * (offset_mm * 0.001)
                asset_ob.data.update()
            if "rgb" in item:
                rgba = (*[float(c) for c in item["rgb"]], 1.0)
                mat = bpy.data.materials.new(f"{char_name}_{cat}_Mat")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = rgba
                    bsdf.inputs["Roughness"].default_value = float(item.get("roughness", 0.78))
                    bsdf.inputs["Specular IOR Level"].default_value = float(item.get("specular", 0.08))
                    bsdf.inputs["Coat Weight"].default_value = 0.0
                asset_ob.data.materials.clear()
                asset_ob.data.materials.append(mat)
            loaded_assets[f"{cat}_{kw or len(loaded_assets)}"] = asset_ob

    body_gi = human.vertex_groups["body"].index if "body" in human.vertex_groups else None
    body_zs = [
        v.co.z
        for v in human.data.vertices
        if body_gi is None or any(g.group == body_gi and g.weight > 0.5 for g in v.groups)
    ]
    z_min_feet = min(body_zs) if body_zs else 0.0
    rig.location.z = -float(z_min_feet)
    bpy.context.view_layer.update()
    return {"rig": rig, "body": human, "assets": loaded_assets}


def set_finger_cup_pose(
    rig: object,
    side: str,
    curl_deg: float = 28.0,
    spread_close_scale: float = 1.0,
    thumb_pitch_deg: float = 22.0,
) -> None:
    """Close starfish finger splay around local Z and curl phalanges 1..3 around local +X (`dot_palm = +0.94`)."""
    from mathutils import Euler
    side_s = 1.0 if side.upper() == "L" else -1.0
    adduct_z = {
        2: math.radians(-11.0 * side_s * spread_close_scale),
        3: 0.0,
        4: math.radians(8.5 * side_s * spread_close_scale),
        5: math.radians(17.0 * side_s * spread_close_scale),
    }
    c_rad = math.radians(float(curl_deg))
    for f_idx in (2, 3, 4, 5):
        for p_idx, w in ((1, 0.85), (2, 1.05), (3, 0.75)):
            bname = f"finger{f_idx}-{p_idx}.{side}"
            pb = rig.pose.bones.get(bname)
            if pb:
                pb.rotation_mode = "XYZ"
                z_rot = adduct_z[f_idx] if p_idx == 1 else 0.0
                pb.rotation_euler = Euler((c_rad * w, 0.0, z_rot), "XYZ")

    t_rad = math.radians(float(thumb_pitch_deg))
    for p_idx, w in ((1, 0.65), (2, 0.90), (3, 0.65)):
        bname = f"finger1-{p_idx}.{side}"
        pb = rig.pose.bones.get(bname)
        if pb:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = Euler((t_rad * w, 0.0, math.radians(12.0 * side_s * (1 if p_idx == 1 else 0))), "XYZ")


def set_hand_grip_contact(
    rig: object,
    side: str,
    surface_contact_world: Tuple[float, float, float],
    finger_dir_world: Tuple[float, float, float],
    palm_normal_world: Tuple[float, float, float],
    elbow_pole_world: Tuple[float, float, float],
    curl_deg: float = 28.0,
) -> Dict[str, object]:
    """Orthonormal World-Space Hand-Grip Solver (`COPY_ROTATION` + IK + symmetric `-115°` pole angle).

    Places the palm center directly against `surface_contact_world` with inward/upward `palm_normal_world`
    and `finger_dir_world`, completely preventing back-of-hand flips or finger-mesh slicing.
    """
    side = side.upper()
    s_pt = Vector(surface_contact_world)
    y_ax = Vector(finger_dir_world).normalized()
    n_palm = Vector(palm_normal_world).normalized()
    y_ax = (y_ax - y_ax.dot(n_palm) * n_palm).normalized()

    if side == "L":
        x_ax = n_palm
    else:
        x_ax = -n_palm
    z_ax = x_ax.cross(y_ax).normalized()
    x_ax = y_ax.cross(z_ax).normalized()

    rot_mat = Matrix((x_ax, y_ax, z_ax)).transposed().to_4x4()
    wrist_pos = s_pt - 0.065 * y_ax - 0.016 * n_palm

    ik_name = f"{rig.name}_HandIK_{side}"
    ik_ob = bpy.data.objects.get(ik_name) or bpy.data.objects.new(ik_name, None)
    if ik_ob.name not in bpy.context.collection.objects:
        bpy.context.collection.objects.link(ik_ob)
    rot_mat.translation = wrist_pos
    ik_ob.matrix_world = rot_mat

    pole_name = f"{rig.name}_ElbowPole_{side}"
    pole_ob = bpy.data.objects.get(pole_name) or bpy.data.objects.new(pole_name, None)
    if pole_ob.name not in bpy.context.collection.objects:
        bpy.context.collection.objects.link(pole_ob)
    pole_ob.location = Vector(elbow_pole_world)

    forearm = rig.pose.bones.get(f"lowerarm02.{side}")
    if forearm:
        for c in list(forearm.constraints):
            if c.type == "IK":
                forearm.constraints.remove(c)
        c_ik = forearm.constraints.new("IK")
        c_ik.target = ik_ob
        c_ik.pole_target = pole_ob
        c_ik.chain_count = 4
        c_ik.pole_angle = math.radians(-115.0)
        c_ik.use_rotation = False

    wrist = rig.pose.bones.get(f"wrist.{side}")
    if wrist:
        for c in list(wrist.constraints):
            if c.type in ("COPY_ROTATION", "DAMPED_TRACK", "TRACK_TO"):
                wrist.constraints.remove(c)
        c_rot = wrist.constraints.new("COPY_ROTATION")
        c_rot.target = ik_ob
        c_rot.target_space = "WORLD"
        c_rot.owner_space = "WORLD"

    set_finger_cup_pose(rig, side, curl_deg=curl_deg)
    bpy.context.view_layer.update()
    return {"ik_target": ik_ob, "pole_target": pole_ob}
