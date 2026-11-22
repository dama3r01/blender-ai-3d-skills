# Universal Imagine-to-3D & Imagine-to-3D-Video Playbook (`imagine_to_3d`)

> **STRICT READ-ONLY RULE FOR AI AGENTS (DO NOT MODIFY ANY `.py` OR `.md` FILES HERE):**
> - **NEVER EDIT, OVERWRITE, REFACTOR, APPEND TO, OR DELETE** any Python (`.py`) module or Playbook (`.md`) file inside `imagine_to_3d/`.
> - Treat every `.py` file in this directory (`imagine_3d_core.py`, `imagine_portrait_object_core.py`, `imagine_vision_solver.py`) as a **strictly read-only library**.
> - **How AI agents must use this module:** Read this Playbook, add `imagine_to_3d/` to `sys.path`, and **import/call** its functions from your own separate external project script. If you need custom helpers or project-specific logic, write them in your own external project file — **NEVER modify these core library files under any circumstances**.

> **NON-NEGOTIABLE QUALITY MANDATE ("ZERO RUSH — WORK UNTIL EXACT PERFECTION"):**
> - **NEVER RUSH TO FINISH:** Speed is never a justification for approximate proportions, crude primitives, flat surfaces, detached limbs, floating props, z-fighting, or generic shaders.
> - **WORK UNTIL IT IS EXACT AND PERFECT:** Keep refining geometry, silhouette, proportions, surface relief, attachments, lighting, and materials across as many visual inspection cycles as needed **until the 3D asset looks completely faithful, seamless, and identical to the intended vision**.
> - **NEVER SETTLE FOR "CLOSE ENOUGH":** Do not move on to the next item, do not begin rigging or animation, and do not launch final rendering while ANY visual, anatomical, topological, or shading defect remains.

> **Core Construction Principles ("100% Pure Imagination -> Direct Native Blender 3D Construction"):**
> 1. **Everything Is Conceived Fresh From the Prompt:**
>    - Every geometry, topology, silhouette, proportion, surface detail, skeleton hierarchy, material shader, color palette, environment, lighting mood, camera lens, and animation curve is constructed **100% fresh from the prompt specification**.
>    - Never reuse hardcoded templates, fixed outfits, fixed colors, or fixed environments across distinct tasks.
> 2. **100% Pure Native Blender 5.2 LTS Construction (`From Scratch`):**
>    - Construct all 3D meshes, characters, creatures, props, vehicles, architectures, hair/fur grooms, armatures, and shaders directly from scratch using native Blender 5.2 APIs (`bmesh`, mathematical surface/volume functions, lofted cross-section profiles, subdivision/displacement modifiers, native `Curves`, native `Armature` bones/constraints, and native Cycles node graphs).
> 3. **Strict Item-by-Item 3D Build & 4-Angle Verification Gate:**
>    - Construct and verify **one 3D item at a time (`Item 1 -> Item 2 -> Item 3...`)**. For every item, render a **4-Angle Standalone 3D Verification Sheet (`Front | 3/4 View | Side Profile | Detail / Clay`)**, inspect every angle honestly and critically, and keep iterating until the asset is spotless before advancing.

---

## 1. Universal Imagination-Driven 3D Architecture (`Any Subject x Any Style`)

| Imagined Subject | Native Blender 3D Geometry & Surface Construction (`imagine_3d_core.py`) | Native Shaders, Skeleton & Motion |
| :--- | :--- | :--- |
| **1. Character or Being (Any Style, Proportion, Anatomy, or Era)** | **Stylized/Custom:** `build_lofted_cross_section_mesh(...)` + `build_custom_bmesh_assembly(...)` + `build_parametric_3d_surface(...)`. **Realistic Anatomical Humans:** `build_imagined_human_3d(...)` / `imagine_portrait_object_core.py` (dynamically configures open-source **MPFB2** Blender Extension targets, collinear rigging at `(0,0,0)` before ground lift, 3D RBF facial sculpting, native `CATMULL_ROM` hair/mustache/brow curves, and 3D tailored garments) | Custom Cycles multi-layer surface shaders (`build_custom_pbr_or_sss_material` / `RANDOM_WALK_SKIN`), 3D multi-part wet eyes (`build_custom_3d_eyeballs`), custom-hierarchy native `Armature` (`build_custom_armature_from_spec`), and orthonormal hand-grip solver (`set_hand_grip_contact`, `set_finger_cup_pose`) |
| **2. Creature, Animal, Monster, or Fantasy Entity** | **Lofted Multi-Branch Quad Topology**: Custom spine, torso, limbs, digits, tail, wings, horns, or tentacles lofted along 3D spline paths with anatomical joint loops, interior 3D mouth cavity, and 3D surface relief | Custom fur/scale/chitin shaders + custom multi-chain IK/FK armature + locomotion & secondary follow-through |
| **3. Object, Artifact, Prop, Organ, Plant, or Weapon** | **`build_parametric_3d_surface(...)` / `build_lofted_cross_section_mesh(...)` / `build_custom_bmesh_assembly(...)`**: Custom 360° 3D manifold shaped by 3D parametric equations $(X(u,v), Y(u,v), Z(u,v))$ and multi-band 3D surface displacement fields | Custom procedural Cycles node graphs (subsurface, metallic, anisotropic, glass/refraction, emission, cavity-driven color & bump) + dynamic state shape keys |
| **4. Vehicle, Spacecraft, Robot, or Mechanism** | **Multi-Part Articulated 3D Assembly**: Custom hard-surface quad hull/chassis with panel seams, bevels, cockpit/interior, and independent moving 3D sub-parts/wheels/rotors/actuators | Custom metallic/clearcoat/weathered PBR shaders, refractive glass, emissive systems, and kinematic constraints |
| **5. 3D World, Architecture, Landscape, or Interior** | **Complete 360° 3D Environment (`Zero Flat Cards`)**: Custom 3D terrain/water/ground, extruded 3D architecture with true 3D depth, 3D foliage/furniture/props, and atmospheric volume | Custom 3D lighting choreography (`configure_lighting_from_spec`), volumetric atmosphere, and 3D camera motion |

---

## 2. The 4 Universal Execution Stages (Zero Shortcuts)

### Stage 1: Complete 3D Design Specification
1. **Decompose the Concept into Discrete 3D Items & Story Beats:**
   - Identify every distinct 3D item (`Item 1, Item 2, Item 3...` plus the `3D Environment/World`), its visual style, and its role in the scene or narrative.
2. **Formulate Each Item's Custom 3D Mathematical & Visual Spec:**
   - For every item, define its custom 3D proportions, silhouette curves, cross-section profiles, 3D surface relief functions, material shader parameters, custom bone hierarchy (if articulated), and motion trajectories.

### Stage 2: Strict Item-by-Item Construction & Iterative Perfection Loop
1. **Custom Lofted, Parametric & Sculpted 3D Geometry (`Zero Raw Primitives`):**
   - Shape every organic and hard-surface component using custom cross-section lofts (`build_lofted_cross_section_mesh`), parametric 3D manifolds (`build_parametric_3d_surface`), or detailed `bmesh` operators (`build_custom_bmesh_assembly`) with continuous 3D surface relief stored in vertex attributes (`relief_height`) for true 360° depth.
2. **Custom Armatures & World-Space Contact Solving:**
   - Build armatures dynamically from `bones_spec` (`build_custom_armature_from_spec`) and use `set_bone_world_contact_pose` for exact world-space surface contact with zero clipping or hovering.
3. **Mandatory 4-Angle Inspection & Zero-Rush Gate (`assert_quality_iteration_gate`):**
   - Render `Front`, `3/4 View`, `Side Profile`, and `Detail / Clay` views for every individual item.
   - Inspect every detail (hands/paws attachment, limb joints, facial depth, surface relief, material response, and grounding).
   - **Loop until perfect:** If ANY defect, proportion error, or visual weakness exists, fix it and re-render all 4 angles. Only advance once `assert_quality_iteration_gate(...)` passes cleanly.

### Stage 3 & Stage 4: Full-Scene Assembly, Multi-Angle QA Gate & GPU Delivery
1. **Full-Scene / Multi-Shot QA Sheet (Do Not Rush Into Final Render):**
   - Assemble all verified 3D items into the 3D world, animate camera trajectories and character/object physics, and render a **Multi-Camera 3D Turnaround Sheet** (for 3D stills) or **Anchor Keyframe Storyboard QA Sheet** (for 3D videos).
   - Inspect every shot for framing, lighting, overlaps, grounding, and silhouette readability. Fix every single issue before starting the final render.
2. **Fast GPU Cycles OPTIX Render & Blender 5.2 VSE Video Muxing (`mux_frames_to_mp4_vse`):**
   - Render on GPU (`Cycles OPTIX` with adaptive sampling and denoising).
   - For 3D animated videos, mux rendered frames and audio in Blender 5.2 VSE (`sc.render.image_settings.media_type = "VIDEO"`, `format = "MPEG4"`, `codec = "H264"`, `audio_codec = "AAC"`).

---

## 3. Module Files (`imagine_to_3d/`)

- **[PLAYBOOK.md](./PLAYBOOK.md)** — Universal pure-imagination `Imagine-to-3D` & `Imagine-to-3D-Video` playbook.
- **[imagine_3d_core.py](./imagine_3d_core.py)** — Template-free, zero-hardcoding Blender 5.2 LTS 3D engine (`configure_scene_from_spec`, `configure_lighting_from_spec`, `build_lofted_cross_section_mesh`, `build_parametric_3d_surface`, `build_custom_bmesh_assembly`, `build_custom_pbr_or_sss_material`, `build_custom_3d_eyeballs`, `build_native_curves_groom`, `build_custom_armature_from_spec`, `set_bone_world_contact_pose`, `build_imagined_human_3d`, `set_finger_cup_pose`, `set_hand_grip_contact`, `mux_frames_to_mp4_vse`, `assert_quality_iteration_gate`).
- **[imagine_vision_solver.py](./imagine_vision_solver.py)** — Standalone vision, depth, silhouette & 16-bit multi-band surface relief solver (`solvePnP`, 3D RBF Thin-Plate Spline, polar contour LUTs, and 16-bit relief extraction; optional weights directory configurable via `VISION_MODELS_DIR`).
- **[imagine_portrait_object_core.py](./imagine_portrait_object_core.py)** — Universal Blender 5.2 portrait & arbitrary object reconstruction core (open-source MPFB2 anatomical quad basemesh + `RANDOM_WALK_SKIN` + 16-bit wrinkle bump + `CATMULL_ROM` hair curves + tailored collar bust + `lathe`/`bilateral_relief` objects).
