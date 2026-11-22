# Manual Reference-to-3D Modeling Playbook (Any Character, Animal, or Object)

> **STRICT READ-ONLY RULE FOR AI AGENTS (DO NOT MODIFY ANY `.py` OR `.md` FILES HERE):**
> - **NEVER EDIT, OVERWRITE, REFACTOR, APPEND TO, OR DELETE** any Python (`.py`) module or Playbook (`.md`) file inside `reference_to_3d/`.
> - Treat every `.py` file in this directory (`exact_ref_core.py`, `vision_solver.py`, `color_calibrator.py`, `skin_calibrator.py`, `compare_ref.py`) as a **strictly read-only library**.
> - **How AI agents must use this module:** Read these Playbooks, add `reference_to_3d/` to `sys.path`, and **import/call** their functions from your own separate external project script. If you need custom helpers or project-specific logic, write them in your own external project file — **NEVER modify these core library files under any circumstances**.

> **NON-NEGOTIABLE QUALITY MANDATE ("ZERO RUSH — ITERATE UNTIL 1:1 IDENTICAL MATCH"):**
> - **DO NOT RUSH:** Never hurry through blocking, sculpting, texturing, or verification, and never stop early because a model looks "roughly similar."
> - **WORK UNTIL IT LOOKS IDENTICAL (1:1 EXACT MATCH):** Keep measuring, adjusting, and re-rendering the 3D asset against the reference image across as many iterations as required **until the silhouette, proportions, depth, features, folds, colors (`ΔE < 3.5`), and specular highlights match the reference identically**.
> - **NEVER ADVANCE OR SHIP WITH DEFECTS:** Do not declare completion, move to the next asset, or start animation/final rendering while ANY proportion drift, silhouette gap, flat facial plane, mesh overlap, floating contact, or color mismatch remains.

> **Core Method:** Pure proportional observation, primary-secondary-tertiary blocking, closed-loop visual comparison, and camera-calibrated 3D reconstruction in **Blender 5.2 LTS** (using pure native `bmesh`/Curves for stylized characters, animals, and objects, and the optional free open-source **MPFB2** Blender Extension via `exact_ref_core.py` + `vision_solver.py` for 1:1 realistic human portraits and full-body characters).

---

## Phase 1: Reference Dissection (Before Creating Any Vertex)

1. **Identify Camera Perspective & Focal Length:**
   - **Orthographic Turnaround (`Front / Side / Back`):** Establish shared horizontal landmark lines (ground `z=0`, ankles/knees, hip/crotch, waist, shoulder, chin, brow, crown). Cross-check height alignments between Front & Side first.
   - **Single Perspective / 3/4 Photo:** Match **focal length + camera elevation/azimuth** before evaluating proportions:
     - Telephoto / character portrait: `85–135 mm` (minimal perspective foreshortening).
     - Standard 3/4 product / animal photo: `45–60 mm`.
     - Dynamic wide-angle / close shot: `24–35 mm` (strong foreground exaggeration).
     - Set **one Blender camera per reference photo** orbiting a shared pivot (`(0, 0, mid_z)`).

2. **Extract Proportional Unit Rulers & Ratios:**
   - **Humans / Stylized Characters:** Measure height in **head-height units** (`~7.5–8` heads realistic; `~3–6` stylized), shoulder-to-hip width ratio, torso-to-leg split (`crotch ~ 0.46–0.50` H realistic; shorter legs in chibi/stylized), arm span, and facial thirds (`hairline -> brow -> nose base -> chin`).
   - **Animals / Quadrupeds:** Measure torso length in **ribcage/head units**, ground-to-belly negative space vs. leg length, shoulder-to-pelvis spine slope, cranium vs. muzzle ratio, and stance kinematics:
     - *Plantigrade* (bear, human): full sole on floor.
     - *Digitigrade* (dog, cat, bird): walks on toes; raised **ankle/hock** (`tarsals`) looks like a backward joint while true knee (**stifle**) sits high near belly; hind leg forms a distinct **Z-zigzag** (`femur -> tibia -> metatarsal`).
     - *Unguligrade* (horse, deer): walks on hoof tip with long vertical cannon bone.
   - **Objects / Props:** Extract outer `L : W : H` bounding box ratio, symmetry axes (radial / bilateral), and sub-part fractions (handle gap, rim thickness, collar step, wall thickness).

3. **Infer Missing 3D Depth (`Y` / Back-to-Front) — Avoiding the 2D Pancake Trap:**
   - **Ground cast shadows & contact footprint** reveal true stance width & forward lean.
   - **Ellipse foreshortening** on circular collars, eyes, wheels, rims, cuffs (`minor / major axis ratio = cos(tilt)`).
   - **Occlusion / T-junctions** show which surface wraps cleanly in front of or behind another.
   - **Cross-sectional barrels/ovoids:** Every torso, head, skull, ribcage, and limb has true `X` and `Y` depth — never flatten faces or chests to a front plane.

4. **Read Shape Language, Negative Space & Material Cues:**
   - Contrast **load-bearing Straights** against **fleshy/dynamic C- and S-curves**; enforce **Big / Medium / Small** mass hierarchy and keep visual rest areas uncluttered.
   - Read **Negative Space** gaps (armpit triangle, belly-to-ground arch, leg gap, inside mug/trigger handles) equally with positive geometry.
   - Read **Bevel Sharpness from Lighting Highlights**: crisp tight specular highlights = narrow bevel (metal/hard plastic); broad soft specular roll-off = wider bevel / organic smooth surface (wood, rubber, clay, fur silhouette).

---

## Phase 2: Primary -> Secondary -> Tertiary Blocking (Never Skip Stages)

1. **Primary Blockout (80% of Likeness — Lock Low-Poly Silhouette First):**
   - **Characters:** Cranium ovoid + jaw wedge, tapered ribcage box, pelvis block, mirrored `A-pose` cylinders/3-loop joint limbs, hand palm slab + 4 fingers + thumb. Keep faces planar (recessed sockets behind brow, projecting nasal wedge, curved mouth barrel — never a flat plane).
   - **Animals:** Build internal construction first: skull + muzzle wedge -> S-curved neck cylinder -> deep ribcage egg -> abdominal pinch -> sloped pelvis -> **independent scapular triangle** sliding over front ribs -> Z-zigzag hind leg -> flowing tail curve.
   - **Props / Hard Surface:** Combine clean primitive silhouettes (`Cube`, `Cylinder`, `Spin/Screw` lathe for radial parts, `Curve` bevel tubes for hoses/cables, `Solidify` for shell/rim thickness) with separate Material IDs per part early.
   - **Gate:** Do not add fine details until the primary silhouette overlay matches the reference 1:1.

2. **Secondary Forms & Structural Plane Breaks:**
   - Carve cheek/temple transitions, brow ridge angle, jawline gonial corner, collar hems, paw/knuckle pads, chamfered housing seams, and **chunked curve ribbons** for stylized hair or fur clumps.
   - Apply **early base colors / roughness contrast / Vertex Paint** so color-zone proportions and light-vs-dark trim strips are judged accurately alongside silhouette.

3. **Tertiary Polish (Only After Primary + Secondary Silhouettes Match Identically):**
   - Edge bevel weights matched to material hardness, recessed trim seams, micro hardware/bolts, clean pole placement onto flat areas (never across curved highlights or joint creases).

---

## Phase 3: The Zero-Rush Exact-Match Verification Loop (Repeat Until Identical)

Never stop after one render. Repeat Steps 1–4 in a strict closed loop until the 3D render is indistinguishable from the reference in shape, proportion, depth, and color:

1. **Matched-Camera Overlay & Numeric Gate (`compare_ref.py --strict`):**
   - Render the model from the exact estimated camera angle/focal length of the reference photo and generate `[Reference] | [Matched Render] | [50% Alpha Overlay] | [Silhouette]`.
   - Inspect every contour, landmark, head scale, muzzle tilt, limb stance, and garment hem. If `compare_ref.py` reports `ITERATE_REQUIRED` or if visual inspection reveals ANY mismatch, refine the 3D model and re-render.
2. **Flat Black Silhouette & Negative-Space Check:**
   - Verify outer contour and internal negative-space cutouts with unlit/flat silhouette comparison against reference edges until silhouette `IoU` and visual alignment lock.
3. **Anti-Pancake Multi-Angle Audit (`Front`, `3/4`, `Side +X`, `Back`, `Top`):**
   - Confirm cross-sectional `X/Y/Z` depth, cranial/facial protrusion, chest-ribcage depth, clean attachments (zero floating limbs or seams), and foot/wheel `z=0.000` grounding from unseen angles.
4. **Highlight, Grayscale Value & Closed-Loop `AgX` Color Check (`color_calibrator.py` + `skin_calibrator.py`):**
   - Compare specular edge sharpness and thumbnail/grayscale value separation.
   - **Never copy sRGB JPG pixels directly into `Base Color`!** Use `FACE_AND_COLOR_PLAYBOOK.md`, `SKIN_PLAYBOOK.md`, `color_calibrator.py` (`calibrate_color_under_agx(..., require_exact=True)`), and `skin_calibrator.py` to convert display-referred sRGB swatches into calibrated scene-linear albedo (`Random Walk (Skin)` `Scale=0.002-0.004 m`, `Radius=(1.0, 0.37, 0.18)` + `3-Zone` warm/rosy/cool variation) and iterate until rendered midtone sRGB matches reference swatches within **`ΔE < 3.5`**.

---

## Companion Guides & Tools in This Folder
- **[exact_ref_core.py](./exact_ref_core.py)** — Universal Blender 5.2 LTS 1:1 portrait & arbitrary-object reconstruction core (`build_exact_portrait_from_ref`, `build_exact_object_from_ref`, `clean_and_bake_raw_mesh`, `setup_studio_lighting`, open-source MPFB2 basemesh + 3D RBF landmark fit + `RANDOM_WALK_SKIN` + 16-bit wrinkle bump + `CATMULL_ROM` hair/mustache curves + 3D tailored collar/coat bust).
- **[vision_solver.py](./vision_solver.py)** — Computer-vision & 16-bit surface relief solver (`solvePnP` focal-length sweep, 478-point facial landmark fitting, macro illumination de-lighting, 4-band 16-bit relief heightmaps, and 360° polar contour LUTs; optional model path configurable via `VISION_MODELS_DIR`).
- **[FACE_AND_COLOR_PLAYBOOK.md](./FACE_AND_COLOR_PLAYBOOK.md)** — Deep guide on 1:1 likeness across **Characters, Babies, Animals, and Props/Objects** + `AgX` / `Random Walk Skin` color matching science.
- **[SKIN_PLAYBOOK.md](./SKIN_PLAYBOOK.md)** — Biophysical chromophore skin lookdev bible, Fitzpatrick I–VI calibrated shader recipes, 6-zone facial color map, and dual-lobe sebum/pore stack.
- **[CLOTHING_PLAYBOOK.md](./CLOTHING_PLAYBOOK.md)** — Universal Realistic & Stylized 3D Clothing guide (hoodies, cargo pants, shirts, coats, skirts/dresses, `Cloth` sewing sim, 7 fold archetypes, cuffs/thickness, rigging loops & Cycles `AgX` fabric shaders).
- **[color_calibrator.py](./color_calibrator.py)** — Universal sRGB reference sampler & automated Cycles `AgX` closed-loop albedo solver.
- **[skin_calibrator.py](./skin_calibrator.py)** — Reference-photo skin de-lighter, Fitzpatrick/ITA profiler, and Blender 5.2 Cycles `RANDOM_WALK_SKIN` shader builder.
- **[compare_ref.py](./compare_ref.py)** — Multi-panel Reference vs Matched Render + 50% Overlay + Black Silhouette + Turnaround compositor with strict exact-match metrics (`IoU` & `MAE`).
