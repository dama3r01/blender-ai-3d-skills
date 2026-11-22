# Universal 3D Clothing Playbook — Stylized & Realistic Garments (Blender 4.x / 5.x Cycles AgX)

> **Scope & Zero-Rush Standard:** Covers **every garment category** — Pullover Hoodies, T-Shirts, Collared Shirts, Coats/Jackets, Jeans/Trousers/Cargo Pants, Skirts/Dresses, Capes, Shoes, and Accessories — in both **Cartoon / Stylized** and **Realistic** styles. Never rush garment silhouette, cuff thickness, fold tension, or fabric shading — keep refining until every seam, drape, and highlight matches the target design identically with zero body clipping.

---

## 1. Universal Garment Anatomy & Design Rules (Both Stylized & Realistic)

### A. Never Trace Naked Anatomy — Bridge Hollows with Air Gap
- Extract base garment shells (`Edit` duplicate + `Alt+S` inflate, or draft `2D` panels), keeping a real **air gap** (`3–12 mm` realistic; `8–20 mm` chunky stylized) over torso, clavicles, ribs, armpits, and groin.
- **Bridge anatomical valleys:** Clothes span *across* sternum hollows, rib dips, spinal grooves, and the groin valley — never shrinkwrap fabric skin-tight into crevices.
- **Model in Relaxed A-Pose (`35°–45°` arm slope):** Avoid `T-pose` (stretches underarm excessively when arms drop) and avoid arms pressed down at hips (collapses armpit scye weights).

### B. Garment Part Structural Blueprints
| Garment Type | Key Pattern / Mesh Features for 1:1 Believability |
|---|---|
| **Torso Bodice (Tees / Shirts / Hoodies)** | **Front neckline scooped deeper** (`6–7.5 cm` real / scaled stylized) than back (`2.2 cm`); **front armhole (scye) scooped deeper** into pec line than flatter back scye. Never drop scye too low toward waist—widen sideways instead. |
| **Sleeves (`Set-In` vs `Raglan`)** | *Set-In:* Asymmetric S-curve sleeve cap (`~2%` ease above scye perimeter), fuller rounded deltoid cap, scooped deeper in front. *Raglan:* Diagonal seam running from underarm straight to collar/neck for sports hoodies & active jackets. Long sleeves drafted ~`5–8%` longer than arm length for natural wrist/elbow stacking. |
| **Collars & 2-Panel Hoods** | *Collars:* Separate **collar stand band** wrapping the neck + rolled outer **collar leaf** (slightly longer outer rim so it rolls smoothly). *Hoods:* Inner lining + outer shell (`extruded/solidified` thickness), rotated/draped naturally backward onto upper back & clavicles with `2–3` broad U-shaped **Diaper folds** + an inner **crew-neck T-shirt ring** visible at the throat. |
| **Hoodie Pocket, Hardware & Cuffs** | Separate beveled **Kangaroo Pocket** patch on lower-front torso (`I` inset perimeter lip + angled hand-entry cuffs + double topstitch groove); rigid metal/plastic **eyelets/grommets** attached via **3-Vertex Parenting** or `Surface Deform`; Bezier **drawstrings** ending in thicker cylindrical **aglets**; inset **ribbed cuffs & waistband** (`~82–88%` sleeve width) with chunky gather folds immediately *above* the band. |
| **Trousers & Cargo Pants** | Asymmetric **back rise / seat curve** (longer & deeper than front rise); **4-sided/diamond crotch gusset** spanning front-to-back seam (keep `5-edge` poles *away* from the inner-thigh crease!); slanted hip pockets + **J-stitch fly overlap**; boxy **side cargo bellows pockets** (gusseted pleats + overlapping flap + snap) kept on the outer-thigh rest area *above* the knee; rolled **double-thick ankle cuffs**. |
| **Open Jackets / Coats** | Center-front button-stand overlap (`~2 cm`), upper **folded lapels** diverging smoothly from chest, lower back/front **vent**, optional belt/drawcord waist gather (`pipe` + `zigzag` pleats). |
| **Skirts, Dresses & Capes** | Fitted bodice/waistband flaring smoothly to `1.4x–2.2x` hip hem; vertical **Pipe folds** hanging around circumference (`even/odd rhythm`), or **Drop folds** fanning from pinned shoulder brooches/clasps on capes. |

### C. Edge Thickness Without Paper Cards or Clipping
1. **Physics First (`Cloth Sim`):** Always simulate a **clean, single-wall quad mesh** (`Modifier order: ARMATURE -> CLOTH -> SUBSURF -> SOLIDIFY`). Simulating double-walled thickness ruins drape and spikes corners.
2. **`Solidify` (`Simple` vs `Complex`):** Keep `use_even_offset = False` (or tight clamp) on sharply folded hems so corners do not spike; enable `High Quality Normals` + `Fill Rim`, plus rim bevel/crease loops so `Subsurf` catches a crisp textile edge highlight.
3. **Rigged / Hero Character Cuff Method (Best for Animation):** On wrists, collars, pockets, and ankle hems, **extrude only the boundary loop inward (`2–5 mm` radial wall -> `6–12 mm` inner tuck) + bevel the outer edge**. Gives rich shadow-catching rim thickness while eliminating inner-shell penetration inside elbows/armpits.

---

## 2. Fold & Wrinkle Mechanics (Anatomical Tension + Stylized Rest Areas)

Every real or sculpted fold has **4 anatomical parts**: *Peak -> Ridge (lit wall) -> Valley (dark crease) -> Eye (fading termination)*. **Rule:** Build positive convex fold volume first (*Inflate/Clay*) before sharpening the valley (*Crease*) so folds never look engraved into clay.

### The 7 Fold Archetypes & Zone Recipes
1. **Pipe:** Parallel vertical tubes from a belt, waistband, or flared skirt hem.
2. **Diaper:** Catenary U-arcs slung between two anchors (hood drape across clavicles, draped cowl neck, relaxed seat).
3. **Drop:** Fanning diagonals from a single elevated hook/shoulder when lifting an arm or carrying a shoulder bag.
4. **Half-Lock:** Hooked Z-pinch formed when a cylindrical tube bends sharply (**inner elbow**, **popliteal back of knee**, hip crease).
5. **Zigzag:** Interlocking asymmetric "drunk triangles/diamonds" under vertical compression (**above hoodie waistband**, **above sleeve rib cuffs**, **stacked pant/jean ankle pools**).
6. **Spiral:** Diagonal twisting folds rotating around turned forearms or striding trouser legs.
7. **Inert:** Collapsed organic pooling where long hems/capes hit the floor.

### Stylized vs. Realistic Discipline
- **Visual Rest Areas (Critical!):** Tension stretches cloth smooth across the **pec/chest**, **upper back**, **front kneecap**, and **thigh**. Leave `60–70%` of these open panels smooth and **cluster 2–4 crisp Big/Medium folds only at tension/compression anchors** (armpit scye, inner elbow, above ribbed cuffs/hem, crotch X-tension, behind knee, above shoe collar). Break L/R symmetry deliberately!
- **Memory Folds vs. Live Sim / Corrective Shapes:**
  - *Bake/Model in Rest Pose:* Seam pucker, welt pocket bevels, permanent denim ankle stacking, pressed center trouser crease, collar roll.
  - *Dynamic Bend:* Keep deep bend creases subtle at rest (`<3–6 mm`) and drive heavy half-locks with **`Smooth Corrective`** (`below Armature`) + angle-driven **Corrective Shape Keys** (or live `Cloth` on low-res proxy cage + `Surface Deform` on detailed garments).

---

## 3. Animation-Ready Garment Topology
- **Joint Loop Budgets:** **6–8 loops** around hero character elbows & knees (`4–6` mobile/game). Use **diamond / V-notch topology** on outer elbow & kneecap (expands loops over extension side, collapses cleanly on inner fold) and keep **all `5`-edge poles off the bend line**.
- **Armhole (`Scye`) Flow:** Route continuous edge loops radially from chest/scapula cleanly into the sleeve cuff; add `2–3` extra horizontal gather loops directly above the ribbed hoodie waist hem for spine twists.
- **Weight Transfer Armpit Fix:** After running **`Data Transfer`** (Vertex Groups -> Nearest Face/Vertex) from body to clothes, **Apply it in Rest Pose**, then open **Edit Mode** and strip stray upper-arm/chest weights across the armpit gap before adding a masked **`Smooth Corrective`** (`Rest Source: Bind`, `Factor ~ 0.6`, `Repeat 6`).

---

## 4. Cycles `Principled BSDF` (`AgX`) Fabric Material Library

> **Golden Rule for `AgX` Colors:** Never leave OpenPBR `Sheen Tint` at pure white `(1,1,1)` over saturated cotton/fleece/twill — white microfiber fuzz washes vibrant colors into dusty gray! **Match `Sheen Tint` to a bright, slightly desaturated/warmed tint of the garment `Base Color`** (`Sheen Roughness = 0.45–0.62`, `Sheen Weight = 0.35–0.65`).

| Fabric | `Principled BSDF` Core Values | Micro-Texture & Secondary Detail Recipe |
|---|---|---|
| **Brushed Cotton Fleece (Hoodie)** | `Roughness 0.80–0.90`, `Specular IOR Level 0.25–0.35`, `Sheen Weight 0.45–0.65`, `Sheen Roughness 0.50`, **Blue-tinted `Sheen Tint`** | Dual-scale knit/fleece: medium `Wave/Brick` knit grain + high-freq soft `Noise` fuzz bump (`Distance = 0.001 m, Strength = 0.20–0.32`) + Cycles `Ambient Occlusion (Only Local, Dist = 0.04 m)` darkening seams/folds + subtle **`Pointiness` crest lift** on fold ridges (`+6%` value). |
| **Ribbed Knit Cuffs / Waistband** | `Roughness 0.84–0.92`, `Sheen Weight 0.60`, `Sheen Roughness 0.45`, color-matched `Sheen Tint` | Vertical wale grooves (`Wave Bands X` along cuff UVs -> `Bump Distance = 0.0018 m, Strength = 0.55`) + alternating knit-valley darkening (`0.88x` in troughs) + slight horizontal contrast groove (`Inset` seam) separating band from sleeve. |
| **Twill / Chino Cargo / Denim** | `Roughness 0.68–0.82`, `Specular IOR Level 0.32`, `Sheen Weight 0.20–0.35`, `Sheen Roughness 0.55`, tan/indigo `Sheen Tint` | **Diagonal `2x1`/`3x1` Twill Math (`U + V` phase `Wave` at `45°` + micro cross-thread `Noise`)** -> `Bump (Distance = 0.0008 m, Strength = 0.32)`; for denim, drive `Mix Color` between indigo warp and off-white ecru weft (`#E5E0D5`) + worn `Pointiness` knee fade. |
| **Silk / Satin** | `Roughness 0.18–0.32`, `Specular IOR Level 0.58`, **`Anisotropy 0.70–0.85`**, `Sheen Weight 0.08` | Plug `Tangent Node (UV Map)` into `Tangent` socket; note Blender 4+/5.x Principled anisotropy is rotated `90°` vs legacy Glossy (`+0.25` rotation for weft direction). |
| **Heavy Wool / Sweater Knit** | `Roughness 0.85–0.95`, `Specular IOR Level 0.22`, `Sheen Weight 0.85–1.0`, `Sheen Roughness 0.38` | Chevron V-stitch knit loop UV displacement/bump + `Layer Weight (Facing)` fuzzy edge halo. |
| **Sneaker Canvas vs. Pebbled Leather + Vulcanized Rubber** | Canvas/Suede: `Roughness 0.78–0.88`, `Sheen 0.30`; Leather: `Roughness 0.34–0.44`, `Coat Weight 0.35`; Rubber Foxing: `Roughness 0.32–0.42`, `Sheen 0.0` | Separate materials (`Roughness Contrast` is key under `AgX`!): **`Voronoi F1` (`Randomness=0.90` + flattened B-Spline Ramp)** pebbled leather grain (`Bump Distance = 0.0006 m`) vs matte canvas knit vs satin nylon laces (`Roughness 0.45`). |
| **Topstitching & Seam Puckers** | Contrasting slightly lighter/brighter thread (`Value * 1.15`, `Roughness 0.48`, `Specular 0.45`, `Sheen 0.15`) | Procedural UV/Object **double-needle dashed stitch pills** along pocket/fly/flap seam borders (`Brick/Wave` dashes) raising `Bump` + lowering thread roughness, or real `Curve` stitch arrays for close-ups. |
