# Universal Reference-to-3D Likeness, Facial Structure & Exact Color Calibration Playbook

> **Goal & Zero-Rush Standard:** Achieve a true **1:1 identical likeness** in **shape, focal features (eyes, nose, ears, mouth, hair), AND exact color/skin tone** from ANY 2D reference image — across **Humans, Stylized Characters, Animals, and Hard-Surface Objects**. Never rush or stop while any feature, ratio, plane, or color differs from the reference; iterate until the 3D render matches identically.

---

## 1. Why Models Fail 1:1 Likeness & Exact Color (Root Causes)

1. **Lens / Distance Mismatch:** Evaluating a `35 mm` close-up portrait through Blender's `50 mm` or Orthographic view causes `+14.2%` nose width drift, `-9.1%` ear height compression, and flat cheeks. Match focal length (`lens_mm`) + camera elevation/distance **first**.
2. **Skipping Cranial / Mass Ratios:** Treating a baby/chibi (`Face = 1/3 of skull, Head = 1/2 body height`) with adult ratios (`Face = 1/2 skull`) makes stylized faces immediately look wrong ("uncanny older kid").
3. **Flattening Structural Facial Planes:** Smoothed sphere heads erase the brow shelf, zygomatic cheek platform, lower-eyelid cheek cushion, and receding jaw.
4. **Separate Stick Appendages:** Separate spheres/cylinders for fingers or boxy ears create dark pinch crevices in close-ups; use topologically bridged sockets (`100% quad manifold`).
5. **The `AgX` Display-Referred RGB Trap (Why Skin Color Changes!):**
   - Sampling an `(R, G, B)` pixel from a JPG/PNG and pasting it directly into Principled `Base Color` fails under Cycles `AgX`:
     1. A PNG pixel is **display-referred sRGB** (already lit + shadowed + tone-mapped by a camera/renderer), NOT scene-linear diffuse albedo.
     2. **`AgX`** compresses `16.5` stops of scene-referred dynamic range into SDR and **desaturates bright colors toward white** when `Base Color` luminance > `~0.65` or lighting exceeds midtones.
     3. **`Random Walk (Skin)`** scatters red wavelength deeper (`R=3.67mm, G=1.37mm, B=0.68mm`), pushing bright/saturated peach albedos toward **neon orange/red** on thin cartilage (ears, nose wings, fingertips) and deepening shadows.

---

## 2. Universal Optical & Structural Matching Pipeline (Any Subject)

### Step A — Reference Camera Calibration
| Reference Shot Type | Typical `lens_mm` | Perspective Characteristic |
|---|---|---|
| Tight selfie / chibi dynamic close-up | `28–38 mm` | Larger foreground nose/cheeks, ears tucked further back |
| Standard stylized character turnaround / hero shot | `55–75 mm` | Balanced cranium-to-ear visibility, minimal nose distortion |
| Studio portrait / telephoto product photo | `85–135 mm` | Compressed depth planes, full ear silhouette, slender jaw |

- Set up one locked Blender camera (`MatchCam`) matching estimated `lens_mm`, elevation, and azimuth, centered on the subject's mid-height pivot.
- Enable **Reference Overlay** via `compare_ref.py` (`[Ref] | [Matched Render] | [50% Alpha Overlay] | [Black Silhouette]`).

### Step B — Structural Hierarchy by Category

#### 1. Humans & Stylized Characters
- **Cranium vs. Facial Ratios:**
  - **Baby / Toddler:** Face occupies bottom **`1/3`** of skull height; total head ≈ **`45–50%`** of full body height; eyes sit well below vertical head midline (`~42%` up from chin); soft double-cushion jaw with submental baby fat; wide inter-ocular gap (`≥ 1.1` eye width).
  - **Kid / Tween Stylized (`6–12 yrs`):** Face occupies bottom **`~42%`** of skull; head ≈ **`28–33%`** of body; pronounced rounded apple cheeks + lower-eyelid shelf + tapered soft V/U chin.
  - **Realistic Adult:** Face occupies bottom **`~50%`** of skull; head ≈ **`12.5–13.5%`** (`1/7.5–1/8`) of height; defined zygomatic arch, gonial jaw corner, nasolabial plane break.
- **Five-Feature 1:1 Precision Blueprint:**
  1. **Eyes & Orbital Socket (`5-Part Stack`):**
     - Do *not* bury stylized eyes deeply beneath harsh brow overhangs; keep a soft curved socket cushion below the brow and a lifted lower-eyelid cheek shelf kissing the lower iris rim.
     - Build **Outer Glossy Eye + Thickened Upper Eyelid/Lash Shelf** (casts a crisp real contact shadow across the upper `10–15%` of sclera/iris — kills the glassy "doll stare") + **Inner Caruncle** + **Concave/Shaded Iris with Dark Limbal Ring** (`Coat Weight = 1.0, Roughness = 0.02–0.03`) lit by a circular **`DISK` key light** for a clean dot catchlight.
  2. **Nose (Always 3 Distinct Planes):**
     - Never just a Gaussian bump. Sculpt: (a) narrow **dorsum bridge** rising cleanly between inner eye corners, (b) rounded/bulbous or sculpted **tip dome**, (c) distinct **alar nostril wings** + gentle **philtrum** column down to upper lip center.
  3. **Ears (Never Flat Pegs):**
     - Tilt **`14°–18°` backward** in profile and splay **`22°–35°` outward** toward camera so front and 3/4 views clearly read the curved silhouette.
     - Model a **`~0.60–0.65` width-to-height wedge** with nested loops: mastoid back cup -> **rolled C-shaped outer helix** -> **scapha shadow valley** -> **broad antihelix ridge** -> **deep recessed concha bowl** -> **tragus knob** -> soft plump **lobule**.
  4. **Mouth & Lips:**
     - Closed smile: model upper lip slight overhang (`1–2 mm` in front of lower lip), clean sharp corner dimple pockets (**modiolus** tucked gently inward into apple cheeks), and subtle chin-labial dip beneath lower lip (`never` a sharp dark crevice ring).
     - Open smile: map lip perimeter along continuous spline `Z_upper(x)` and `Z_lower(x)` (avoid staircasing base quad grids!), keep crisp curved upper teeth flush inside upper lip wall, warm maroon interior `(0.14, 0.03, 0.03)`.
  5. **Hair Architecture (Match the Reference's Style Tier!):**
     - *Sculpted Cartoon / Stylized Clay Clumps:* Subsurfed solid sculpted swoops/ribbons tapering smoothly into the scalp (`Anisotropic = 0.55`, soft sheen).
     - *Tousled Clumps + Strands:* Full contoured base hair cap (including temple/sideburn transitions!) + S-curved overlapping hair curves (`Principled Hair BSDF`).

#### 2. Animals (Quadrupeds, Birds, Stylized Critters)
- Measure **Cranium-to-Muzzle Ratio** (`longhound 1:1` vs `kitten/pug 2.5:1`), **Eye Axis Divergence** (predator front-facing vs prey lateral), **Ear Pinna Funnel Flare**, and **Leg Kinematics** (plantigrade / digitigrade hock Z-zigzag / unguligrade cannon bone).
- Match key material pairs: **Wet Rhinarium Nose Pad** (`Voronoi` bump + `Roughness = 0.10–0.18, Coat Weight = 0.70`), **Glossy Tapetum/Dome Eye**, and **Root-Darkened Fur Curve Shader** with soft ear translucency (`SSS Scale = 0.005 m`).

#### 3. Objects & Hard-Surface Props
- Extract exact `L : W : H` bounding ratios + negative-space ratios first.
- Match **Highlight Roll-off width** cleanly to **Bevel radius** (tight specular line = crisp narrow bevel `0.5–1 mm` metallic/acrylic; soft wide highlight = broad fillet `4–8 mm` molded plastic/ceramic).
- Never use `Base Color > 0.85` on white plastic or `< 0.04` on black rubber (`keeps AgX shading richness`).

---

## 3. Exact Color & `AgX` Skin Calibration Science (Cycles Blender 4.x / 5.x)

### Recommended Cycles `Principled BSDF` `Random Walk (Skin)` Recipe
1. **Real-World Mesh Scale:** Apply Scale (`Ctrl+A`). Keep closed manifold skin geometry at real meters (`adult head ~0.23 m`, `baby head ~0.16–0.18 m`).
2. **Core Shader Settings:**
   - `Subsurface Method`: **`RANDOM_WALK_SKIN`**
   - `Subsurface Weight`: **`1.0`** (or masked slightly lower over bony temple/forehead `0.80` vs ears/cheeks `1.0`)
   - `Subsurface Scale`: **`0.0025` to `0.0045` m** (`2.5–4.5 mm`; use smaller values like `0.0020–0.0030` on baby characters so fingers/ears don't turn gummy red!)
   - `Subsurface Radius`: **`(1.0, 0.37, 0.18)`** (physical epidermis R/G/B mean free path ratio)
   - `Subsurface IOR`: **`1.40`**
   - `Subsurface Anisotropy`: **`0.78–0.80`**
   - `Specular IOR Level`: **`0.34–0.42`** (white dielectric Fresnel)
   - `Roughness`: **T-zone (forehead, nose bridge/tip, chin): `0.32–0.38`** | **Cheeks/jaw/body: `0.45–0.52`**
3. **Three-Zone Skin Color Map (Never Flat One-Color Skin!):**
   - **Zone 1 — Golden / Warm Peach Base (Forehead, Nose Bridge, Chin):** Clean neutral-warm peach (`linear RGB value ~ 0.45–0.62`, **never `>0.75`**).
   - **Zone 2 — Vascular Rosy-Coral Blush (Cheek Apples, Nose Tip, Ears, Lips, Elbows/Knees):** Warmer, slightly deeper coral-peach hue shift (`hue -4°..-8°, sat +25%..+45%, value * 0.78`).
   - **Zone 3 — Cool / Soft Shadow Crease (Under-Eye Hollow, Lower Chin Undertone, Neck):** Slightly lower saturation & subtle cool balance to prevent muddy dark-brown crevices (`keep AO multiply color warm terracotta-rose (0.84, 0.62, 0.54)`, NEVER near-black!).

### Closed-Loop Color Calibration Protocol (`color_calibrator.py`)
1. **Diffuse Midtone Sampling (Critical Trap Guard!):** Never pick a specular glare pixel (where `G ≈ R` near white) as your `Base Color` target — sample **clean diffuse midtone skin** (e.g., shaded/mid forehead or chin where true `R > G > B` chromaticity is intact). Sampling a blown highlight causes the solver to push `G ≈ R`, turning peach skin olive/lemon-yellow!
2. **Neutral Studio Baseline + Exposure Headroom (`6500 K` White Lights):** Calibrate scene exposure/key light FIRST using `False Color` (so a neutral `0.78` white card hits `[240, 240, 240]` and midtone skin sits inside **`22%–55%` luminance**) *before* solving albedo. Trying to brighten a dim studio by pushing linear `Base Color > 0.72` triggers AgX highlight desaturation and muddies hue!
3. **Closed-Loop Render-vs-Reference Convergence:**
   - Sample exact diffuse `sRGB` target colors from reference (`Midtone Base`, `Rosy Blush`, `Hair`, `Clothing/Paint`).
   - Convert reference `sRGB -> Linear` (`^2.2`), attenuate slightly (`* 0.72..0.80`) for initial albedo guess.
   - Render a fast **`64x64` OPTIX diagnostic patch**, measure actual rendered `sRGB`, compute `ΔE_rgb` error against reference target, and automatically adjust **Linear Albedo Gain & Chromaticity** across `3–4` iterations until **measured rendered `sRGB` matches reference pixels within `ΔE < 4`**!
   - For stylized / toy characters (stylized cartoon, clay, vinyl), keep `Subsurface Weight ~ 0.45–0.65` with `Scale = 0.002 m` so tiny nose tips & smile creases retain crisp 3D shading instead of washing out like translucent wax.
