# Universal Reference-to-3D Skin Research & Lookdev Bible (Blender 5.2 Cycles + AgX)

> **Purpose & Zero-Rush Standard:** Deep, generalizable skin lookdev doctrine for extracting **true diffuse albedo, undertones, facial color zones, specular sebum maps, micro-pores, and subsurface scattering depth** from **ANY** reference image (fair ivory, warm golden, olive, South Asian brown, deep ebony Black skin, stylized animation, or cross-polarized/uncontrolled photos) without plastic waxiness, orange SSS bleed, or ashy grey shadows. Never rush skin calibration — iterate light exposure, albedo, SSS depth, and specular roughness until rendered midtones match the reference identically (`ΔE < 3.5`).

---

## 1. Cycles `RANDOM_WALK_SKIN` Kernel Physics (Blender 4.2 – 5.2 LTS)

Verified directly against Blender Cycles kernel C++ source (`bssrdf.h`, `svm/closure.h`, `integrator/subsurface_random_walk.h`) and probed in installed **Blender 5.2.3 LTS**:

1. **Unified Albedo Input:** Blender 4+ removed `Subsurface Color`; Cycles drives subsurface scattering directly from `Base Color * Subsurface Weight`.
2. **Exact Effective Scatter Radius Formula (`RANDOM_WALK_SKIN`):**
   - Unlike standard `Random Walk` (which reads global `IOR` and surface `Roughness²` for entry refraction) or `Christensen-Burley` (empirical dipole diffusion), **`RANDOM_WALK_SKIN` uniquely:**
     1. Reads dedicated **`Subsurface IOR`** (`default = 1.40`, matching human epidermis).
     2. Uses a **50% cosine-weighted diffuse + 50% rough GGX refractive entry bounce** at **fixed internal entry roughness $\alpha = 1.0$** — preserving fine bump/pore sharpness and eliminating dark grazing terminator artifacts.
     3. Does **NOT** ignore `Subsurface Radius` or `Subsurface Scale`: it multiplies `max(Radius * Scale, 0)` by a channel-wise Newton-solved correction from `Base Color` and `Subsurface IOR`:
        $$\mathbf{r}_{\text{eff}} = \mathbf{r}_{\text{input}} \cdot \text{Scale} \cdot \sqrt{3(1 - \alpha')}$$
        where $\alpha'$ is solved from each RGB channel of `Base Color` via Jensen's diffuse reflectance $R_d$.
     4. Applies **`Subsurface Anisotropy`** (`g = 0.75–0.80` for forward-scattering dermal tissue) with polynomial scattering coefficient fitting $\sigma_t = 1 / (d \cdot (1 - g))$.

### Critical Kernel Consequence for Diverse Skin Tones
- Because darker `Base Color` yields smaller $\alpha'$, $\sqrt{3(1 - \alpha')}$ approaches $\sqrt{3} \approx 1.732$.
- **Result:** If you put a high `Subsurface Scale` (`4–5 mm`) and pale `Radius` (`(1.0, 0.37, 0.18)`) onto dark skin (Types V–VI), Cycles scatters light **too far** relative to true melanin photon absorption, causing **chalky red-orange edge bleed** on lips, nostrils, and ears! Conversely, lowering `Base Color` value while keeping high base `Roughness` and flat specular turns deep skin **ashy / dusty grey** because white Fresnel reflectance ($F_0 \approx 0.028$ at $\text{IOR} = 1.4$) overpowers a low-luminance (`Y ~ 0.05–0.12`) diffuse layer.

---

## 2. Extracting True Albedo from ANY Reference Image (De-Lighting & Constancy)

Never paste a raw JPG/PNG `(R, G, B)` pixel directly into `Base Color`. A portrait image already contains:
$$\text{Photo}_{\text{sRGB}} = \text{ToneMap}\big(\text{Illuminant} \times (\text{Diffuse}(\mathbf{A}) \cdot \text{Shading}(N \cdot L) + \text{SSS} + \text{Specular})\big)$$

### 4-Step Reference De-Lighting Protocol (`skin_calibrator.py`)
1. **Conservative Illuminant White Balance (Color Constancy):**
   - If a photo was shot under warm tungsten (`~3200 K`), golden sunset, or cool blue shade (`~7500 K`), raw sampling bakes the lamp color into skin.
   - Sample a **neutral diffuse/sclera highlight cue** (living eye sclera mid-highlight or neutral white collar/wall — remembering healthy sclera carries subtle capillary warmth and is NOT `#FFFFFF`, so only use mild per-channel gain balancing clamped to `[0.82, 1.22]`).
   - Also validate against **Pointer's Gamut** physical diffuse limits: keep true linear diffuse reflectance strictly within `0.020 – 0.810` (`Macbeth Dark Skin Patch #1` baseline $\approx$ Linear `[0.168, 0.084, 0.058]`, $Y \approx 0.100$).
2. **Band-Pass Trimmed Skin Sampling (Specular + Shadow Rejection):**
   - Convert region pixels to **Linear sRGB** and **HSV**.
   - **Reject Specular Glare:** Drop top **`~15%`** highest-luminance and low-saturation (`V` spike + `S` drop) pixels where dielectric Fresnel $F_0$ reflects the white light source instead of skin pigment.
   - **Reject Cast Shadows & Hairs:** Drop bottom **`~20%`** darkest pixels (`nasolabial creases`, pores, lashes, stubble, under-nose) where long-wavelength SSS red bleed or blue sky fill skews ratios.
   - Compute the **trimmed median / 40th–65th percentile linear RGB** across clean mid-cheek, temple, and forehead samples.
3. **2D Frequency Separation for Projected Facial Albedo Maps:**
   - When extracting an actual texture map from a reference portrait: decompose linearized pixels into **Low-Frequency (`LF`, large Gaussian blur)** and **High-Frequency (`HF = Original - LF`)**.
   - Flatten directional key-to-shadow gradients strictly on `LF` (`Albedolf = LF / smooth_illum_field`), then recombine `HF` freckles, moles, pores, and capillary grain untouched!
4. **Biophysical Chromophore Anchor (Beer-Lambert Model):**
   - True human skin reflectance obeys Beer-Lambert optical absorption across 4 chromophores:
     $$\mathbf{A}_{\text{lin}} = (0.84, 0.66, 0.58) \cdot \exp\!\big(-[M_{\text{eu}}(1.0, 1.22, 1.50) + M_{\text{pheo}}(0.15, 0.45, 1.0) + H_{\text{oxy}}(0.02, 0.55, 0.45) + H_{\text{deoxy}}(0.60, 0.45, 0.25)]\big)$$
   - Checking extracted RGB ratios guarantees skin stays on the real human chromophore manifold (`R > G > B` preserved even at $Y = 0.05$ on Type VI skin!).

---

## 3. Fitzpatrick I–VI Calibrated Shader Recipes (Under `AgX - Medium High Contrast`)

Measured linear benchmarks across Fitzpatrick I–VI skin types:

| Type | Linear sRGB Base (`R, G, B`) | Display Approx (`0–255`) | `SSS Radius (R,G,B)` | `SSS Scale (m)` | Base `Roughness` / Sebum `Coat Weight` | Key Undertone & Lip Rule |
|---|---|---|---|---|---|---|
| **I — Pale Fair / Ivory** | `(0.847, 0.638, 0.552)` *(or `~0.68` headroom under bright studio key)* | `(237, 209, 196)` | `(0.482, 0.169, 0.109)` | `0.0040 – 0.0050` | `0.48` / Coat `0.28` (`rough 0.24`) | Cool rosy pink cheek flush, subtle freckles, lighter coral-pink lips; avoid `V > 0.72` clipping under AgX |
| **II — Light / Fair Peach** | `(0.799, 0.485, 0.347)` *(studio target `~0.58–0.70`)* | `(231, 185, 159)` | `(0.482, 0.169, 0.109)` | `0.0035 – 0.0045` | `0.46` / Coat `0.32` (`rough 0.23`) | Warm peach forehead, strong coral-rose cheeks & ears, delicate capillary warmth |
| **III — Medium / Golden Olive** | `(0.623, 0.433, 0.343)` | `(207, 176, 158)` | `(0.482, 0.169, 0.109)` | `0.0030 – 0.0038` | `0.45` / Coat `0.36` (`rough 0.22`) | Golden/olive forehead (`G/B` ratio up), soft rose-terracotta cheeks, cool olive jaw shadow |
| **IV — Tan / South Asian** | `(0.436, 0.227, 0.131)` | `(176, 131, 101)` | `(0.367, 0.137, 0.068)` | `0.0022 – 0.0030` | `0.43` / Coat `0.42` (`rough 0.21`) | Rich wheat/amber ochre base, warm spiced-rose cheeks, plum-brown upper lip vs warmer peach lower lip |
| **V — Brown / Deep Caramel** | `(0.283, 0.148, 0.079)` | `(145, 107, 80)` | `(0.367, 0.137, 0.068)` | `0.0016 – 0.0024` | `0.40` / Coat `0.48` (`rough 0.19`) | Keep `R/B > 3.2` saturation! Lower SSS depth (`0.5x`), strong warm bronze Coat highlights |
| **VI — Deep Ebony / Rich Black** | `(0.090, 0.050, 0.020)` | `(85, 64, 39)` | `(0.367, 0.137, 0.068)` | `0.0011 – 0.0017` | `0.36 – 0.40` / Coat `0.55` (`rough 0.17`) | **Never desaturate toward grey** (`R/G ~ 1.8, R/B ~ 4.5`)! Low SSS prevents grey/brick-red edges; lower base roughness + richer T-zone sebum `Coat` produces luminous bronze specular instead of chalky ash; unmask lighter peach-pink lower lip & palms |

### Preventing Deep-Skin "Ashy Grey" & Pale-Skin "AgX Whiteout"
- **Why Dark Skin turns Ashy/Grey:** White dielectric Fresnel reflections sit ON TOP of low-luminance diffuse albedo. High matte roughness spreads grey white reflection broadly across cheeks, washing out dark brown chroma. Fix: use **`Roughness = 0.36–0.40` + focused `Coat Weight 0.45–0.55, Coat Roughness 0.17–0.20`** + clean `Cavity` pore damping so highlights stay tight, golden-bronze, and rich.
- **Why Light Skin turns Neon-Orange or Chipped White under `AgX`:** `AgX` compresses highlights gently toward white when linear irradiance exceeds ~1.0, while excessive `Scale (> 0.005 m)` floods nostrils/ears red. Fix: calibrate key light first so neutral `0.78` white card hits `[235–245]`, keep fair diffuse `Base Color` scaled cleanly (`<= 0.72` max channel in heavy keys), and mask SSS depth thinner on brow/chin vs cheeks/ears.

---

## 4. 6-Zone Facial Color Map & Dual-Lobe Micro-Detail Stack

1. **Zonal Color & Mask Map (Apply subtly as glazes, never clown stripes):**
   - **Forehead / Brow:** Warm golden/yellow-ochre shift, **high T-zone `OilMask` (`~1.0`)**, lower bone SSS weight (`0.60–0.75×`).
   - **Mid-Face Cheeks / Nose Bridge & Tip / Ears:** Hemoglobin rose-red flush (`hue -4°..-10°, sat +18%..+35%`), nose very oily (`Coat` high), cheeks broader softness (`Roughness 0.46–0.52`), ears/nostril wings full `1.0` SSS bleed.
   - **Lower Jaw / Chin / Perioral Muzzle:** Slightly cooler olive/slate undertone balancing warm lips; chin center oily (`0.74` mask).
   - **Under-Eye Hollow:** Thin skin (`cool violet-brown` shadow shift in deeper tones, **suppressed SSS weight `0.50`** so lower lids never glow neon orange).
   - **Lips:** Two-tone rule (**darker melanin plum/terracotta upper lip** vs **lighter warmer peach/berry lower lip**), fine face pores masked off (`0.15×`), wetter clear **`Roughness = 0.20–0.30`** + higher **`Coat Weight = 0.65, Coat Roughness = 0.08–0.14`**.
2. **Blender 5.2 Multi-Scale Specular & Micro-Relief Stack (`skin_calibrator.py`):**
   - **Primary Epidermis Lobe:** `IOR = 1.40` (`Specular IOR Level = 0.5 * lerp(1.0, cavity, 0.60)` — pore occlusion kills sparkly specular dots inside pits!), `Roughness = base + cavity_pit * 0.14`.
   - **Secondary Lipid/Sebum Coat Lobe:** `Coat IOR = 1.45`, `Coat Roughness = 0.18–0.24`, `Coat Weight = T_zone_oil * scale`, sharing micro-pore `Normal` into `Coat Normal`.
   - **Vellus "Peach-Fuzz" Rim Lobe:** `Sheen Weight = 0.04–0.07` (photoreal) or `0.15–0.22` (soft stylized rim), `Sheen Roughness = 0.50`, `Sheen Tint = Mix(0.50, White, Base Color)`.
   - **Physical Micro-Relief (`Distance = 0.00025 m`):** Warped `Voronoi F1` pores (`0.9 mm` primary + `0.40 mm` fine secondary) + `Voronoi DISTANCE_TO_EDGE` micro-skin polygons (`0.70 mm` + `0.30 mm`) + `3.5 mm` anisotropic fine lines + `0.40 mm` dermis grain + **`Cavity Darkening (~0.25–0.35)`** multiplying subtly into crevices.

---

## 5. Style Adaptation Matrix: Avoid "Plastic Doll" vs. "Waxy Candle"

| Style Target | `SSS Method` | `SSS Weight` | `SSS Scale` | `Anisotropy` | `Coat / Oil` | `Pore Relief` |
|---|---|---|---|---|---|---|
| **Photoreal Digital Human** | `RANDOM_WALK_SKIN` | `1.0` (`0.55` eye sockets) | `0.0012` (Type VI) – `0.0045` (Type I) | `0.80` | Full T-zone mask (`0.28–0.55`) | Full `0.9/0.4 mm` pores + `0.25 mm` bump + cavity damping |
| **Semi-Real Stylized PBR** | `RANDOM_WALK_SKIN` | `0.75–0.90` | `0.0022–0.0038` | `0.65` | Moderate (`0.20–0.35`) | Softened (`0.12 mm` dermis grain + soft T-zone) |
| **Stylized / Animation** | `RANDOM_WALK_SKIN` | `0.55–0.75` | `0.0020–0.0032` *(prevent tiny nose wax!)* | `0.50` | Clean soft (`0.18–0.25`), wet glossed lips (`0.70`) | Broad sculptural facial zone glaze + high-sheen vellus rim (`0.14–0.22`) |

---

## 6. Closed-Loop Reference Verification (`color_calibrator.py` + `skin_calibrator.py`)

When a reference photo is supplied:
1. Run `profile_reference_portrait()` with normalized face-zone ROIs (neutral cue + forehead + cheeks + jaw + lips) to obtain de-lit linear albedos, `ITA°`, auto-matched Fitzpatrick `SSS Radius`/`Scale`/`Coat`/`Roughness`, and lip gradient.
2. Build shader via `build_universal_skin_shader(name, profile, style=...)`.
3. Render `64×64` fast OPTIX diagnostic pass under neutral `6500K` studio light (calibrated against gray/white reference card first) and run closed-loop convergence (`ΔE < 3.5`), followed by full-head **3/4 Hero + Backlit Rim Ear Check + Side-Profile Reference Match (`compare_ref.py`)**.
