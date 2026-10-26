# DitherPal

A PyQt6 desktop app for dithering photos and video, rendering depth-aware pixel art,
and creating 3D camera animations from a photograph. **Neural depth is the shared
source for every depth-aware stage**: High Res shading, Retro seams, video frames,
and the textured 3D surface used for camera animation.

## Examples

### Still Image

These still-image examples predate neural depth becoming the default.

<table>
  <tr>
    <th>Original</th>
    <th>B&W Dither</th>
    <th>Full Color Dither</th>
  </tr>
  <tr>
    <td><img src="media/img2.webp" width="571" height="761"></td>
    <td><img src="media/dithered/img2_dithered_bw_1.webp" width="571" height="761"></td>
    <td><img src="media/dithered/img2_dithered_fullcolor.webp" width="571" height="761"></td>
  </tr>
</table>

### Video (Click to watch)

<table>
  <tr>
    <th>Original</th>
    <th>B&W Dither</th>
    <th>Full Color Dither</th>
  </tr>
  <tr>
    <td>
      <a href="https://youtu.be/6NttwZXAyww"><img src="https://img.youtube.com/vi/6NttwZXAyww/maxresdefault.jpg" width="1280" height="720"></a>
    </td>
    <td>
      <a href="https://youtu.be/xMnGWsTfyyQ"><img src="https://img.youtube.com/vi/xMnGWsTfyyQ/maxresdefault.jpg" width="1280" height="720"></a>
    </td>
    <td>
      <a href="https://youtu.be/3B4rneGYVPo"><img src="https://img.youtube.com/vi/3B4rneGYVPo/maxresdefault.jpg" width="1280" height="720"></a>
    </td>
  </tr>
</table>

### Pixel Art Mode

#### Retro

Flat color bands and checkerboard seams in a fjord landscape. Click either image
to inspect it at full size.

| Original | Retro |
| --- | --- |
| [![Original fjord landscape](media/comparisons/fjord.jpg)](media/comparisons/fjord.jpg) | [![Fjord rendered with Retro pixel art](media/comparisons/fjord_Retro.jpg)](media/comparisons/fjord_Retro.jpg) |

Photo: [Alexey Topolyanskiy / Unsplash](https://unsplash.com/photos/-oWyJoSqBRM).

#### High Res

Fine dithered shading across a city skyline at sunset. Click either image
to inspect it at full size.

| Original | High Res |
| --- | --- |
| [![Original city skyline at sunset](media/comparisons/city_skyline_sunset.jpg)](media/comparisons/city_skyline_sunset.jpg) | [![City skyline rendered with High Res pixel art](media/comparisons/city_skyline_sunset_High-Res.jpg)](media/comparisons/city_skyline_sunset_High-Res.jpg) |

Photo: [Kevin Young / Unsplash](https://unsplash.com/photos/-icmOdYWXuQ).

> [!TIP]
> **Crank up the upscale factor!** Pixel Art Mode starts you at 2×, but **4× and beyond is where it really shines** — every step up gives you finer art pixels and brings back leaves, edges and small details. See [Pixel art](#pixel-art) below.

> [!IMPORTANT]
> ### 🔍 Play with the Upscale Factor — it's the secret to great pixel art
>
> Pixel Art Mode sets the *Block Size* (how big each art pixel is) for a **2×** upscale.
> When you raise the **Upscale Factor** and leave the block size alone, every art
> pixel covers **less of the original picture** — so you get **more, finer art
> pixels** and the small details come back.
>
> - **2×** — chunky, bold pixel art; small details can get lost
> - **4×** — a useful starting point for finer detail in photos
> - **6×–8×** — even finer detail for busy pictures
>
> Try a few values to find the right factor for your picture. Higher upscale
> factors also increase processing time and memory use.
>
> #### 🎚️ Dial in the exact look: Upscale and Block Size work as a sliding window
>
> The two settings pull in opposite directions:
>
> - **Upscale Factor ↑** → closer to the original picture, with more detail
> - **Block Size ↑** → chunkier, more pixel-art-like
> - **Block Size ↓** → finer art pixels, greater fidelity, and more original detail
>
> You can adjust Block Size in either direction: raise it for a stronger pixel-art
> style, or lower it to recover detail while keeping the same Upscale Factor.
>
> To find the balance you want:
>
> 1. **Push the Upscale Factor up** until the details you care about (faces, text,
>    edges) come through.
> 2. **Nudge Block Size up in small steps** until it looks like pixel art again.
> 3. Too blocky? Take Block Size back a step, or raise the upscale another notch
>    and repeat.
>
> What matters is the ratio: an art pixel covers approximately **Block Size ÷
> Upscale Factor** pixels of the original. **4× upscale with block 16** makes the
> same size art pixels as **2× with block 8**. The higher upscale gives you finer
> steps to adjust the ratio. With **Scale result back to original size**, choose
> a Block Size that's a multiple of the Upscale Factor so every art pixel comes
> out the same size; the ratio warning helps flag uneven combinations.
>
> #### Adjust depth to guide the visual focus
>
> Open **Depth map** and adjust its strength slider while tuning the pixel-art look:
>
> - **Move left (lower strength)** to help background detail come into focus.
> - **Move right (higher strength)** to put more visual emphasis on the foreground.
>
> **About 20%–100%** is a useful range for most images. Try **above 100%** when a
> subject is particularly close to the camera and needs stronger foreground
> emphasis. These adjustments control the depth-driven pixel-art shading; compare
> renders to find the balance that suits your image.

**What Pixel Art Mode sets for you:**

| Setting | What it picks |
| --- | --- |
| Method | Hybrid Bayer + Floyd–Steinberg, Grid style, 85% Pattern Strength |
| Art pixel grid | Suggested from your picture's detail: busy photos get finer grids, simple cartoons chunkier ones |
| Colors | Cel Shading on, with a palette built from your image's colors; custom-palette modes use your chosen colors |
| Upscale | 2× to start — try raising it for finer detail |

All settings stay editable. **Art Pixels Across Short Side** lets you override
the grid directly; lower values make chunkier art pixels. **Pattern Strength**
sets the amount of pixel-art dithering; 0% gives flat cel shading.

**Cel Shading controls** (Grid style):

| Control | What it does |
| --- | --- |
| Color Bands | Size of the image-derived palette (default 24). More bands preserve more colors; fewer give a flatter, more stylized result |
| Shading Dither | Without a Pixel Art Mode, adds checkerboard shading between color bands on gradients (default 35%). 0% gives flat bands. High Res and Retro use their own dithering instead |

With a **Custom palette** color mode, cel shading uses your colors instead of
building a palette.

---

## Install

Use Python **3.12–3.14** in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ditherpal.py
```

On Windows, activate with `.venv\Scripts\activate` instead. A graphical desktop is
required for the app. CPU inference works without CUDA or a dedicated GPU.
OpenCV uses its headless wheel because PyQt6 supplies the GUI. Use a fresh virtual
environment rather than installing multiple OpenCV variants together.

The first depth-aware operation downloads **Depth Anything V2 Small** (about
99 MB) into `models/`. The download is pinned and SHA-256 verified. Images are
processed locally. Later runs reuse the weights, an in-process ONNX session, and
cached source-image maps. Models and generated output are excluded from Git.
The source cache holds at most four small previews. Neural inference releases its
temporary activation buffers between jobs instead of retaining peak allocations
for every image shape. Completed render workers release their image buffers;
the model session stays loaded to avoid reloading weights for each render.
There is one Python environment; the app does not depend on an experimental folder.

To download the models ahead of time (depth, plus the LaMa inpainting model used
by 3D animation gap repair):

```bash
python -m ditherpal_depth --download-model
```

For an alternate writable model-cache directory, set `DITHERPAL_MODEL_DIR` before
starting the app. After the first successful download, depth inference works offline.
A model/load failure is reported explicitly; depth-aware jobs do not fall back to
texture-based estimates. Ordinary dithering methods that do not use depth still
work without downloading weights.

## Use

1. Select an image or video.
2. Choose a dither method and color mode, or select **High Res** / **Retro** Pixel Art Mode.
3. Adjust the art-pixel grid, block size, palette, and upscale factor as desired.
4. Click **Process Image**. Outputs save in `dithered/` beside the source, with
   numbered filenames when a version already exists.

The settings panel scrolls; the process controls remain visible. Block-size ratio
warnings overlay the settings and do not reserve a row or move the scroll area.

### One depth control

Click **Depth map** to open a compact neural-depth preview. It has one slider:

- **0%:** flat depth.
- **100%:** the model's estimated relative depth.
- **200%:** stronger separation between near and far.

White is near and black is far. Moving the slider updates the preview immediately
without running the model again. The selected strength is saved and used by the
next still-image, video, or 3D camera render. Video previews show their first frame;
processing uses a neural map for **each** video frame. There is no depth-source
selector, brush editor, or heuristic depth path.

Neural depth is relative, not a measurement in metres. The model can still make
mistakes on ambiguous scenes. The slider controls depth only; it does not change
your palette, dither method, or color settings.

### Pixel art

- **High Res:** solid cel-shaded art pixels with fine dithering between nearby
  shades. Neural depth, texture, and gradients control where shading appears.
- **Retro:** flat color bands with checkerboard seams between neighboring shades.
  Neural depth controls seam placement and width.

Both modes enable Cel Shading and start with a 2× upscale. The art-pixel grid is
suggested from image detail and stays editable. **Pattern Strength** controls the
amount of dithering; **Color Bands** controls the image-derived cel palette (4–64).
Custom-palette modes preserve your chosen colors.

**Export at native pixel size** saves one image pixel per art pixel in High Res
and Retro. It uses the renderer's exact grid, including partial edge cells,
without averaging or introducing colors. Stills export as lossless PNGs, including
when the source is a JPEG. The output dimensions appear beside
the selected image. Upscaling still controls art detail; both restore-original-size
options are overridden while this checkbox is selected. For example, a 640×480
image rendered at 2× with 8-pixel blocks exports at 160×120.
Video uses the same grid; MP4 pads an odd dimension with an edge pixel rather
than stretching the grid. Neural 3D animation uses the smaller texture and canvas
while retaining smooth camera sampling, so moving 3D surfaces can blend pixels.

**Preserve bright details** (on by default) gives narrow bright features more
weight before art-pixel averaging, keeps rare highlight shades within the chosen
cel palette size, and prevents valid sparks from being removed as stray pixels.
It applies inside grid cel shading for stills, video, and baked 3D textures.
Every output art pixel still comes from the palette and keeps the selected block
size. High Res can add its usual intermediate ramp shades; custom palettes stay
exact. Disable this option to use the original averaging and cleanup.

Highlight detection uses perceived brightness and contrast within each art cell
so a strong blue channel alone does not turn an ordinary edge into a spark.
Palette fitting covers hue as well as lightness, and shading ramps avoid opposing
color families when compatible shades exist. This reduces cool-colored speckles
on warm materials such as skin without removing real blue lighting.

Depth is inferred from source RGB before grayscale conversion or cel shading,
then shared with the selected pixel-art style. Re-rendering the same unchanged
photo with different dither settings reuses the cached map.

### Neural 3D animation

For a still image, choose **Animated GIF**, **MP4 video**, or **Animated SVG** under
**Neural 3D Animation**. The app builds a textured triangle mesh from neural depth,
then renders a perspective camera moving in and out with a Z buffer. The old
image-warp zoom has been removed.

**AI detail**, **Camera travel**, **Side motion**, and **Fill small gaps** control
the camera render. The same depth-strength setting shapes the geometry and drives
pixel-art shading. Your chosen dither method runs once on the source image, and
the result is baked into the mesh texture before the camera animation. Frames
render directly at the output size with smooth bilinear texture sampling, so the
pattern follows the surface during motion. Filtering can introduce intermediate
colors and soften the pixel grid. Animations contain 48 frames over four seconds
and have a 2048-pixel output cap. SVG output is an animated sequence of embedded
raster frames.

**Fill small gaps** selects nearby background donor patches once, then attaches
them to background repair geometry. **The seam contains no foreground extensions
or foreground feathering.** The original foreground stays fully 3D and keeps its
baked dithered texture; pixels already covered by the original mesh are preserved.

Each camera frame matches donor tone to visible, depth-compatible background and
heals patch boundaries with OpenCV seamless cloning. High Res and Retro then apply
the selected pixel-art algorithm to the repairs, using the source art-pixel grid
and only colors found in each donor's background material. Foreground samples are
excluded before averaging or shading. The main texture is not dithered again.
Exact repeated repair inputs reuse a bounded pattern cache.

Background texture continues up to 28 source pixels behind exposed edges. The
extra geometry is clipped against source depth; per-frame depth checks reject
near repair fragments before they can hide a valid farther background patch.
Residual gaps copy accepted background repair pixels. Turn **Fill small gaps**
off to render the raw surface without repairs.

**Hidden layers** stop the background from showing through inside a subject,
for example between an arm and the robe behind it. With **Fill small gaps** on,
the app paints two hidden layers once with a local
[LaMa](https://github.com/advimman/lama) inpainting model (about 208 MB, downloaded
on first use). The sky layer paints the subject out of the scene. The inner layer
paints out each foreground piece that stands on a steep depth step in front of the
subject's own interior. Both layers are baked with the same dither method and
locked palette as the main texture, meshed as continuous sheets just behind the
surface they continue, and shown only where the camera opens a gap. Gaps that no
layer covers fall back to the background repair. Inner gaps there are split
between the nearer and farther surface colors in whole art pixels.

The camera mesh has up to 1024 vertices along the longer side, about 2x the
source resolution for small pixel art. Tears follow depth edges closely, and soft
slopes such as a hand meeting an arm stretch instead of tearing open. Background
donor patches still follow the coarser source-resolution tears.

If the inpainting model cannot be downloaded or loaded, the render continues with
background repair only and the status line says so. The subject/background split
assumes one main subject in front of a farther background. Hidden layers are
invented content, and they can be wrong when depth is wrong.

Per-frame healing and pixel-art repair add CPU work, especially on images with
many depth boundaries. Hidden layers add about 10 seconds of preparation, plus
about 15 seconds the first time the inpainting model loads. The dense mesh adds
about 1–2 seconds per large frame. Camera angles render in parallel worker processes
(up to six, fewer when free memory is low; each needs about 0.6 GB) and share the scene's
memory. Frames are identical to a one-at-a-time render. If a worker dies or stops
responding, the remaining frames render one at a time. Repeated or mirrored texture, depth mistakes,
thin edges, and complex effects such as flames can still show artifacts.

**3D Mesh** beneath the result opens a standalone textured viewer in your browser.
You can orbit the geometry and show its triangles. The viewer and mesh exports use
the same baked texture as the animation and show the original visible surface.
OBJ/MTL, PLY, depth maps, raw peak renders, uncovered-pixel masks, and a report are
saved in `dithered/neural-zoom/`. When gap repair adds geometry, its triangle layers
and vertex alpha are saved separately in `repair_layers.npz`, alongside the donor
atlases and source locations. Peak repair masks identify covered gaps, and peak
donor/continuation atlases record the frame's repair textures. Each hidden layer
saves its mask, painted image and baked texture (`hidden_<name>_*.png`), and
`hidden_layer_mask.png` marks the peak frame's gaps that hidden layers filled. The
raw mesh viewer does not composite these animation repairs.

This is a single-photo visible-surface reconstruction, often called 2.5D. The
geometry has real XYZ coordinates and occlusion, but unseen object backs and
backgrounds remain unknown. Gap filling is a cosmetic correction, not recovered
geometry. Small camera motions usually work best.

### Video and GIF input

Choose **Video / GIF output** to export imported media as MP4 or animated GIF.
Frames are decoded in order, processed in bounded batches, and encoded
incrementally. A reusable thread pool handles dithering. **Lock video palette to
first frame** reduces repeated fitting and color shifts; disable it when later
scenes introduce substantially different colors.

High Res and Retro infer neural depth from each source frame before resizing and
share that map across its shading stages. Neural inference adds CPU work to video
processing; it does not drop frames. Non-depth-aware dither methods avoid that work.
A single-image depth model can still exhibit temporal variation on moving scenes.

GIF supports at most 256 colors per frame and centisecond timing. GIF export is
not inherently faster than MP4. Current video exports are silent: source audio
is not copied.

### Review and keep results

- **View:** open the selected result in your default image/media viewer.
- **Finalize:** copy it into `dithered/Finalized/`, preserving existing versions.
- **Comparison:** save a labeled original/result PNG in `dithered/Comparisons/`.
  The dither keeps its resolution; the original fits alongside it without cropping.
  Animations use the first frame. Older results can prompt for their original.
- **Recent:** browse up to 100 available outputs and restore one to the result
  viewer. The original and mesh associations are remembered across launches.
- **3D Mesh:** inspect the selected neural camera result's geometry.

Selecting a recent output leaves source-image and rendering controls in place.

## Dither methods and color limits

| Method | Description |
| --- | --- |
| Floyd–Steinberg | Error diffusion |
| Jarvis–Judice–Ninke | Wider grayscale diffusion; color currently uses Floyd–Steinberg |
| Hybrid Bayer + Floyd–Steinberg | Grid, Hatched, and Woven styles; Grid powers Pixel Art Mode |
| Bayer / Ordered | Ordered threshold patterns |
| Rosette | Color-channel halftone dots |
| Text Pattern | Custom text used as a halftone mask |

Output Levels ranges from 2–16. In full-color palette-based paths, levels 2–6
produce up to `levels³` RGB colors; larger settings learn an image palette capped
at 64 colors. Bayer/Ordered/Rosette use binary channel combinations. Text Pattern
can retain source RGB colors. Cel-shaded pixel art uses Color Bands instead; High
Res can add intermediate ramp shades. GIF encoding limits each frame to 256 colors.

## Development and GitHub

```bash
python -m pip check
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

Tests mock neural inference where appropriate, so CI needs no GPU, model weights,
or image upload/download. Geometry tests exercise real projection, near-surface
occlusion, perspective-correct texture interpolation, and portable exports.
GitHub Actions runs the suite on Python 3.12 and 3.14.

Production depth code lives in `ditherpal_depth/`; `depth_editor.py` implements the
single-slider popup. The former `test-neural-zoom/` sandbox is local-only and ignored.
Do not add model weights, private photos, caches, virtual environments, credentials,
or generated media to commits. Existing files under `media/` are intentional examples.

To export a neural depth preview without launching the GUI:

```bash
python -m ditherpal_depth --image photo.jpg --output photo-depth.png
```

The preview is a 16-bit PNG at the cached depth-preview resolution. Choose a new
output filename; existing files are never overwritten by this command.

See [MODEL_NOTICE.md](MODEL_NOTICE.md) for model provenance and license links.

## Acknowledgements

Special thanks to **[LeoBorcherding](https://github.com/LeoBorcherding)** /
**[BlueberryMathematics](https://github.com/BlueberryMathematics)** for color dithering
implementation and speed optimizations. Neural depth uses
[Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) and its
[ONNX Community conversion](https://huggingface.co/onnx-community/depth-anything-v2-small).
3D animation hidden layers use [LaMa](https://github.com/advimman/lama) via the
[Carve LaMa ONNX export](https://huggingface.co/Carve/LaMa-ONNX).
