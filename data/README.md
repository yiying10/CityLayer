# Data layout

Put your own 360° panoramas here. Name them like [`examples/inputs/`](../examples/inputs/).

## Input format

- **Projection:** equirectangular 360° panorama
- **Aspect ratio:** 2:1 recommended (demo images are `2048×1024`)
- **Files:** `input_a.png` and `input_b.png` (PNG)
- **Scene type:** street / outdoor (`scene_type=outdoor`)

`run_layerpano3d.sh` resizes each image to `1440×720` and writes `SAVE_DIR/rgb.png`. You do not need to resize beforehand.

Reference images:

- [`examples/inputs/input_a.png`](../examples/inputs/input_a.png)
- [`examples/inputs/input_b.png`](../examples/inputs/input_b.png)

Demo videos: [`examples/outputs/`](../examples/outputs/).

## Folder layout

Create one folder per scene directly under `data/`. Put **two** panoramas in it, named exactly like the examples:

```text
data/
  demo/                 # same images as examples/inputs
    input_a.png
    input_b.png
    gps.json            # optional
    mask_a.png          # optional
    mask_b.png          # optional
  my_scene/
    input_a.png
    input_b.png
```

Optional `gps.json` (used by `scripts/run_transition.sh` to pick `fov_B`):

```json
{
  "A": {"lat": 24.797881, "lon": 120.962515},
  "B": {"lat": 24.797828, "lon": 120.962431}
}
```

`mask_a.png` / `mask_b.png`: white = valid pixels for feature matching.

`scripts/run.sh` reads `data/<scene_name>/` and writes to `outputs/<scene_name>/` with `model_a` and `model_b`:

```text
outputs/my_scene/
  model_a/              # from input_a.png
  model_b/              # from input_b.png
```

Scene folders under `data/` are gitignored. Copy files in locally; do not commit large panoramas.

## Run

From the repo root (requires the LayerPano3D environment at `/LayerPano3D` and a GPU):

```bash
bash scripts/run.sh <scene_name> [scene_type]
```

`scene_type` is `outdoor` (default) or `indoor`.

```bash
# demo pair
bash scripts/run.sh demo

# your folder data/my_scene/{input_a,input_b}.png
bash scripts/run.sh my_scene outdoor
```

This calls `scripts/run_layerpano3d.sh` twice, then `scripts/run_transition.sh`:

| Input | Output |
|---|---|
| `data/<scene>/input_a.png` | `outputs/<scene>/model_a/` |
| `data/<scene>/input_b.png` | `outputs/<scene>/model_b/` |

Alignment video: `outputs/<scene>/align/transition_video.mp4`.

To rerun only the transition:

```bash
bash scripts/run_transition.sh my_scene
```

## Output contents

Each `model_*` directory looks like:

```text
outputs/demo/model_a/
  rgb.png                 # 1440×720 panorama
  layering/               # depth, layer masks, inpaint
  traindata/              # 3DGS training views
  scene/
    gsplat_layer3.ply
    renders_0deg.mp4
    renders_zigzag.mp4

outputs/demo/align/
  transition_video.mp4
  align_matrix.npy
  trajectory.json
  step5/model_B_aligned.ply
```
