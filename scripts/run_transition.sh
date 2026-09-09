#!/bin/bash
# Align model_a / model_b and render a transition video.
#
#   bash scripts/run.sh <scene_name>
#   bash scripts/run_transition.sh <scene_name>
#
# Reads  (produced by scripts/run.sh):
#   outputs/<scene>/model_a/{rgb.png, layering/depth.npy, scene/gsplat_layer3.ply}
#   outputs/<scene>/model_b/{...}
#
# Optional (data/<scene>/):
#   gps.json      GPS of A/B, used to pick fov_B
#   mask_a.png    white = valid pixels for matching
#   mask_b.png
#
# Writes:
#   outputs/<scene>/align/transition_video.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
LAYERPANO_DIR="${LAYERPANO_DIR:-${REPO_ROOT}/third_party/LayerPano3D_new}"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash scripts/run_transition.sh <scene_name>"
    echo "  Expects outputs/<scene>/model_{a,b} from scripts/run.sh"
    exit 1
fi

INPUT_DIR="${REPO_ROOT}/data/${SCENE_NAME}"
MODEL_DIR="${REPO_ROOT}/outputs/${SCENE_NAME}"
MODEL_A_DIR="${MODEL_DIR}/model_a"
MODEL_B_DIR="${MODEL_DIR}/model_b"
OUT_DIR="${MODEL_DIR}/align"

PANO_A="${MODEL_A_DIR}/rgb.png"
PANO_B="${MODEL_B_DIR}/rgb.png"
DEPTH_A="${MODEL_A_DIR}/layering/depth.npy"
DEPTH_B="${MODEL_B_DIR}/layering/depth.npy"
MODEL_A="${MODEL_A_DIR}/scene/gsplat_layer3.ply"
MODEL_B="${MODEL_B_DIR}/scene/gsplat_layer3.ply"

MASK_A=""
MASK_B=""
[[ -f "${INPUT_DIR}/mask_a.png" ]] && MASK_A="${INPUT_DIR}/mask_a.png"
[[ -f "${INPUT_DIR}/mask_b.png" ]] && MASK_B="${INPUT_DIR}/mask_b.png"

FRAMES="${FRAMES:-240}"
B_FRAMES="${B_FRAMES:-120}"
ROTATE_FRAMES="${ROTATE_FRAMES:-180}"
MOVE_YAW="${MOVE_YAW:-180}"
MOVE_SCALE="${MOVE_SCALE:-3.0}"
TRANSITION_DURATION="${TRANSITION_DURATION:-10}"
FPS="${FPS:-30}"
TRANSITION_WIDTH="${TRANSITION_WIDTH:-1024}"
TRANSITION_HEIGHT="${TRANSITION_HEIGHT:-1024}"
TRANSITION_BLEND_Z_WEIGHT="${TRANSITION_BLEND_Z_WEIGHT:-0.6}"
MATCHER="${MATCHER:-lightglue}"
if [[ -n "${YAWS:-}" ]]; then
    read -r -a YAWS <<< "${YAWS}"
else
    YAWS=(0 45 90 135 180 225 270 315)
fi
FOV="${FOV:-90}"
RES="${RES:-512}"
RANSAC_ITERATIONS="${RANSAC_ITERATIONS:-500}"
RANSAC_THRESHOLD="${RANSAC_THRESHOLD:-0.75}"
LIGHTGLUE_THRESH="${LIGHTGLUE_THRESH:-0.9}"
REPROJ_THRESH="${REPROJ_THRESH:-1.5}"
MIN_INLIERS="${MIN_INLIERS:-25}"

missing=0
for f in "${PANO_A}" "${PANO_B}" "${DEPTH_A}" "${DEPTH_B}" "${MODEL_A}" "${MODEL_B}"; do
    if [[ ! -f "${f}" ]]; then
        echo "Missing: ${f}"
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    echo "Run first: bash scripts/run.sh ${SCENE_NAME}"
    exit 1
fi

if [[ ! -d "${LAYERPANO_DIR}" ]]; then
    echo "LayerPano3D not found: ${LAYERPANO_DIR}"
    echo "Place it at third_party/LayerPano3D_new or set LAYERPANO_DIR."
    exit 1
fi

for py in panorama_to_patches.py feature_matching.py depth_unprojection.py similarity_alignment.py transform_model.py build_trajectory.py render_transition_hq.py; do
    if [[ ! -f "${SRC_DIR}/${py}" ]]; then
        echo "Missing: ${SRC_DIR}/${py}"
        exit 1
    fi
done

GPS_ARGS=()
GPS_FILE="${INPUT_DIR}/gps.json"
if [[ -f "${GPS_FILE}" ]]; then
    gps_line="$(python3 - "${GPS_FILE}" <<'PY'
import json, sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
a = d.get("A") or d.get("a")
b = d.get("B") or d.get("b")
if isinstance(a, dict) and isinstance(b, dict):
    vals = (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon"))
else:
    vals = (d.get("gps_a_lat"), d.get("gps_a_lon"), d.get("gps_b_lat"), d.get("gps_b_lon"))
if any(v is None for v in vals):
    sys.exit(1)
print(vals[0], vals[1], vals[2], vals[3])
PY
)" || gps_line=""
    if [[ -n "${gps_line}" ]]; then
        read -r GPS_A_LAT GPS_A_LON GPS_B_LAT GPS_B_LON <<< "${gps_line}"
        GPS_ARGS=(
            --gps_a_lat "${GPS_A_LAT}"
            --gps_a_lon "${GPS_A_LON}"
            --gps_b_lat "${GPS_B_LAT}"
            --gps_b_lon "${GPS_B_LON}"
        )
        echo "Loaded GPS from ${GPS_FILE}"
    else
        echo "Warning: ${GPS_FILE} is present but incomplete; ignoring GPS."
    fi
fi

mkdir -p "${OUT_DIR}"
export OUT_DIR LAYERPANO_DIR
export PYTHONPATH="${LAYERPANO_DIR}:${SRC_DIR}:${PYTHONPATH:-}"

STEP1_OUT="${OUT_DIR}/step1"
STEP2_OUT="${OUT_DIR}/step2"
STEP3_OUT="${OUT_DIR}/step3"
STEP4_OUT="${OUT_DIR}/step4"
STEP5_OUT="${OUT_DIR}/step5"

echo "============================================================"
echo "[Step 1] Panorama -> patches"
echo "  scene: ${SCENE_NAME}"
echo "  out:   ${OUT_DIR}"
echo "============================================================"
STEP1_CMD=(
    python3 "${SRC_DIR}/panorama_to_patches.py"
    --panoA "${PANO_A}" --panoB "${PANO_B}"
    --out-dir "${STEP1_OUT}"
    --fov "${FOV}" --res "${RES}"
    --yaws "${YAWS[@]}"
)
[[ -n "${MASK_A}" ]] && STEP1_CMD+=(--maskA "${MASK_A}")
[[ -n "${MASK_B}" ]] && STEP1_CMD+=(--maskB "${MASK_B}")
"${STEP1_CMD[@]}"

echo ""
echo "============================================================"
echo "[Step 2] Panorama matching"
echo "============================================================"
python3 "${SRC_DIR}/feature_matching.py" \
    --manifest "${STEP1_OUT}/patch_manifest.json" \
    --out-dir  "${STEP2_OUT}" \
    --matcher  "${MATCHER}" \
    --lightglue-thresh "${LIGHTGLUE_THRESH}" \
    --reproj-thresh    "${REPROJ_THRESH}" \
    --min-inliers      "${MIN_INLIERS}"

echo ""
echo "============================================================"
echo "[Step 3] Depth unprojection"
echo "============================================================"
python3 "${SRC_DIR}/depth_unprojection.py" \
    --matches "${STEP2_OUT}/filtered_matches.json" \
    --depthA  "${DEPTH_A}" --depthB "${DEPTH_B}" \
    --out-dir "${STEP3_OUT}" \
    --panoA   "${PANO_A}" --panoB "${PANO_B}"

echo ""
echo "============================================================"
echo "[Step 4] RANSAC + Umeyama similarity alignment"
echo "============================================================"
python3 "${SRC_DIR}/similarity_alignment.py" \
    --point-pairs "${STEP3_OUT}/point_pairs.npy" \
    --matches     "${STEP3_OUT}/step3_filtered_matches.json" \
    --out-dir     "${STEP4_OUT}" \
    --iterations  "${RANSAC_ITERATIONS}" \
    --threshold   "${RANSAC_THRESHOLD}" \
    --panoA       "${PANO_A}" --panoB "${PANO_B}"

echo ""
echo "============================================================"
echo "[Step 5] Apply transform to model_B"
echo "============================================================"
python3 "${SRC_DIR}/transform_model.py" \
    --modelB    "${MODEL_B}" \
    --out       "${STEP5_OUT}/model_B_aligned.ply" \
    --out-dir   "${STEP5_OUT}" \
    --transform "${STEP4_OUT}/transform.json"

echo ""
echo "============================================================"
echo "[Step 6] Build alignment matrix"
echo "============================================================"
python3 - <<'PY'
import json, os, numpy as np
from pathlib import Path
base = Path(os.environ["OUT_DIR"])
with open(base / "step4" / "transform.json") as f:
    t = json.load(f)
R   = np.asarray(t["R"], dtype=np.float64)
vec = np.asarray(t["t"], dtype=np.float64)
s   = float(t["s"])
T = np.eye(4, dtype=np.float64)
T[:3, :3] = R.T / s
T[:3, 3]  = -(R.T @ vec) / s
np.save(base / "align_matrix.npy", T)
print(f"Saved align_matrix.npy, B origin in A: {T[:3,3]}")
PY

echo ""
echo "============================================================"
echo "[Step 6.5] Build trajectory"
echo "============================================================"
TRAJ_CMD=(
    python3 "${SRC_DIR}/build_trajectory.py"
    --model_A       "${MODEL_A}"
    --model_B       "${STEP5_OUT}/model_B_aligned.ply"
    --stats_csv     "${STEP2_OUT}/match_pair_stats.csv"
    --output        "${OUT_DIR}/trajectory.json"
    --A_frames      "${FRAMES}"
    --B_frames      "${B_FRAMES}"
    --move_yaw      "${MOVE_YAW}"
    --move_scale    "${MOVE_SCALE}"
    --rotate_frames "${ROTATE_FRAMES}"
)
if ((${#GPS_ARGS[@]})); then
    TRAJ_CMD+=("${GPS_ARGS[@]}")
fi
"${TRAJ_CMD[@]}"

FOV_B="$(cat "${OUT_DIR}/fov_B.txt" 2>/dev/null || echo "70")"
echo "fov_B: ${FOV_B}"

echo ""
echo "============================================================"
echo "[Step 7] Render transition video"
echo "============================================================"
python3 "${SRC_DIR}/render_transition_hq.py" \
    --model_A             "${MODEL_A}" \
    --model_B             "${STEP5_OUT}/model_B_aligned.ply" \
    --trajectory          "${OUT_DIR}/trajectory.json" \
    --output_video        "${OUT_DIR}/transition_video.mp4" \
    --width               "${TRANSITION_WIDTH}" \
    --height              "${TRANSITION_HEIGHT}" \
    --fps                 "${FPS}" \
    --transform           "${STEP4_OUT}/transform.json" \
    --transition_duration "${TRANSITION_DURATION}" \
    --B_frames            "${B_FRAMES}" \
    --fov-deg             90 \
    --fov-deg-B           "${FOV_B}" \
    --blend-z-weight      "${TRANSITION_BLEND_Z_WEIGHT}"

echo ""
echo "============================================================"
echo "Done: ${OUT_DIR}/transition_video.mp4"
echo "============================================================"
