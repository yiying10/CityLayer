#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE_NAME="${1:-}"
SCENE_TYPE="${2:-outdoor}"

if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash scripts/run.sh <scene_name> [scene_type]"
    echo "  scene_name  folder under data/ (e.g. demo)"
    echo "  scene_type  outdoor (default) or indoor"
    exit 1
fi

INPUT_DIR="${REPO_ROOT}/data/${SCENE_NAME}"
OUTPUT_DIR="${REPO_ROOT}/outputs/${SCENE_NAME}"
INPUT_A="${INPUT_DIR}/input_a.png"
INPUT_B="${INPUT_DIR}/input_b.png"

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Input folder not found: ${INPUT_DIR}"
    echo "Create data/${SCENE_NAME}/ with input_a.png and input_b.png (see examples/inputs/)."
    exit 1
fi

missing=0
for f in "${INPUT_A}" "${INPUT_B}"; do
    if [[ ! -f "${f}" ]]; then
        echo "Missing: ${f}"
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    echo "Name the two panoramas input_a.png and input_b.png, same as examples/inputs/."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/model_a" "${OUTPUT_DIR}/model_b"

# ---------------------------------------------------------------------------
# LayerPano3D: input_a / input_b -> outputs/<scene>/model_a / model_b
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[LayerPano3D] scene=${SCENE_NAME}  type=${SCENE_TYPE}"
echo "  A: ${INPUT_A}"
echo "  B: ${INPUT_B}"
echo "  out: ${OUTPUT_DIR}/model_{a,b}"
echo "============================================================"

bash "${SCRIPT_DIR}/run_layerpano3d.sh" "${INPUT_A}" "${OUTPUT_DIR}/model_a" "${SCENE_TYPE}"
bash "${SCRIPT_DIR}/run_layerpano3d.sh" "${INPUT_B}" "${OUTPUT_DIR}/model_b" "${SCENE_TYPE}"

echo "============================================================"
echo "LayerPano3D done:"
echo "  ${OUTPUT_DIR}/model_a/scene/"
echo "  ${OUTPUT_DIR}/model_b/scene/"
echo "============================================================"

# ---------------------------------------------------------------------------
# Align + transition: outputs/<scene>/align/transition_video.mp4
# ---------------------------------------------------------------------------
bash "${SCRIPT_DIR}/run_transition.sh" "${SCENE_NAME}"
