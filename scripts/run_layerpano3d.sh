#!/bin/bash

INPUT_IMAGE=$1
SAVE_DIR=$2
SCENE_TYPE=${3:-outdoor}

base_dir="third_party/LayerPano3D"

mkdir -p ${SAVE_DIR}
python -c "
from PIL import Image
img = Image.open('${INPUT_IMAGE}').convert('RGB')
w, h = img.size
if (w, h) != (1440, 720):
    img = img.resize((1440, 720), Image.LANCZOS)
    print(f'Resized from {w}x{h} to 1440x720')
img.save('${SAVE_DIR}/rgb.png')
"

echo "============Done! Next step: generate depth map==========="
python ${base_dir}/gen_panodepth.py \
    --save_dir ${SAVE_DIR} \
    --depth_model DepthAnythingv2 \
    --input_path "${SAVE_DIR}/rgb.png"

echo "============Done! Next step: generate layer data==========="
python ${base_dir}/gen_autolayering.py \
    --input_dir ${SAVE_DIR} \
    --scene_type ${SCENE_TYPE}

echo "============Done! Next step: generate layered RGB panoramas ==========="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ${base_dir}/gen_layerdata.py \
    --lora_path 'checkpoints/lora_hubs/pano_lora_720*1440_v1.safetensors' \
    --base_dir "${SAVE_DIR}/layering" \
    --seed 16806

echo "============Done! Next step: generate train data==========="
python ${base_dir}/gen_traindata.py \
    --save_dir ${SAVE_DIR} \
    --depth_model DepthAnythingv2 \
    --layerpano_dir "${SAVE_DIR}/layering"

echo "============Done! Next step: train 3DGS==========="
python ${base_dir}/run_layerpano.py \
    --input_dir ${SAVE_DIR} \
    --save_dir ${SAVE_DIR}

echo "============Done! Next step: render video==========="
python -m ${base_dir}/rendering/render_video_360.py \
    --save_dir "${SAVE_DIR}/scene" \
    --elevation 0

python -m ${base_dir}/rendering/render_video_zigzag.py \
    --save_dir "${SAVE_DIR}/scene"

echo "Done! PLY files in ${SAVE_DIR}/scene/"
