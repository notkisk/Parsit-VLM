export NUM_GPUS=1
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

# # Check if running from repo root (should find 'parsit/pyproject.toml')
# if [ ! -f "parsit/pyproject.toml" ]; then
#   echo "Error: Please run this script from the repository root directory (where 'parsit/pyproject.toml' exists)."
#   exit 1
# fi

LLM_VERSION="Qwen/Qwen3-1.7B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="parsit-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"


# Get absolute paths
REPO_ROOT="$(pwd)"
TRAIN_SCRIPT="$REPO_ROOT/parsit/train/train.py"
DEEPSPEED_CONFIG="$REPO_ROOT/scripts/zero3.json"
DATA_PATH="$REPO_ROOT/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="$REPO_ROOT/LLaVA-Pretrain"

ACCELERATE_CPU_AFFINITY=1 torchrun --standalone --nproc_per_node=$NUM_GPUS \
    "$TRAIN_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "${LLM_VERSION}" \
    --version "${PROMPT_VERSION}" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "/checkpoints/projectors/${BASE_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "$BASE_RUN_NAME" \
    --attn_implementation sdpa

# Error handling for torchrun
if [ $? -ne 0 ]; then
  echo "Error: torchrun command failed. Please check the logs above for details."
  exit 2
fi

# You can delete the sdpa attn_implementation if you want to use flash attn