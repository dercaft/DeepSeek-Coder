# MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
# MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-base"
## MBPP datasets
# https://github.com/google-research/google-research/tree/master/mbpp#evaluation-details
# Task IDs 11-510 are used for testing.
# Task IDs 1-10 were used for few-shot prompting and not for training.
# Task IDs 511-600 were used for validation during fine-tuning.
# Task IDs 601-974 are used for training. 974 - 601 = 374 tasks.
export NCCL_SOCKET_NTHREADS=8
DATASET_ROOT=MBPP_SEED/data
LOG_DIR="/mnt/82_store/xxw/code/DeepSeek-Coder/outputs/0324"
mkdir -p $LOG_DIR
MODEL_NAME_OR_PATH="/mnt/82_store/xxw/models/deepseek-coder-1.3b-base"
echo \#\#\#\# Evaluating: ${MODEL_NAME_OR_PATH}  \#\#\#\#
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m accelerate.commands.launch --config_file MBPP_SEED/test_config.yaml MBPP_SEED/seed.py \
    --model_path ${MODEL_NAME_OR_PATH} \
    --logdir $LOG_DIR \
    --dataroot ${DATASET_ROOT}

