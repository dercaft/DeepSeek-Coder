# MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
# MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-base"
## MBPP datasets
# https://github.com/google-research/google-research/tree/master/mbpp#evaluation-details
# Task IDs 11-510 are used for testing.
# Task IDs 1-10 were used for few-shot prompting and not for training.
# Task IDs 511-600 were used for validation during fine-tuning.
# Task IDs 601-974 are used for training. 974 - 601 = 374 tasks.
DATASET_ROOT="data/"

MODEL_NAME_OR_PATH="Salesforce/codegen-2B-mono"
MODEL_NAME_OR_PATH="/mnt/42_store/wyh/DeepSeek-Coder/SEED/output"
echo \#\#\#\# Evaluating: ${MODEL_NAME_OR_PATH}  \#\#\#\#
CUDA_VISIBLE_DEVICES=6 python -m accelerate.commands.launch --config_file test_config.yaml seed.py \
    --model_path ${MODEL_NAME_OR_PATH} \
    --logdir logs/${MODEL_NAME_OR_PATH} \
    --dataroot ${DATASET_ROOT}

