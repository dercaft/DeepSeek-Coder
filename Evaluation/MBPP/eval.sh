# MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_NAME_OR_PATH="deepseek-ai/deepseek-coder-1.3b-base"
## MBPP datasets
# https://github.com/google-research/google-research/tree/master/mbpp#evaluation-details
# Task IDs 11-510 are used for testing.
# Task IDs 1-10 were used for few-shot prompting and not for training.
# Task IDs 511-600 were used for validation during fine-tuning.
# Task IDs 601-974 are used for training. 974 - 601 = 374 tasks.
DATASET_ROOT="data/"
# MODEL_NAME_OR_PATH="Salesforce/codegen-2B-nl"
export HF_HOME="/mnt/82_store/huggingface_cache"
echo \#\#\#\# Evaluating: ${MODEL_NAME_OR_PATH}  \#\#\#\#
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export CUDA_LAUNCH_BLOCKING=1
# export torch.backends.cudnn.enable =True
python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py \
    --model_path ${MODEL_NAME_OR_PATH} \
    --logdir logs/${MODEL_NAME_OR_PATH} \
    --dataroot ${DATASET_ROOT}