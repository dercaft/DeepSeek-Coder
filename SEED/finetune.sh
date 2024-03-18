DATA_PATH="mbpp"
OUTPUT_PATH="./output"
# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL_PATH="Salesforce/codegen-2B-mono"
export CUDA_VISIBLE_DEVICES=4,5,6
# # export CUDA_VISIBLE_DEVICES=6
input=${CUDA_VISIBLE_DEVICES}
# Set IFS to comma so that word splitting occurs at commas
IFS=','
# Read the input string into an array
read -ra arr <<<"$input"
# Display length of the array to get count
nproc=${#arr[@]}
# echo nproc ${nproc}

# 默认起始端口
start_port=12345
# 函数来检查端口是否被占用
is_port_in_use() {
    local port="$1"
    nc -z 127.0.0.1 "$port"
}
# 寻找空闲端口
current_port="$start_port"
while is_port_in_use "$current_port"; do
    ((current_port++))
done
# 输出找到的空闲端口
echo "找到一个空闲端口：$current_port"
deepspeed --master_port=$current_port\
     finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero2.json \
    --bf16 True
