CUDA_VISIBLE_DEVICES=0,1 \
python ../lora_finetune.py \
    --dataset_path ../example_data/luxun_dataset \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 5000 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir ../saved_models