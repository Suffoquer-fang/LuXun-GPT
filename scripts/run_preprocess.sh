# preprocess
python ../random_instruction.py \
    --data_path ../example_data/selected_aug.jsonl \
    --save_path ../example_data/luxun_data.jsonl


# tokenize
python ../tokenize_dataset.py \
    --jsonl_path ../example_data/luxun_data.jsonl \
    --save_path ../example_data/luxun_dataset \
    --max_seq_length 400 