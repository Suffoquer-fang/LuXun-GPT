CUDA_VISIBLE_DEVICES=9 \
python ../inference.py \
    --lora Suffoquer/LuXun-lora \
    --instruction 用鲁迅风格的语言改写，保持原来的意思： \
    --input_path ../test_data/test.txt \
    --output_path ../test_data/output.txt 


# CUDA_VISIBLE_DEVICES=9 \
# python ../inference.py \
#     --lora Suffoquer/LuXun-lora \
#     --instruction 用鲁迅风格的语言改写，保持原来的意思： \
#     --interactive


