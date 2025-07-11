#TODO: 使用SWIFT 轻量级训练推理工具进行监督微调
#_________________________ \
#    #TODO:指定模型类型，使用Llama3.2的11B的参数版本
#	--model_type _________________________\
#    #TODO: 指定模型的存储路径
#	--model_id_or_path <_________________________> \
#	--dataset latex-ocr-print \
#    --num_train_epochs 1 \
#    --max_steps 10 \
#    #TODO：指定微调类型为 LoRA
#    --sft_type _________________________ \
#    --output_dir output \
#    --eval_steps 200 

swift sft \
    --model_type llama3_2-11b-vision-instruct \
    --model_id_or_path /workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-11B-Vision-Instruct/ \
    --dataset latex-ocr-print \
    --num_train_epochs 1 \
    --max_steps 10 \
    --sft_type lora \
    --output_dir output \
    --eval_steps 200

# 获取最新checkpoint路径
ckpt_dir=$(ls -dt output/llama3_2-11b-vision-instruct/v*/checkpoint-* | head -n 1)
# 复制到你指定的目录（如 output/llama3_2-11b/latex-ocr-print-lora）
target_dir=output/llama3_2-11b/latex-ocr-print-lora
mkdir -p "$target_dir"
cp -r "$ckpt_dir"/* "$target_dir"/


