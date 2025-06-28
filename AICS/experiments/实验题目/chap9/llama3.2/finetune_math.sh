#TODO: 使用SWIFT 轻量级训练推理工具进行监督微调
_________________________ \
    #TODO:指定模型类型，使用Llama3.2的3B的参数版本
	--model_type _________________________\
    #TODO: 指定模型的存储路径
	--model_id_or_path <_________________________> \
	--dataset blossom-math-zh \
    --num_train_epochs 1 \
    --max_steps 10 \
    #TODO：指定微调类型为 LoRA
    --sft_type _________________________ \
    --output_dir output \
    --eval_steps 200 