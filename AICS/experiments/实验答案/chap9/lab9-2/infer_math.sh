#TODO:执行推理任务，并指定实际的模型检查点(blossom-math-zh数据集上微调后得到的LoRA 权重)存储路径
#______________________________--ckpt_dir ________________________________

#swift infer --ckpt_dir output

# 执行推理任务，并指定实际的模型检查点(blossom-math-zh数据集上微调后得到的LoRA 权重)存储路径
#swift infer --ckpt_dir output/llama3_2-3b/v0-20250606-215845/checkpoint-10 --model_type llama3_2-3b

swift infer --ckpt_dir output/llama3_2-3b/blossom-math-zh-lora --model_type llama3_2-3b