#TODO:执行推理任务，并指定实际的模型检查点(latex-ocr-print数据集上微调后得到的LoRA 权重)存储路径
#______________________________--ckpt_dir ________________________________

swift infer --ckpt_dir output/llama3_2-11b/latex-ocr-print-lora --model_type llama3_2-11b