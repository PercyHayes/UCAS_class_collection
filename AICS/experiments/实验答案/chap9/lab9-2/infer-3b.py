import torch
import torch_mlu
#TODO: 从 transformers 库中导入 pipeline
from transformers import pipeline
#_______________________________________________________
model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-3B/"
#TODO: 创建一个文本生成的管道，指定任务类型、模型路径、数据类型，并在MLU设备上运行
#pipe = _______________________________________________________

pipe = pipeline(
    task="text-generation",
    model=model_id,
    device=torch.device("mlu"),
    torch_dtype=torch.float16,
    truncation=True
)

#TODO：使用管道对象执行文本生成任务，传入提示词"The key to life is"
#output = _______________________________________________________

#output = pipe("The key to life is", max_length=50, num_return_sequences=1)
output = pipe("The key to life is")
# 打印生成的文本结果，提取输出字典中的 'generated_text' 字段
#_______________________________________________________
print(output[0]['generated_text'])
print("Llama3.2 textchat PASS!")