import transformers
import torch
import torch_mlu
import time

model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-3B/"

#TODO: 创建一个文本生成的管道，指定任务类型、模型路径、数据类型，并在MLU设备上运行
pipeline = _______________________________________________________________
messages = [
    {"role": "system", "content": "You are a story writing chatbot"},
    {"role": "user", "content": "Once upon a time, .... start to write a very long story"},
]

#TODO: 应用聊天模板，将消息转化为适合模型的输入格式
prompt = _______________________________________________________________
#TODO: 定义文本生成的终止符列表，包含模型的结束标记和自定义标记<|eot_id|> 对应的 token ID。
terminators = _______________________________________________________________

# 初始化一个空列表，用于存储每次迭代中计算出的每秒生成的tokens数量
times = []
for i in range(1):
    max_length = 256
    # print("========================")
    # print("Iteration", i)
    # print("========================")
    #TODO: 记录开始时间，用于计算生成任务所需的时间
    start_time = _______________________________________________________________
    #TODO: 调用文本生成管道，根据给定的 prompt 和参数生成文本
    outputs = pipeline(
        _______________________________,
        max_new_tokens=_______________________________,
        eos_token_id=_______________________________,
        do_sample=_______________________________,
        temperature=0.6,
        top_p=0.9,
    )
    # print(outputs[0]["generated_text"][len(prompt):])
    #TODO: 记录生成结束时间
    end_time = ________________________________________________
    #TODO: 计算本次生成任务的耗时
    elapsed_time = ________________________________________________
    #TODO: 计算每秒生成的 tokens 数量，作为本次生成任务的吞吐量
    tokens_per_sec = ________________________________________________
    #TODO: 将计算出的每秒生成的 tokens 数量存入 times 列表
    ________________________________________________
    print(f"iter: {i}, Tokens per second: {tokens_per_sec}")

print("========================")
#TODO: 计算并打印平均每秒生成的 tokens 数量
print("Average tokens per second:",  __________________________)
print("========================")

print("INFERSPEED PASS!")
