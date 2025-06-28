1、# 进入/opt/code_chap_8_opt_student/目录
    cd /opt/code_chap_8_opt_student/
    #进行环境激活
    source env.sh

2、补全 infer-3b.py、infer_speed_test.py、infer-11b.py，完成原生 Llama3.2 系统构建；

    #进入目录
    cd llama3.2
    #运行原生Llama3.2模型的文本生成
    python infer-3b.py
    #文本推理速度测试
    python infer_speed_test.py
    #运行原生Llama3.2模型的多模态生成
    python infer-11b.py

3、补全finetune_math.sh、finetune_ruozhiba.sh、finetune_latexocr.sh、infer_math.sh 、 infer_ruozhiba.sh、infer_latexocr.sh  文件完成基于 Llama3.2模型的 LoRA 微调与推理。

    # 基于blossom-math-v2的LoRA微调
    bash finetune_math.sh
    # 基于弱智吧的LoRA微调
    bash finetune_ruozhiba.sh
    # 基于latex-ocr-print的LoRA微调
    bash finetune_latexocr.sh
    # 基于blossom-math-v2的LoRA推理
    bash infer_math.sh
    # 基于弱智吧的LoRA推理
    bash infer_ruozhiba.sh
    # 基于latex-ocr-print的LoRA推理
    bash infer_latexocr.sh


4、完成基于寒武纪 Triton 的 Flash attention 算子封装，补全 llama3.2/flash_attention_triton_opt.py 文件，构建起前向传播模块、
反向传播模块、高效注意力机制模块，并完成单算子测试。

    #进入目录
    cd llama3.2
    #完成算子封装并进行单算子测试
    python flash_attention_triton_opt.py

5、完成 llama3.2 模型中 Flash Attention 算子的替换，修改/opt/tools/nativate/transformers_mlu/src/transformers/models/mllama/modeling_mllama.py文件。


6、再次运行 Llama3.2的系统构建及推理测试程序，验证在 DLP 平台上替换 Flash Attention算子后 Llama3.2的正确性，并进行基于LoRA的微调和推理；

    #进入目录
    cd llama3.2
    #运行原生Llama3.2模型的文本生成
    python infer-3b.py
    #文本推理速度测试
    python infer_speed_test.py
    #运行原生Llama3.2模型的多模态生成
    python infer-11b.py
    # 基于blossom-math-v2的LoRA微调
    bash finetune_math.sh
    # 基于弱智吧的LoRA微调
    bash finetune_ruozhiba.sh
    # 基于latex-ocr-print的LoRA微调
    bash finetune_latexocr.sh
    # 基于blossom-math-v2的LoRA推理
    bash infer_math.sh
    # 基于弱智吧的LoRA推理
    bash infer_ruozhiba.sh
    # 基于latex-ocr-print的LoRA推理
    bash infer_latexocr.sh
