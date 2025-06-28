1、# 进入/opt/code_chap_8_opt_student/目录
    cd /opt/code_chap_8_opt_student/
    #进行环境激活
    source env.sh

2、补全 llama_mlu/model.py 文件、llama_mlu/generation.py 文件、example_completion_mlu.py文件、example_infilling_mlu.py文件以及
example_instructions_mlu.py文件，完成自定义 Code Llama 推理系统构建；

    #进入目录
    cd codellama
    # 执行run_completion.sh文件
    bash run_completion.sh
    # 执行run_infilling.sh文件
    bash run_infilling.sh
    # 执行run_instruction.sh文件
    bash run_instruction.sh

3、补全 Codellama-infer.py 文件完成基于 Transformers 库的 CodeLlama 推理系统构建。

    # 执行run-cll.sh文件
    bash run-cll.sh

4、完成基于寒武纪 Triton 的 Flash attention 算子封装，补全 Codellama/flash_attention_triton_opt.py 文件，构建起前向传播模块、
反向传播模块、高效注意力机制模块，并完成单算子测试。

    #进入目录
    cd codellama
    #完成算子封装并进行单算子测试
    python flash_attention_triton_opt.py

5、完成 Codellama 模型中 Flash Attention 算子的替换，修改/opt/tools/native/transformers_mlu/src/transformers/models/llama/modeling_llama.py文件。


6、再次运行相关程序验证在 DLP 平台上替换 Flash Attention 算子后 CodeLlama 的正确性。
   
    # 执行run_completion.sh文件
    bash run_completion.sh
    # 执行run_infilling.sh文件
    bash run_infilling.sh
    # 执行run_instruction.sh文件
    bash run_instruction.sh
    # 执行run-cll.sh文件
    bash run-cll.sh
	
