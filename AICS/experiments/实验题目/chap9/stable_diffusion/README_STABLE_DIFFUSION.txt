1、# 进入/opt/code_chap_8_opt_student/目录
    cd /opt/code_chap_8_opt_student/
    #进行环境激活
    source env.sh

2、补全stable_diffusion/ldm/models/diffusion/ddim.py、stable_diffusion/scirpts/img2img.py、stable_diffusion/scirpts/txt2img.py、
stable_diffusion/scirpts/gradio/inpainting.py文件完成图生图、文生图以及图像修复一系列功能。

    #进入目录
    cd stable_diffusion
    #运行图生图推理程序
    bash run-stable-diffusion-img2img.sh
    #运行文生图推理程序
    bash run-stable-diffusion-txt2img.sh
    #运行图像修复推理程序
    bash run-stable-diffusion-painting.sh

3、完成基于寒武纪 Triton 的 Flash attention 算子封装，补全 stable_diffusion/flash_attention_triton_opt.py 文件，构建起前向传播模块、
反向传播模块、高效注意力机制模块，并完成单算子测试。

    #进入目录
    cd stable_diffusion
    #完成算子封装并进行单算子测试
    python flash_attention_triton_opt.py

4、完成 Stable Diffusion 模型中 Flash Attention 算子的替换，修改完 stable_diffusion/ldm/modules/attention.py 文件。

5、再次运行 Stable Diffusion 的模型构建程序，从而验证在 DLP 平台上替换 Flash Attention 算子后 Stable Diffusion 的正确性。

    #进入目录
    cd stable_diffusion
    #运行图生图推理程序
    bash run-stable-diffusion-img2img.sh
    #运行文生图推理程序
    bash run-stable-diffusion-txt2img.sh
    #运行图像修复推理程序
    bash run-stable-diffusion-painting.sh
