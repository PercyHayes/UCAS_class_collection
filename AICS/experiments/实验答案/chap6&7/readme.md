# AICS 实验6&7 - Verilog实现说明

本目录包含AICS课程第6章和第7章相关的Verilog实验代码和仿真文件。

## 📁 文件结构

```
chap6&7/
├── readme.md                # 本说明文件
├── sim_iverilog/           # iVerilog仿真环境
│   ├── *.v                 # Verilog仿真源文件
│   ├── *.sh                # Linux启动脚本
│   └── *.vcd               # 波形文件（仿真输出）
└── 其他实验相关文件/
```
## 💡 使用提示
1. 实验教程提供的modelsim功能强大但安装和使用复杂，对于本实验而言使用轻量化的iverilog配合gtkwave/wavetrace插件足矣
2. 代码在linux环境下使用，已经写好了三个实验的启动命令，使用时注意修改脚本中的路径。windows环境下用GPT重新生成启动脚本即可使用
3. 环境配置可参考下面的介绍


## 🔧 环境配置

### Linux环境（推荐）
```bash
# 安装iVerilog和GTKWave
sudo apt-get update
sudo apt-get install iverilog gtkwave

# 验证安装
iverilog -V
gtkwave --version
```

### Windows环境
1. 下载iVerilog安装包
2. 运行压缩包，选择添加到执行路径，选择下载gtkwave
3. 安装参考教程[https://www.cnblogs.com/quantoublog/articles/18089793](https://www.cnblogs.com/quantoublog/articles/18089793)



## 🚀 运行说明
### Linux/macOS环境
```bash
# 修改脚本路径信息
# 运行仿真脚本
chmod +x *.sh
bash matrix_sim.sh

# 查看波形（如果生成了.vcd文件）
cd sim_iverilog/matrix_sim
gtkwave matrix_pe.vcd
```











---
*最后更新: 2025年6月27日*