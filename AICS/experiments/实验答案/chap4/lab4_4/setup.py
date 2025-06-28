from setuptools import setup
from torch.utils import cpp_extension
import os

setup(
    #TODO: 给出编译后的链接库名称
    #______________________________________
    name='hsigmoid_extension.so',
    ext_modules=[
        cpp_extension.CppExtension(
    #TODO：以正确的格式给出编译文件即编译函数
    #______________________________________
        'hsigmoid_extension',
        ['hsigmoid.cpp']
        )
    ],
    # 执行编译命令设置
    cmdclass={						       
        'build_ext': cpp_extension.BuildExtension
    }
)
print("generate .so PASS!\n")

# 移动路径并重命名
directory = './'
# 获取当前脚本所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".so"):
        new_filename = "hsigmoid_extension.so"
        os.rename(os.path.join(directory, filename), os.path.join(os.path.dirname(current_dir), new_filename))
