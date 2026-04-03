import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"PyTorch是否包含CUDA支持: {torch.version.cuda}")

# 检查CUDA是否可用
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"可用GPU数量: {torch.cuda.device_count()}")

# 如果有多个GPU，列出它们
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("没有可用的CUDA设备，将使用CPU")

import os
print(f"当前工作目录：{os.getcwd()}")
print(f"脚本所在目录：{os.path.dirname(os.path.abspath(__file__))}")