import torch
import torch_directml
import time

#use torch_directtml to test AMD GPU load

# 初始化DirectML设备，dxdiag
device = torch_directml.device()
print(f"使用设备: {device}")

# 创建中等规模矩阵 (调整这个大小可以控制负载)
matrix_size = 4096 # 增大这个值会增加GPU负载
x = torch.randn(matrix_size, matrix_size, device=device)
y = torch.randn(matrix_size, matrix_size, device=device)

print(f"已创建 {matrix_size}x{matrix_size} 矩阵，开始计算...")

# 进行连续计算（足够观察但不会让GPU满载）
for i in range(30): # 30次迭代足够观察
    start_time = time.time()
    z = torch.matmul(x, y)
    elapsed = time.time() - start_time

    # 打印每次计算耗时（观察计算强度）
    print(f"迭代 {i + 1}: 矩阵乘法耗时 {elapsed:.3f}秒")

# 适度调整矩阵（保持计算可见但不过载）
if i % 10 == 0:
    x = x * 1.1 # 轻微调整矩阵值

print("计算完成！现在可以查看任务管理器的GPU占用情况")