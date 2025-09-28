import torch
print(f"Pytorch version:{torch.__version__}")
print(f"CUDA available:{torch.cuda.is_available()}")

x=torch.randn(3,3)
print(f"tensor device:{x.device}")
print("tensor content:")
print(x)

model = torch.nn.Linear(10,5)
print("model test successful!")
