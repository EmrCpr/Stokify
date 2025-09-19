import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # True 
print(f"GPU name: {torch.cuda.get_device_name(0)}")  # RTX 2060 SUPER 
print(f"CUDA version: {torch.version.cuda}")  # 12.1  #Cuda versiyon kontrolu için açılmış script