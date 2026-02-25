import torch

if torch.cuda.is_available():
    print(f"CUDA доступна, найдено {torch.cuda.device_count()} GPU.")
    print(f"Имя текущего устройства: {torch.cuda.get_device_name(0)}")
else:
    print("GPU не обнаружено.")

