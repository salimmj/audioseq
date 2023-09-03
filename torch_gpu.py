import torch

# Check for CUDA support
if torch.cuda.is_available():
    print(f'GPUs available: {torch.cuda.device_count()}')
    print(f'GPU Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')
else:
    print('No GPUs available.')

