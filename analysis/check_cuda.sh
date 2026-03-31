#!/bin/bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
"
