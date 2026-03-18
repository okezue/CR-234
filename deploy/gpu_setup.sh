#!/bin/bash
set -e
sudo dnf install -y -q tmux htop 2>/dev/null || sudo yum install -y -q tmux htop 2>/dev/null || true
/opt/pytorch/bin/pip install --quiet pandas tqdm tensorboard matplotlib 2>/dev/null || true
mkdir -p /home/ec2-user/cr234
echo "GPU setup complete"
nvidia-smi || echo "No GPU detected"
/opt/pytorch/bin/python3 -c "import torch;print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
