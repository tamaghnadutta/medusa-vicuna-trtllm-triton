git clone https://github.com/FasterDecoding/Medusa.git
sudo apt-get update && sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
git clone https://github.com/ctlllll/axolotl.git

cd axolotl
virtualenv env
source env/bin/activate

pip install -e .

pip install tensorflow==2.17.0 huggingface-hub==0.24.6

# If CUDA Driver Version, installed torch's CUDA compat version 
# and nvcc -V are not pointing to the same CUDA Toolkit/Driver, 
# then you will have to follow https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local
# before installing fash-attn
# In ~/.bashrc
# export PATH=/usr/local/cuda-12.4/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# Then restart your VM/Notebook Server

pip install flash-attn

# This training was run on a machine with A100 40GB GPU

# Every 0.1s: nvidia-smi                                                                                                                        experimental-1: Fri Aug 30 17:59:52 2024

# Fri Aug 30 17:59:52 2024
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA A100-SXM4-40GB          On  |   00000000:00:04.0 Off |                    0 |
# | N/A   60C    P0            445W /  400W |   28448MiB /  40960MiB |     98%      Default |
# |                                         |                        |             Disabled |
# +-----------------------------------------+------------------------+----------------------+

# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |    0   N/A  N/A     33244      C   /opt/conda/bin/python3.10                   28436MiB |
# +-----------------------------------------------------------------------------------------+

accelerate launch -m src.axolotl.cli.train examples/medusa/vicuna_7b_stage1.yml


