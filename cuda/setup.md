# CUDA Setup

Okay, so after struggling a bit, here is a straightforward way to have CUDA + torch in your machine:

1. Install CUDA via `conda`:

`conda install cuda -c nvidia/label/cuda-\<version>`

Replace version with a version compatible with your installed NVIDIA drivers. (in my case, it was 12.0.1)

2. Verify installation with `nvcc`: Make sure `which nvcc` points to the cuda compiler installed in your conda environment. Also, run `nvcc -V` to make sure the installation was successful.

3. Install torch in the same environment via `conda` or `pip`: This one can be done via conda or pip, and I've found that it works either way:

`pip3 install torch torchvision torchaudio`

PyTorch will figure out right wheels for torch (based on the cuda version) and install torch in your environment. Verify with:

`python -c "import torch; print(torch.cuda.is_available())"`