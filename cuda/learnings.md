# How GPU Computing Works | GTC 2021
- The compute ratio: max FLOPS / Memory BW.
    - How many ops you need to
be doing to
make sure processing units 
are not idle.

- The GPU is a throughput 
machine: you have many more 
threads available than 
dictated by compute ratio and DRAM memory latency

- The number of registers you have dictates the number of threads you can run.

- GPUs have large register count that have higher latency than CPU

- Moving data across the 
PCI bus is 3x as very slow 
one of biggest bottleneck

- NVlink is closer to HBM latency than PCIe!
![nvlink](images/nvlink_matters.png)

- A100 - 108 S streaming multiprocessor.

    Each SM: 64 warps/SM.

    In each clock cycle, you have 4 warps doing something.

- The GPU can switch between 
warps back-to-back.
- Essentially zero cost of context 
switch.
- It's important to have 
more threads than you can 
run.

- GPU vs CPU is like Train vs Car
    - Trains being full as essential
    - It's good to have people
    waiting at the station.
    - Oversubscription means 
    keeping the GPU busy.
    - GPU/CPU asynchrony is 
    essential.

- Matmul has an arithmetic
intensity of O(N).

![compute intensity](images/compute_intensity.png)

- **Tensor Cores**: Custom hardware units built into the SM. They can do a matrix multiplication in one go.
    - FMA does 2 FLOPs per instruction. Tensor cores do way more.
    - You need a much bigger matrix to match compute intensity.

![where's data](images/wheres_my_data.png)

# How CUDA Programming Works | GTC 2022
- FLOPS don't matter (for most cases). Look at memory bandwidth