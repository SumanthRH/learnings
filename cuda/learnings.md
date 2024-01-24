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
- FLOPS don't matter (for most cases). Look at memory bandwidth.
- A100: 1555 GB / sec (HBM2 memory bandwidth)
- Each Sm in A100 can request 64 Bytes per clock cycle
- Ratio of B/W requested / provided =  9750 / 1555 = 6.3x
- Random Access Memory: 
    - Row decoder and column decoder. Activate ro w and pull data in to sense amplifiers. This destroys data in the row as capacitors drain.
    - Column decoder will read from the sense amplifier and you can read repeatedly from the sense amplifier because they can hold their voltage.
    - You can use this in "burst" mode. 
    - Before a new row(page) is fetched, you have a write back step since the sense amplifier has to restore the state of the capacitors. 
    - You are limited by physics with data reading: youcan only read as fasts as you can charge capacitors back up, etc.
- Memory access pattern matters for memory bandwidth:
    - With the largeest stride size between two reads, you get a speed of 111 GB/sec vs 1418 GB/sec (only 8% of peak bandwidth)
    - Thus, the biggest optimization you can do is handling memory access patterns. No other optimization comes close.
- "The reason you're using a GPU is for performance. High performance means being able to use all the GPU resources you can, which means paying attention to memory access patterns"