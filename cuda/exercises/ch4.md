# Chapter 4 exercises
1. Q1: block size = 128, num blocks = 8  
    a. Ans: 128/32 = 4 warps per block  
    b. Ans: 4*8 = 32 warps in the grid  
    c.     
            i. Ans: 2 warps per block * 8 blocks = 16 warps  
            ii. Ans: 4 - 2 = 2 warps per block. Total = 16 warps.  
            iii. Ans: 100%. SIMD efficiency for an instruction = % of assigned threads executing an instruction. for warp 0, all threads are active, so the answer is 100%.  
            iv. Ans: 25%. Only 8 threads (32-39) are active in warp 1 (32-63).  
            v. Ans: 75%. 24 threads (104-127) are active in warp 3 (96-127).  
    d.   
            i. Ans: All warps are active.  
            ii. Ans: All warps are divergent. (thread divergence seen in all warps)  
            iii. Ans: 50%.  
    e. Here, the number of iterations is 3, 4 or 5, depending on the value of $i\\%3$.  
            i. Ans: 3  
            ii. Ans: 2
2. Vector length = 2000. Block size = 512, so we need to have  $512 * \lceil \frac{2000}{512}\rceil = 2048$ threads or 64 warps.
3. Due to the boundary check on vector length, 48 threads will show divergence. Thus, 2 warps (warp size is 32) or about 3\% of warps show divergence.
4. I think the question can be rephrased as "On average, what is the percentage of time spent waiting for the barrier?" The execution time is 3.0 seconds (the longest execution time). The times are 2.0, 2.3, 3.0, 2.8,2.4, 1.9, 2.6, 2.9. The percentage is $\dfrac{1 + 0.7 + 0.0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1}{3\times 8} = 17\\%$
5. This is of course not a good idea. You would still want to have a synchronization operation in general. If the block size is equal to the warp size, then of course all the threads are scheduled to be executed at the same time and for the same instruction. However, there can be minor differences in thread latencies simply based on the nature of data each thread is operating on. There can also be differences in cases of control divergence where one set of threads might take longer time even within a warp. 
6. the answer is c. 512 threads per block.   
7.    
    a. Possible. The occupancy is 1024/2048 = 50\%.  
    b. Possible. The occupancy is 50\% again.  
    c. Possible. The occupancy is 50\%.
    d. Possible. The occupancy is 100\%.
    e. Possible. The occupancy is 100\% again.
8. a. Possible. You'll need 16 blocks and 61440 registers in the SM for full occupancy, which is available.   
    b. Not possible. You'll need 2048/32 = 64 blocks per SM to achieve full occupancy, but the maximum is 32 blocks.  
    c. Not possible. You'll need $34\times 2048 = 69632$ registers per SM but only 65,536 registers are available.
9. One would be surprised to hear this because the given configuration is not possible. If you're using 32x32 thread blocks for multiplying the given matrices, then you'll need 1024 threads per block to be supported. However, the SM can support a maximum of 512 threads per block. 

