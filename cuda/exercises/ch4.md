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
    e. Here, the number of iterations is 3, 4 or 5, depending on the value of $i\%3$.  
        i. Ans: 3  
        ii. Ans: 2
