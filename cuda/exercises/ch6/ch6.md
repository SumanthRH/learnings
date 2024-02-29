1. Solved in [ex1.cu](/cuda/exercises/ch6/ex1.cu).
2. (Not sure) For TILE_WIDTH > 1, it is always better to coalesce adjacent memory requests.
3.  a. Coalesced  
    b. Not applicable (shared memory is SRAM)  
    c. Coalesced  
    d. Coalesced (I think? For consecutive threads, the array indices are off by 4, but that can still fall in the same channel.)  
    e. Not applicable (shared memory)  
    f. Not applicable (shared memory)  
    g. Coalesced (but write, instead of read)  
    f. Not applicable (shared memory)  
    i. Coalesced
4. 
    a. For the simple kernel, we have one addition and multiplication operation for 2 4-byte accesses (one element from M and N each). Thus, the ratio is 2/8 = 0.25 OP/B  
    b. With a tile size of 32x32, the total number of bytes accessed reduces by a factor of 32 (since each thread block will now only access one element in a 32xc32 block once). Thus, the ratio is 0.25*32 = 8 OP/B. With thread coarsening, the number of accesses for M reduces by a factor of COARSE_FACTOR (4). In this case, we still have the same amount of accesses for matrix N. Calculating the net reduction in the number of accesses, we get a factor of 5/8. Thus, the ratio is 8/(5/8) = 64/5 = 12.8 OP/B.
