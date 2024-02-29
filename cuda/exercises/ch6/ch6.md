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