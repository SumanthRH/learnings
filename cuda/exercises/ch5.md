1. No we can't use shared memory to improve throughput here because memory loads across threads can't be overlapped.
2. (skipped, it's a question on drawing)
3. With the first `__syncthreads()`, you have a read-after-write dependency, and thus an incorrect vlaue might be read by a thread. With the second, you have a write-after-read dependency, and thus one thread can get ahead and corrupt the values in shared memory that were needed by a slower thread.
4. it is still valuable to have variables in shared memory rather than registers. A simple reason is that a case like matrix multiplication has overlapping memory accesses across threads, and register memory is isolated i.e separate for each thread. 
5. The reduction is 32.
6. 1000 thread blocks, 512 threads per block. A local variable in a kernel is stored in the registers - per thread -  and thus you need 512*1000 = 512,000 versions of the variable.
7. If it is in shared memory, then you'll have 1000 versions (one per block)
8. Each is requested N times without tiling. With tiling, it is N/T times.
9. Num ops = 36. Num bytes accessed = 7*4 = 28 B per thread.
Arithmetic intensity $\approx$  1.286.  
    a. Peak FLOPS = 200 GFLOPS. Peak B/W = 100 GB/sec. Max intensity achievable = $200 / 100 = 2 > 1.286$. Thus, the kernel is memory-bound.  
    b. Peak FLOPS = 300 GFLOPS. Peak B/W = 250 GB/sec. Max intensity achievable = $300/250 = 1.2 < 1.286$. Thus, the kernel is compute-bound.
10. a. I think the kernel would only execute correctly for a BLOCK_WIDTH of 1. If the value is 1, then you basically don't have any shared memory across threads, each will do it's job. If BLOCK_WIDTH > 1, then you need synchronization because of a read-after-write dependency.  
b. The fix is to add `__syncthreads()` after line 10. A better way is to avoid shared memory altogether since there's no point in using this for a simple transpose operation.
11. a. 1024 versions of $i$.  
    b. 1024 versions of $x[]$.  
    c. 8 versions of $y\_s$.  
    d. 8 versions of $b\_s[]$.  
    e. Per block, we have 128+1 = 129 variables = 129*4 = 516 B of shared memory.  
    f. Num. of Ops = 10 per thread. Number of global memory accesses = 5 per thread. Thus, ratio = 2 OP/B.
