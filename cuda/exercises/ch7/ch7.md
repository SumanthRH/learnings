1. P[0] = 60
2. 8, 21, 13, 20, 7
3. a. Nothing. x = y  
    b. Left shift.  
    c. Right shift.  
    d. Edge filter  
    e. Blur filter.  
4. a. Assume N is odd. The number of ghost cells is N-1.  
    b. N multiplications of every entry in M. SO M*N.   
    c. M*N - (N-1) (1 for every ghost cell)
5. a. $(M-1) \times (2N+M-1)$
    b. $N\times N\times M\times M$
    c.  $N\times N\times M\times M  - (M-1)\times(2N+M-1)$  
6. a. $(M_2 - 1+ N_2) \times (N_1 + M_1 - 1) - N_2 \times N_2$  
    b. $N_1\times N_2 \times M_1 \times M_2$  
    c. $N_1\times N_2 \times M_1 \times M_2 - [(M_2 - 1+ N_2) \times (N_1 + M_1 - 1) - N_2 \times N_2]$
7. a. Number of thread blocks needed : (M $\times$ M)/(T $\times$ T)    
b. Threads per block = $(T + (N-1)/2) \times (T + (N-1)/2)$   
c.  Shared memory per block = $(T + (N-1)/2) \times (T + (N-1)/2)$  
d. New algorithm:
    1. Number of thread blocks needed:  (M $\times$ M)/(T $\times$ T)   
    2. Threads per block = $(T + (N-1)/2) \times (T + (N-1)/2)$  
    3. Shared memory per block = $T \times T$  
8. TODO
9. TODO
10. TODO