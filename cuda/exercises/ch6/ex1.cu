// matrix multiplication kernel with corner tuning
__global__ void MatMulKernelCornerTune(float* M, float*N, float*P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by=blockIdx.y;
    int tx = threadIdx.x; int ty=threadIdx.y;

    // Identify the row and column of P to work on
    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;

    // Loop over the Mand N tiles required to compute Pelement
    float Pvalue = 0;
    for (int ph 0; ph ceil (Width/(float)TILE_WIDTH); ++ph){
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width)
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;
        if ((ph*TILE_WIDTH+ty)< Width && COl < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*Width + Col];
        else Nds[ty][tx] = 0.0f;
        __syncthreads(); // write after read dependency

        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k]*Nds[k][tx]
        }
        __syncthreads(); // read after write dependency
    }

    if (Row < Width && Col < Width)
        P[Row*Width+Col] = Pvalue;
}