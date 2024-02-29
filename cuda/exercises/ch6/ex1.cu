
#include <stdio.h>

#define TILE_WIDTH 16
// matrix multiplication kernel with corner tuning
__global__ void MatMulKernelCornerTune(float* M, float*N, float*P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by=blockIdx.y;
    int tx = threadIdx.x; int ty=threadIdx.y;

    // Identify the row and column of P to work on
    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;
    int CornerTuneCol = bx*TILE_WIDTH + ty; //col value to use for corner tuning

    // Loop over the Mand N tiles required to compute Pelement
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph){
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width)
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;
        // loading matrix B with corner tuning: exchange roles of threadIdx.x and threadIdx.y.
        // for a snapshot in time, for threads with the same 
        if ((ph*TILE_WIDTH+tx)< Width && CornerTuneCol < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH+tx)*Width + CornerTuneCol];
        else Nds[ty][tx] = 0.0f;
        __syncthreads(); // write after read dependency

        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k]*Nds[k][tx];
        }
        __syncthreads(); // read after write dependency
    }

    if (Row < Width && Col < Width)
        P[Row*Width+Col] = Pvalue;
}

int main(void){
  int Width = 1<<10;
  printf("Initialization: Matrix Numel %d\n", Width*Width);
  float *x, *y, *d_x, *d_y, *d_P, *P;
  x = (float*)malloc(Width*Width*sizeof(float));
  y = (float*)malloc(Width*Width*sizeof(float));
  P = (float*)malloc(Width*Width*sizeof(float));

  cudaMalloc(&d_x, Width*Width*sizeof(float)); 
  cudaMalloc(&d_y, Width*Width*sizeof(float));
  cudaMalloc(&d_P, Width*Width*sizeof(float));

  for (int i = 0; i < Width*Width; i++) {
    x[i] = 2.0f;
    y[i] = 2.0f;
  }

  cudaError_t err = cudaMemcpy(d_x, x, Width*Width*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
  err = cudaMemcpy(d_y, y, Width*Width*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
  printf("Initialization done\n");
  // Perform Matmul
  dim3 dimGrid(Width/TILE_WIDTH, Width/TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  MatMulKernelCornerTune<<<dimGrid, dimBlock>>>(d_x, d_y, d_P, Width);
  printf("Multiplication done\n"); 
  err = cudaMemcpy(P, d_P, Width*Width*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
  float maxError = 0.0f;
  for (int i = 0; i < Width*Width; i++)
    maxError = max(maxError, abs(P[i]-4.0*Width));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_P);
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
  free(x);
  free(y);
  free(P);
}