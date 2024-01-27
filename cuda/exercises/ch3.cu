#include <stdio.h>

__global__
void MatrxiMultiplyRow(float *M, float *N, float *P, int Width)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < Width){
    for(int col=0; col < Width; col++){
        float val = 0;
        for(int k=0; k<Width; k++){
            val += M[row*Width + k] * N[k*Width + col];
        }
        P[row*Width + col] = val;
    }
  }
  
}
//  execution parameters: ceil(Width/16), 16 

__global__
void MatrxiMultiplyCol(float *M, float *N, float *P, int Width)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(col < Width){
    for(int row=0; row < Width; row++){
        float val = 0;
        for(int k=0; k<Width; k++){
            val += M[row*Width + k] * N[k*Width + col];
        }
        P[row*Width + col] = val;
    }
  }
  
}
//  execution parameters: ceil(Width/16), 16 

// pros and cons: row-wise access pattern is better

// Q 2: matrix vector multiplication
__global__
void MatrixVectorMultiply(float *B, float *c, float *A, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size){
    float val = 0;
    for(int col=0; col < size; col++){
        val += B[i*size + col] * c[col];
    }
    A[i] = val;
  }
  
}

void matVec(float *B_d, float *c_d, float *A_d){
    int size = 100;
    dim3 dimGrid(ceil(size/16), 1, 1);
    dim3 dimBlock(16, 1, 1);
    MatrixVectorMultiply<<< dimGrid, dimBlock >>>(B_d, c_d, A_d, size);
}

// Q 3: threads per bloc: 16x32
// Q 3: number of threadsin grid = 315x181
// Q 4: row major order 20*400 + 10 = 8010. column major order = 10*500 + 20 = 5020.
// Q 5: row major order 10*120,000  + 20*300 + 5
