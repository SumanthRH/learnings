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
    for(int col=0; col < Width; col++){
        float val = 0;
        val += B[row*Width + col] * c[col];
    }
    A[row*Width + col] = val;
  }
  
}

void matVec(float *B_d, float *c_d, float *A_d){
    int size = 100;
    dim3 dimGrid(ceil(size/16), 1, 1);
    dim3 dimBlock(16, 1, 1);
    MatrixVectorMultiply<<< dimGrid, dimBlock >>>(B_d, c_d, A_d, size);
}
