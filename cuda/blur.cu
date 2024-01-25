#include <stdio.h>

#define BLUR_SIZE 3

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if(row < h && col < w){
    float val = 0
    for(int i=-(BLUR_SIZE)/2; i<= BLUR_SIZE/2; i++){
        for(int j=-(BLUR_SIZE)/2; j<=BLUR_SIZE/2; j++){
            if(row+i < h and col+j < w){
                val += (unsigned char)in[i*w + j]/(BLUR_SIZE*BLUR_SIZE);
            }
        }
    }
    out[row*w + col] = val
  }
  
}