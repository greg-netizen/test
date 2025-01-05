#include "utils.h"
#include "opencv2/opencv.hpp"

namespace gpu
{
    __device__ void matrixMultiply(const float *a, const float *b, float *c, const int rows, const int cols)
    {
        int row = blockDim.x *blockIdx.x + threadIdx.x;
        int col = blockDim.y *blockIdx.y +threadIdx.y;

        if(col <0 || col >= cols || row >=rows || row<0)
            return;

        int s = 0;
        for(int i=0;i< rows;i++)
        {
            s+= a[row*rows + i] * b[i * rows + col];
        }

        c[row * rows + col] = s;


    }

__global__ void flatten(float* const* input, float* output, const dim3 inSz)
{
    const int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    const int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    const int outDep = blockDim.z * blockIdx.z + threadIdx.z;

    if (outCol >= inSz.x || outRow >= inSz.y || outDep >= inSz.z)
        return;

    output[outDep * inSz.y * inSz.x + outRow * inSz.x + outCol] = (input[outDep])[outRow * inSz.x + outCol];
}


    __global__ void convolution2D(float *input, const float *kernel, float *output, const dim3 inSz, const dim3 outSz, const dim3 kSz, const int stride) {
        const int outRow = blockIdx.y * blockDim.y + threadIdx.y;
        const int outCol = blockIdx.x * blockDim.x + threadIdx.x;

        if(outRow < 0 || outCol < 0 || outRow >= outSz.x || outCol >= outSz.y){
            return;
        }

        const int inRow = outRow * stride;
        const int inCol = outCol * stride;

        float sum = 0.0f;
        for (int i = 0; i < kSz.x; i++) {
            for (int j = 0; j < kSz.y; j++) {
                int inRowIdx = inRow + i;
                int inColIdx = inCol + j;

                if (inRowIdx >= 0 && inRowIdx < inSz.x && inColIdx >= 0 && inColIdx < inSz.y) {
                    sum += input[inRowIdx * inSz.y + inColIdx] * kernel[i * kSz.y + j];
                }
            }
        }

        output[outRow * outSz.y + outCol] = fmaxf(0.0f, fminf(255.0f, sum));
    }

    __global__ void maxPooling2D(const float *input,  float* output, const dim3 inSz, const dim3 outSz,const int poolingSize, const int stride){
        const int outCol = blockDim.x*blockIdx.x + threadIdx.x;
        const int outRow = blockDim.y*blockIdx.y + threadIdx.y;

        if(outRow < 0 || outCol < 0 || outRow >= outSz.x || outCol >= outSz.y){
            return;
        }

        const int inRow = outRow * stride;
        const int inCol = outCol * stride;

        float maxElem = -FLT_MAX;
        for(int i=0;i<poolingSize;i++){
            for(int j = 0; j<poolingSize;j++){
                maxElem = max(maxElem, input[(inRow + i) * inSz.y + (inCol + j)]);
            }
        }

        output[outRow*outSz.y + outCol] = maxElem;


    }
}

