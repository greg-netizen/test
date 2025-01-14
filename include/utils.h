#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

struct Matrix
{
    int rows;
    int cols;
    std::vector<float> data;
    Matrix();
    Matrix(int rows, int cols, std::vector<float> data=std::vector<float>{});
    Matrix(const Matrix& other);
    Matrix operator+(const Matrix &other)const;
    void fromCVMAT(const cv::Mat& mat);
    cv::Mat toCVMAT(int type = CV_8UC1) const;

    static Matrix zeros(int rows, int cols){
        Matrix output(rows,cols);
        for(int i=0;i<rows*cols;i++)
            output.data[i] = 0;

        return output;

    }

};

struct Dense {
    int inputSize;
    int outputSize;
    std::vector<float> input;
    std::vector<float> output;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    Dense();
    Dense(int inSz, int outSz, std::vector<float> input);
    Dense(const Dense& other);

    void forward(std::function<std::vector<float>(std::vector<float>, int)> activationFunction);
    void showOutput() const;
    void showInput() const;
};

struct MaxPooling {
    Matrix input;
    Matrix output;
    int padding;
    int stride;

    MaxPooling(const Matrix& in=Matrix()    , int stride = 1, int padding = 0);
    MaxPooling(const MaxPooling& other);
    void pool(int poolSize = 2);
};

struct Convolution2D {
    Matrix input;
    Matrix output;
    Matrix kernel;
    int padding;
    int stride;

    Convolution2D();
    Convolution2D(const Matrix& in, const Matrix& kernel, int stride = 1, int padding = 0);
    Convolution2D(const Convolution2D& other);
    Convolution2D& operator=(const Convolution2D& other);
    void conv();
};

namespace gpu {
    __device__ void matrixMultiply(const float* a, const float* b, float* c, int rows, int cols);
    __global__ void flatten(const float* const* input, float* output, dim3 inSz);
    __global__ void convolution2D(float* input, const float* kernel, float* output, dim3 inSz, dim3 outSz, dim3 kSz, int stride);
    __global__ void maxPooling2D(const float* input, float* output, dim3 inSz, dim3 outSz, int poolingSize, int stride);
}

namespace aifunc
{
    std::vector<float> softmax(std::vector<float> input, int inSz);
    std::vector<float> relu(std::vector<float> input, int inSz);
}

#endif