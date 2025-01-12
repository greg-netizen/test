#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

struct Matrix {
    int rows = 0;
    int cols = 0;
    float* data;
    Matrix();
    Matrix(int rows, int cols , const float* data);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    void fromCVMAT(const cv::Mat& mat);
    cv::Mat toCVMAT(int type = CV_8UC1) const;
    Matrix& operator+=(const Matrix& other);
    static Matrix zeros(int rows, int cols);

    void show() const;
};

struct Dense {
    int inputSize;
    int outputSize;
    float* input;
    float* output;
    float** weights;
    float* biases;

    Dense();
    Dense(int inSz, int outSz, const float* input = nullptr);
    Dense(const Dense& other);
    Dense& operator=(const Dense& other);
    ~Dense();

    void initWeights(float** wghts);
    void forward(std::function<float*(float*, int)> activationFunction = nullptr);
    void showOutput() const;
    void showInput() const;
};

struct MaxPooling {
    Matrix input;
    Matrix output;
    int padding;
    int stride;

    MaxPooling();
    MaxPooling(const Matrix& in, int stride = 1, int padding = 0);
    MaxPooling(const MaxPooling& other);
    MaxPooling& operator=(const MaxPooling& other);
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

namespace aifunc {
    float* softmax(float* input, int inSz);
    float* relu(float* input, int inSz);
}

#endif // UTILS_H