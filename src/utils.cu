#include "utils.h"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdexcept>
Matrix::Matrix(){};
Matrix::Matrix(int rows, int cols, std::vector<float>data):rows(rows),cols(cols),data(std::vector<float>(rows*cols)){};
Matrix::Matrix(const Matrix& other):rows(other.rows),cols(other.cols),data(other.data){};

Matrix Matrix::operator+(const Matrix& other) const
{
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows * cols; ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    }


void Matrix::fromCVMAT(const cv::Mat& mat) {
    if (mat.type() != CV_8UC1)
        throw std::invalid_argument("Incorrect matrix type. Matrix must be CV_8UC1!");
    if (mat.empty())
        throw std::invalid_argument("Image cannot be empty!");

    this->rows = mat.rows;
    this->cols = mat.cols;
    this->data = std::vector<float>(rows*cols);

    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            data[row * cols + col] = mat.at<uint8_t>(row, col);
}

cv::Mat Matrix::toCVMAT(int type) const {
    if (data.empty() || rows <= 0 || cols <= 0)
        throw std::invalid_argument("No valid data to create the matrix");

    cv::Mat m(rows, cols, type);
    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            m.at<uint8_t>(row, col) = data[row * cols + col];
    return m;
}

Dense::Dense(){};
Dense::Dense(int inSz, int outSz, std::vector<float> input):inputSize(inSz),outputSize(outSz),input(input)
{
    this->biases = std::vector<float>(outSz);
    for(int i=0;i<outSz;i++)
        this->biases[i] = 0.1;
};

Dense::Dense(const Dense& other):
inputSize(other.inputSize), outputSize(other.outputSize), input(other.input),
output(other.output), weights(other.weights), biases(other.biases){}



void Dense::forward(std::function<std::vector<float>(std::vector<float>, int)> activationFunction) {
    for (int neuron = 0; neuron < outputSize; neuron++) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++)
            sum += weights[neuron][i] * input[i];
        output.push_back(sum + biases[neuron]) ;
    }

    if (activationFunction != nullptr) {
        std::vector<float> activated_output = activationFunction(output, outputSize);
        this->output = activated_output;
    }
}


// MaxPooling Implementations
MaxPooling::MaxPooling(const Matrix& in, int stride, int padding)
    : input(in), output(Matrix(0, 0)), stride(stride), padding(padding) {
    int outputRows = (input.rows - 2 * padding) / stride + 1;
    int outputCols = (input.cols - 2 * padding) / stride + 1;
    output = Matrix(outputRows, outputCols);
}

MaxPooling::MaxPooling(const MaxPooling& other)
    : input(other.input), output(other.output), stride(other.stride), padding(other.padding) {}

void MaxPooling::pool(int poolSize) {
    dim3 inSz(input.rows, input.cols);
    dim3 outSz(output.rows, output.cols);

    float* d_in, *d_out;
    cudaMalloc(&d_in, inSz.x * inSz.y * sizeof(float));
    cudaMalloc(&d_out, outSz.x * outSz.y * sizeof(float));
    cudaMemcpy(d_in, input.data.data(), inSz.x * inSz.y * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu::maxPooling2D<<<numBlocks, threadsPerBlock>>>(d_in, d_out, inSz, outSz, poolSize, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data.data(), d_out, output.rows * output.cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// Convolution2D Implementations
Convolution2D::Convolution2D(){};
Convolution2D::Convolution2D(const Matrix& in, const Matrix& kernel, int stride, int padding)
    : input(in), output(Matrix()), kernel(kernel), stride(stride), padding(padding)
{
    int outputRows = (input.rows - kernel.rows + 2 * padding) / stride + 1;
    int outputCols = (input.cols - kernel.cols + 2 * padding) / stride + 1;
    output = Matrix(outputRows, outputCols);
}

Convolution2D::Convolution2D(const Convolution2D& other)
    : input(other.input), output(other.output), kernel(other.kernel), stride(other.stride), padding(other.padding) {}

Convolution2D& Convolution2D::operator=(const Convolution2D& other) {
    if (this == &other)
        return *this;
    input = other.input;
    output = other.output;
    kernel = other.kernel;
    stride = other.stride;
    padding = other.padding;
    return *this;
}

void Convolution2D::conv() {
    dim3 inSz(input.rows, input.cols);
    dim3 outSz(output.rows, output.cols);
    dim3 kSz(kernel.rows, kernel.cols);

    float* d_in, *d_out, *d_kernel;
    cudaMalloc(&d_in, inSz.x * inSz.y * sizeof(float));
    cudaMalloc(&d_out, outSz.x * outSz.y * sizeof(float));
    cudaMalloc(&d_kernel, kSz.x * kSz.y * sizeof(float));
    cudaMemcpy(d_in, input.data.data(), inSz.x * inSz.y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data.data(), kSz.x * kSz.y * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu::convolution2D<<<numBlocks, threadsPerBlock>>>(d_in, d_kernel, d_out, inSz, outSz, kSz, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data.data(), d_out, output.rows * output.cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
}

// Namespace aifunc Implementations
std::vector<float> aifunc::relu(std::vector<float> input, int inSz) {
    std::vector<float> output(inSz);
    for (int i = 0; i < inSz; ++i)
        output[i] = std::max(0.0f, input[i]);
    return output;
}

std::vector<float> aifunc::softmax(std::vector<float> input, int inSz) {
    float sum = 0.0f;
    float maxElem = *std::max_element(input.begin(), input.end());
    std::vector<float> output(inSz);

    for (int i = 0; i < inSz; ++i)
        sum += std::exp(input[i] - maxElem);

    for (int i = 0; i < inSz; ++i)
        output[i] = std::exp(input[i] - maxElem) / sum;

    return output;
}


__global__ void gpu::convolution2D(float* input, const float* kernel, float* output, dim3 inSz, dim3 outSz, dim3 kSz, int stride) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < 0 || outCol < 0 || outRow >= outSz.x || outCol >= outSz.y)
        return;

    int inRow = outRow * stride;
    int inCol = outCol * stride;

    float sum = 0.0f;
    for (int i = 0; i < kSz.x; ++i) {
        for (int j = 0; j < kSz.y; ++j) {
            int idxRow = inRow + i;
            int idxCol = inCol + j;
            if (idxRow >= 0 && idxRow < inSz.x && idxCol >= 0 && idxCol < inSz.y)
                sum += input[idxRow * inSz.y + idxCol] * kernel[i * kSz.y + j];
        }
    }
    output[outRow * outSz.y + outCol] = fmaxf(0.0f, fminf(255.0f, sum));
}
__global__ void gpu::maxPooling2D(const float* input, float* output, dim3 inSz, dim3 outSz, int poolingSize, int stride) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outRow < 0 || outCol < 0 || outRow >= outSz.x || outCol >= outSz.y)
        return;

    int inRow = outRow * stride;
    int inCol = outCol * stride;

    float maxElem = -FLT_MAX;
    for (int i = 0; i < poolingSize; ++i) {
        for (int j = 0; j < poolingSize; ++j) {
            int idxRow = inRow + i;
            int idxCol = inCol + j;
            if (idxRow >= 0 && idxRow < inSz.x && idxCol >= 0 && idxCol < inSz.y)
                maxElem = fmaxf(maxElem, input[idxRow * inSz.y + idxCol]);
        }
    }
    output[outRow * outSz.y + outCol] = maxElem;
}
