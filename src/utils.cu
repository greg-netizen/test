#include "utils.h"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdexcept>

Matrix::Matrix(): data(nullptr){};
// Matrix Implementations
Matrix::Matrix(int rows, int cols, const float* matrixData=nullptr)
    : rows(rows), cols(cols) {
    data = new float[rows * cols];
    if (matrixData != nullptr)
        std::copy(matrixData, matrixData + rows * cols, data);
}

Matrix::Matrix(const Matrix& other)
    : rows(other.rows), cols(other.cols), data(new float[rows * cols]) {
    std::copy(other.data, other.data + rows * cols, data);
}

Matrix& Matrix::operator=(const Matrix& m) {
    if (this == &m)
        return *this;

    delete[] data;
    rows = m.rows;
    cols = m.cols;
    data = new float[rows * cols];
    std::copy(m.data, m.data + rows * cols, data);
    return *this;
}

Matrix::~Matrix() {
    delete[] data;
}

void Matrix::fromCVMAT(const cv::Mat& mat) {
    if (mat.type() != CV_8UC1)
        throw std::invalid_argument("Incorrect matrix type. Matrix must be CV_8UC1!");
    if (mat.empty())
        throw std::invalid_argument("Image cannot be empty!");

    rows = mat.rows;
    cols = mat.cols;
    delete[] data;
    data = new float[rows * cols];

    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            data[row * cols + col] = mat.at<uint8_t>(row, col);
}

cv::Mat Matrix::toCVMAT(int type) const {
    if (data == nullptr || rows <= 0 || cols <= 0)
        throw std::invalid_argument("No valid data to create the matrix");

    cv::Mat m(rows, cols, type);
    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            m.at<uint8_t>(row, col) = data[row * cols + col];
    return m;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for addition.");

    for (int i = 0; i < rows * cols; ++i)
        data[i] += other.data[i];

    return *this;
}

Matrix Matrix::zeros(int rows, int cols) {
    Matrix mat(rows, cols);
    std::fill(mat.data, mat.data + rows * cols, 0.0f);
    return mat;
}

void Matrix::show() const {
    if (data == nullptr) {
        std::cout << "nullptr";
        return;
    }

    for (int i = 0; i < rows; i++) {
        std::cout << std::endl;
        for (int j = 0; j < cols; j++)
            std::cout << data[i * cols + j] << " ";
    }
}


// Dense Implementations
Dense::Dense(): inputSize(0), outputSize(0), input(nullptr), output(nullptr), weights(nullptr), biases(nullptr)
{
};
Dense::Dense(int inSz, int outSz, const float* input)
    : inputSize(inSz), outputSize(outSz) {
    input = new float[inputSize];
    output = new float[outputSize];
    weights = new float*[outputSize];
    biases = new float[outputSize];

    for (int i = 0; i < outputSize; ++i)
        weights[i] = new float[inputSize];

    if (input)
        std::memcpy(this->input, input, inputSize * sizeof(float));

    std::fill(biases, biases + outputSize, 0.0f);
    std::fill(output, output + outputSize, 0.0f);
}

Dense::Dense(const Dense& other)
    : inputSize(other.inputSize), outputSize(other.outputSize) {
    input = new float[inputSize];
    std::memcpy(input, other.input, inputSize * sizeof(float));

    output = new float[outputSize];
    std::memcpy(output, other.output, outputSize * sizeof(float));

    weights = new float*[outputSize];
    for (int i = 0; i < outputSize; ++i) {
        weights[i] = new float[inputSize];
        std::memcpy(weights[i], other.weights[i], inputSize * sizeof(float));
    }

    biases = new float[outputSize];
    std::memcpy(biases, other.biases, outputSize * sizeof(float));
}

Dense& Dense::operator=(const Dense& source) {
    if (this == &source)
        return *this;

    delete[] input;
    delete[] output;
    delete[] biases;
    for (int i = 0; i < outputSize; ++i)
        delete[] weights[i];
    delete[] weights;

    inputSize = source.inputSize;
    outputSize = source.outputSize;

    input = new float[inputSize];
    std::memcpy(input, source.input, inputSize * sizeof(float));

    output = new float[outputSize];
    std::memcpy(output, source.output, outputSize * sizeof(float));

    weights = new float*[outputSize];
    for (int i = 0; i < outputSize; ++i) {
        weights[i] = new float[inputSize];
        std::memcpy(weights[i], source.weights[i], inputSize * sizeof(float));
    }

    biases = new float[outputSize];
    std::memcpy(biases, source.biases, outputSize * sizeof(float));

    return *this;
}

void Dense::initWeights(float** wghts) {
    for (int i = 0; i < outputSize; i++)
        biases[i] = 0.1f;

    for (int i = 0; i < outputSize; i++)
        std::copy(wghts[i], wghts[i] + inputSize, weights[i]);
}

void Dense::forward(std::function<float*(float*, int)> activationFunction) {
    for (int neuron = 0; neuron < outputSize; neuron++) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++)
            sum += weights[neuron][i] * input[i];
        output[neuron] = sum + biases[neuron];
    }

    if (activationFunction != nullptr) {
        float* activated_output = activationFunction(output, outputSize);
        std::copy(activated_output, activated_output + outputSize, output);
        delete[] activated_output;
    }
}

Dense::~Dense() {
    delete[] input;
    delete[] output;
    delete[] biases;
    for (int i = 0; i < outputSize; ++i)
        delete[] weights[i];
    delete[] weights;
}

// MaxPooling Implementations
MaxPooling::MaxPooling(): padding(0), stride(1){};
MaxPooling::MaxPooling(const Matrix& in, int stride, int padding)
    : input(in), output(Matrix(0, 0)), stride(stride), padding(padding) {
    int outputRows = (input.rows - 2 * padding) / stride + 1;
    int outputCols = (input.cols - 2 * padding) / stride + 1;
    output = Matrix(outputRows, outputCols);
}

MaxPooling::MaxPooling(const MaxPooling& other)
    : input(other.input), output(other.output), stride(other.stride), padding(other.padding) {}

MaxPooling& MaxPooling::operator=(const MaxPooling& other) {
    if (this == &other)
        return *this;
    input = other.input;
    output = other.output;
    stride = other.stride;
    padding = other.padding;
    return *this;
}

void MaxPooling::pool(int poolSize) {
    dim3 inSz(input.rows, input.cols);
    dim3 outSz(output.rows, output.cols);

    float* d_in, *d_out;
    cudaMalloc(&d_in, inSz.x * inSz.y * sizeof(float));
    cudaMalloc(&d_out, outSz.x * outSz.y * sizeof(float));
    cudaMemcpy(d_in, input.data, inSz.x * inSz.y * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu::maxPooling2D<<<numBlocks, threadsPerBlock>>>(d_in, d_out, inSz, outSz, poolSize, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_out, output.rows * output.cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// Convolution2D Implementations
Convolution2D::Convolution2D(): padding(0), stride(1){};
Convolution2D::Convolution2D(const Matrix& in, const Matrix& kernel, int stride, int padding)
    : input(in), kernel(kernel), stride(stride), padding(padding) {
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
    cudaMemcpy(d_in, input.data, inSz.x * inSz.y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data, kSz.x * kSz.y * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu::convolution2D<<<numBlocks, threadsPerBlock>>>(d_in, d_kernel, d_out, inSz, outSz, kSz, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_out, output.rows * output.cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
}

// Namespace aifunc Implementations
float* aifunc::relu(float* input, int inSz) {
    float* output = new float[inSz];
    for (int i = 0; i < inSz; ++i)
        output[i] = std::max(0.0f, input[i]);
    return output;
}

float* aifunc::softmax(float* input, int inSz) {
    float sum = 0.0f;
    float maxElem = *std::max_element(input, input + inSz);
    float* output = new float[inSz];

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
