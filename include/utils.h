#ifndef UTILS_H
#define UTILS_H
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


struct Matrix;
struct Convolution2D;
struct MaxPooling;
struct Dense;

namespace gpu
{
    __device__ void matrixMultiply(const float* a, const float* b, float *c, const int rows, const int cols);
    __global__ void flatten( float* const* input, float *output, const dim3 inSz);
    __global__ void convolution2D(float *input, const float *kernel, float *output, const dim3 inSz, const dim3 outSz, const dim3 kSz, const int stride);
    __global__ void maxPooling2D(const float *input,  float* output, const dim3 inSz, const dim3 outSz,const int poolingSize, const int stride);
}

namespace aifunc
{
    float* softmax(float* input, int inSz);
}



struct Dense
{
    int inputSize;
    float *input;

    int outputSize;
    float *output;

    float **weights;
    float *biases;
public:
    Dense(int inSz, int outSz);
    void initWeights();
    void forward(std::function<float*(float*,int)> activationFunction = nullptr);
    ~Dense();
};

struct Matrix{
    int rows = 0;
    int cols = 0;
    float* data = nullptr;
    Matrix(const int rows, const int cols);

    Matrix& operator=(const Matrix&);

    void fromCVMAT(cv::Mat m);
    cv::Mat toCVMAT(int type );

    void pad(int padSize);
    void normalize();

    void show() const;

    ~Matrix();

};
struct MaxPooling{
    Matrix input;
    Matrix output;
    int padding;
    int stride;

    MaxPooling(const Matrix& in, const int stride=1,const int padding=0);

    void pool(const int poolSize = 2 );

};

struct Convolution2D{
    Matrix input;
    Matrix output;
    Matrix kernel;
    int padding;
    int stride;

    Convolution2D(const Matrix& in, const Matrix& kernel, const int stride = 1, const int padding = 0);

    void conv();

};





inline Matrix::Matrix(const int rows, const int cols):rows(rows),cols(cols)
{
    this->data = new float[rows*cols];

}

inline void Matrix::fromCVMAT(cv::Mat m)
{
    if(m.type() != CV_8UC1){
        throw std::invalid_argument("Incorrect matrix type. Matrix must be CV_8UC1!");
    }
    if(m.empty())
    {
        throw std::invalid_argument("Image cannot be empty!");
    }

    this->cols = m.cols;
    this->rows = m.rows;

    delete []this->data;
    this->data = new float[m.total()];

    for(int row=0; row<m.rows;row++)
        for(int col=0; col<m.cols;col++)
            this->data[row*this->cols + col] = m.at<uint8_t>(row,col);

}

inline cv::Mat Matrix::toCVMAT(int type = CV_8UC1)
{
    if(this->data == nullptr || this->rows<= 0 || this->cols <= 0)
    {
        throw std::invalid_argument("No valid data to create the matrix");
    }

    cv::Mat m = cv::Mat(this->rows, this->cols, type);

    for(int row = 0; row < this->rows; row++){
        for(int col = 0; col < this->cols; col++){
            m.at<uint8_t>(row,col) = this->data[row*this->cols+col];
        }
    }
    return m;


}

inline void Matrix::pad(int padSize = 1) {
    int padded_rows = this->rows + 2 * padSize;
    int padded_cols = this->cols + 2 * padSize;
    Matrix padded_a = Matrix(padded_rows, padded_cols);
    for (int i = 0; i < padded_rows; i++) {
        for (int j = 0; j < padded_cols; j++) {
            if (i >= padSize && i < padSize + this->rows && j >= padSize && j < padSize + this->cols) {
                padded_a.data[i * padded_cols + j] = this->data[(i - padSize) * this->cols + (j - padSize)];
            } else {
                padded_a.data[i * padded_cols + j] = 0;
            }
        }
    }
    *this = padded_a;
}

inline Matrix& Matrix::operator=(const Matrix& m)
{
    if (this == &m) {
        return *this;
    }

    delete[] this->data;

    this->rows = m.rows;
    this->cols = m.cols;
    this->data = new float[m.rows * m.cols];
    for (int i = 0; i < m.rows * m.cols; ++i) {
        this->data[i] = m.data[i];
    }

    return *this;
}

inline void Matrix::show() const
{
    if(this->data == nullptr)
    {
        std::cout<<"nullptr";
        return ;
    }

    for(int i=0;i<this->rows;i++)
    {   std::cout<<std::endl;
        for(int j=0; j<this->cols; j++)
        {
            std::cout<<this->data[i*this->cols+j]<<" ";
        }
    }
}

inline Matrix::~Matrix(){
    delete[] this->data;
}

inline Dense::Dense(int inSz, int outSz):inputSize(inSz),outputSize(outSz)
{
    this->input = new float[inSz];
    this->output = new float[outSz];

    this->biases = new float[outSz];
    this->weights = new float*[outSz];

    for(int i=0;i<outSz;i++)
        this->weights[i] = new float[inputSize];
};

    inline void Dense::initWeights(){
    for(int i =0;i< this->outputSize;i++)
        this->biases[i] = i;

    for(int i=0;i<this->outputSize;i++)
        for(int j=0;j<this->inputSize;j++)
            this->weights[i][j] = j;
}

inline void Dense::forward(std::function<float*(float*,int)> activationFunction)
    {
        for(int neuron = 0; neuron<this->outputSize;neuron++){
            int sum = 0;
                for(int i=0;i<this->inputSize;i++)
                    {
                        sum+=this->weights[neuron][i]*this->input[i];
                    }
            this->output[neuron] = sum + this->biases[neuron];

    }

        if(activationFunction!=nullptr)
            this->output = activationFunction(this->output,this->outputSize);

}



inline Dense::~Dense()
{
    delete []this->input;
    delete []this->output;
    delete []this->biases;

    for(int i= 0; i<this->outputSize;++i)
    {
        delete []this->weights[i];
    }
    delete[] this->weights;

}


MaxPooling::MaxPooling(const Matrix& in, const int stride,const int padding): input(in), output(Matrix(0,0)), stride(stride),
                                                                      padding(padding){
        int outputRows = (input.rows + 2*padding) / stride + 1;
        int outputCols = (input.cols + 2*padding) / stride + 1;

        this->output = Matrix(outputRows, outputCols);
    };

inline void MaxPooling::pool(const int poolSize){
    if(this->padding != 0)
        this->input.pad(this->padding);

    const dim3 inSz = dim3(this->input.rows, this->input.cols);
    const dim3 outSz = dim3(this->output.rows, this->output.cols);

    int e = 0;
    float *d_in, *d_out;

    e=cudaMalloc(&d_in, inSz.x * inSz.y * sizeof(float));
    e|=cudaMalloc(&d_out, outSz.x * outSz.y * sizeof(float));

    e|=cudaMemcpy(d_in, this->input.data, inSz.x * inSz.y * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gpu::maxPooling2D<<<numBlocks, threadsPerBlock>>>(d_in, d_out, inSz, outSz, poolSize, stride);
    e|=cudaDeviceSynchronize();

    e|=cudaMemcpy(this->output.data, d_out, this->output.cols * this->output.rows * sizeof(float), cudaMemcpyDeviceToHost);

    e|=cudaFree(d_in);
    e|=cudaFree(d_out);


    assert(e == 0);
}

inline Convolution2D::Convolution2D(const Matrix& in, const Matrix& kernel, const int stride, const int padding): input(in), output(Matrix(0,0)), kernel(kernel), stride(stride), padding(padding)
{
    int outputRows = (input.rows - kernel.rows + 2*padding) / stride + 1;
    int outputCols = (input.cols - kernel.cols + 2*padding) / stride + 1;

    this->output = Matrix(outputRows, outputCols);
};
inline void Convolution2D::conv()
{
        if(this->padding != 0)
            this->input.pad(this->padding);

        const dim3 kSz = dim3( this->kernel.rows, this->kernel.cols);
        const dim3 inSz = dim3(this->input.rows, this->input.cols);
        const dim3 outSz = dim3(this->output.rows, this->output.cols);

        int e = 0;
        float *d_in, *d_out, *d_kernel;

        e=cudaMalloc(&d_in, inSz.x * inSz.y * sizeof(float));
        e|=cudaMalloc(&d_out, outSz.x * outSz.y * sizeof(float));
        e|=cudaMalloc(&d_kernel, kSz.x * kSz.y * sizeof(float));

        e|=cudaMemcpy(d_in, this->input.data, inSz.x * inSz.y * sizeof(float), cudaMemcpyHostToDevice);
        e|=cudaMemcpy(d_kernel, this->kernel.data, kSz.x * kSz.y * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(
            (output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        gpu::convolution2D<<<numBlocks, threadsPerBlock>>>(d_in, d_kernel, d_out, inSz, outSz, kSz, stride);
        e|=cudaDeviceSynchronize();

        e|=cudaMemcpy(this->output.data, d_out, this->output.cols * this->output.rows * sizeof(float), cudaMemcpyDeviceToHost);

        e|=cudaFree(d_in);
        e|=cudaFree(d_out);
        e|=cudaFree(d_kernel);

        assert(e == 0);
    }





#endif