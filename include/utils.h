#ifndef UTILS_H
#define UTILS_H
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct Matrix;
struct Dense;

namespace gpu
{
    __device__ void matrixMultiply(const float* a, const float* b, float *c, const int rows, const int cols);
    __global__ void flatten( float* const* input, float *output, const dim3 inSz);
    __global__ void convolution2D(float *input, const float *kernel, float *output, const dim3 inSz, const dim3 outSz, const dim3 kSz, const int stride);
    __global__ void maxPooling2D(const float *input,  float* output, const dim3 inSz, const dim3 outSz,const int poolingSize, const int stride);
}



struct Dense
{
    int inputSize;
    int outputSize;

    int **weights;
    int *biases;
public:
    Dense(int inSz, int outSz);
    void initParams();
    void forward();
    ~Dense();
};

struct Matrix
{
public:
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

inline void Matrix::normalize(){
    int min_v = INFINITY;
    int max_v = -INFINITY;

    for(int i=0;i<this->cols * this->rows; i++)
    {
        if(min_v>this->data[i])
            min_v = this->data[i];

        if(max_v < this->data[i])
            max_v = this->data[i];
    }

    for(int i=0;i<this->cols * this->rows; i++)
        this->data[i] = (this->data[i] - min_v)/(max_v - this->data[i]);
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
    this->biases = new int[outSz];
    this->weights = new int*[outSz];

    for(int i=0;i<outSz;i++)
        this->weights[i] = new int[inputSize];
};

inline void Dense::initParams(){
    for(int i =0;i< this->outputSize;i++)
        this->biases[i] = i;

    for(int i=0;i<this->outputSize;i++)
        for(int j=0;j<this->inputSize;j++)
            this->weights[i][j] = j;
}

inline void Dense::forward(){

}



inline Dense::~Dense()
{
    delete []this->biases;

    for(int i= 0; i<this->outputSize;++i)
    {
        delete []this->weights[i];
    }
    delete[] this->weights;

}



#endif