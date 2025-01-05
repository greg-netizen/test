#include <iostream>
#include <opencv2/opencv.hpp>
#include <utils.h>


void test(cv::Mat img) {
    const int stride = 1;
    const int poolSize = 2;

    Matrix original = Matrix(4, 4);
    original.fromCVMAT(img);


    Matrix filter = Matrix(3, 3);
    filter.data = new float[]{0, -1,0,-1,5,-1,0,-1,0};

    // int output_rows = (original.rows - filter.rows)/stride + 1;
    // int output_cols = (original.cols - filter.cols)/stride + 1;

    int output_rows = (original.rows - poolSize)/stride + 1;
    int output_cols = (original.cols - poolSize)/stride + 1;

    Matrix final = Matrix(output_rows, output_cols);

    float *da, *db, *dc;
    cudaError_t err;

    err = cudaMalloc(&da, original.rows * original.cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&db, filter.rows * filter.cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&dc, final.rows * final.cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(da, original.data, original.rows * original.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, filter.data, filter.rows * filter.cols * sizeof(float), cudaMemcpyHostToDevice);


    const dim3 inSz(original.rows, original.cols);
    const dim3 outSz(output_rows, output_cols);
    const dim3 kSz(filter.rows, filter.cols);

    constexpr dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks(
        (output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    //gpu::convolution2D<<<numBlocks, threadsPerBlock>>>(da, db, dc, inSz, outSz, kSz, stride);
gpu::maxPooling2D<<<numBlocks, threadsPerBlock>>>(da, dc, inSz, outSz,2,1);
    cudaDeviceSynchronize();

    cudaMemcpy(final.data, dc, final.rows * final.cols * sizeof(float), cudaMemcpyDeviceToHost);


    auto newim = final.toCVMAT();
    cv::imshow("image",newim);
    cv::waitKey(0);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}


void test2()
{
    // Allocate and initialize host data
    const int depth = 5, height = 2, width = 2;
    float* h_data = new float[depth * height * width];
    for (int i = 0; i < depth * height * width; i++)
        h_data[i] = i + 1; // Initialize with appropriate values

    // Allocate device memory
    float* d_data;
    float* d_rez;

    cudaMalloc(&d_data, depth * height * width * sizeof(float));
    cudaMalloc(&d_rez, depth * height * width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, depth * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device pointer array
    float** d_inputPtrs;
    cudaMalloc(&d_inputPtrs, depth * sizeof(float*));

    // Assign device pointers for each depth slice
    for (int d = 0; d < depth; d++) {
        cudaMemcpy(&d_inputPtrs[d], &d_data[d * height * width], sizeof(float*), cudaMemcpyHostToDevice);
    }

    // Set kernel launch parameters
    const dim3 blockSize(2, 2, 1);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y,
                        (depth + blockSize.z - 1) / blockSize.z);

    // Launch the kernel
    gpu::flatten<<<gridSize, blockSize>>>(d_inputPtrs, d_rez, dim3(width, height, depth));
    cudaDeviceSynchronize();

    // Allocate host memory for results
    float* rez = new float[depth * height * width];

    // Copy results from device to host
    cudaMemcpy(rez, d_rez, depth * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < depth * height * width; i++)
        std::cout << rez[i] << " ";
    std::cout << std::endl;

    // Free allocated memory
    delete[] h_data;
    delete[] rez;
    cudaFree(d_data);
    cudaFree(d_rez);
    cudaFree(d_inputPtrs);

}

int main() {
    // test(img); // Commented out for clarity
    test2();
    return 0;
}