#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <utils.h>
#include <weights.h>

struct Lenet
{
    std::vector<Convolution2D> convLayer1;
    std::vector<MaxPooling> poolLayer1;
     std::vector<Convolution2D> convLayer2;
    std::vector<MaxPooling>  poolLayer2;
    std::vector<float> flattenInput;

    Dense denseLayer1;
    Dense denseLayer2;
    Dense output;

    Matrix filters;

    int predict(Matrix &image);
};

std::vector<float> flattenPoolLayerOutputs(const std::vector<MaxPooling>& poolLayer2)
{
    std::vector<float> flattened;
    size_t totalSize = 0;
    for (const auto& layer : poolLayer2) {
        totalSize += layer.output.data.size();
    }
    flattened.reserve(totalSize);
    for (const auto& layer : poolLayer2) {
        flattened.insert(flattened.end(),
                         layer.output.data.begin(),
                         layer.output.data.end());
    }

    return flattened;
}

namespace fs = std::filesystem;

int main()
{
    int correctCount = 0;
    int totalCount = 0;


    for (const auto& entry : fs::directory_iterator("../media"))
    {
        const auto& path = entry.path();


        if (path.extension() == ".png" || path.extension() == ".jpg" || path.extension() == ".jpeg")
        {
            totalCount++;


            auto img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);

            if (img.empty())
            {
                std::cerr << "Failed to load image: " << path << std::endl;
                continue;
            }
            Matrix a;
            a.fromCVMAT(img);

            Lenet lenet{};
            int output = lenet.predict(a);

            if (output == 4)
            {
                correctCount++;
            }

            std::cout << "Image: " << path.filename() << ", Prediction: " << output << std::endl;
        }
    }

    std::cout << "Total Images Processed: " << totalCount << std::endl;
    std::cout << "Correct Predictions: " << correctCount << std::endl;
    std::cout << "Accuracy: " << (static_cast<double>(correctCount) / totalCount) * 100 << "%" << std::endl;


    return 0;
}


int Lenet::predict(Matrix& image)
{

    const int conv1FilterSize = 6;
    const int conv2FilterSize = 16;


    std::vector<std::vector<float>> conv1Filters={
        filters::conv1Filter0,
        filters::conv1Filter1,
        filters::conv1Filter2,
        filters::conv1Filter3,
        filters::conv1Filter4,
        filters::conv1Filter5
    };
    std::vector<std::vector<float>> conv2Filters={
        filters::conv2Filter0,
        filters::conv2Filter0,
        filters::conv2Filter1,
        filters::conv2Filter2,
        filters::conv2Filter3,
        filters::conv2Filter4,
        filters::conv2Filter5,
        filters::conv2Filter6,
        filters::conv2Filter7,
        filters::conv2Filter8,
        filters::conv2Filter9,
        filters::conv2Filter10,
        filters::conv2Filter11,
        filters::conv2Filter12,
        filters::conv2Filter13,
        filters::conv2Filter14,
        filters::conv2Filter15,
    };

    this->convLayer1.clear();
    for (int i = 0; i < conv1FilterSize; i++)
    {
        this->convLayer1.emplace_back(image, Matrix(5, 5, conv1Filters[i]));
        this->convLayer1.back().conv();
    }
    this->poolLayer1.clear();
    for (int i = 0; i < conv1FilterSize; i++) {

        this->poolLayer1.emplace_back(this->convLayer1[i].output, /*stride=*/2);

        this->poolLayer1.back().pool();
    }
    const int sparseConnections[16][6] = {
        {1, 1, 0, 0, 1, 1},
        {1, 0, 1, 0, 1, 0},
        {0, 1, 1, 0, 0, 1},
        {1, 0, 0, 1, 1, 0},
        {0, 1, 0, 1, 0, 1},
        {1, 1, 0, 0, 1, 0},
        {0, 1, 1, 0, 0, 0},
        {1, 0, 0, 1, 0, 1},
        {0, 0, 1, 1, 1, 0},
        {1, 1, 1, 0, 0, 0},
        {0, 1, 0, 1, 1, 1},
        {1, 0, 1, 1, 0, 0},
        {0, 1, 1, 0, 1, 1},
        {1, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 1, 1},
        {1, 0, 1, 0, 0, 1}
    };

    this->convLayer2.clear();
    for (int j = 0; j < conv2FilterSize; j++) {
        Matrix combinedInput = Matrix::zeros(
            this->poolLayer1[0].output.rows,
            this->poolLayer1[0].output.cols
        );

        for (int i = 0; i < conv1FilterSize; i++) {
            if (sparseConnections[j][i] == 1) {
                combinedInput = combinedInput + this->poolLayer1[i].output;
            }
        }
        this->convLayer2.emplace_back(combinedInput, Matrix(5, 5, conv2Filters[j]));
        this->convLayer2.back().conv();
    }
    this->poolLayer2.clear();
    for (int i = 0; i < conv2FilterSize; i++)
    {
        this->poolLayer2.emplace_back(this->convLayer2[i].output,2);
        this->poolLayer2.back().pool();
    }


    this->flattenInput = flattenPoolLayerOutputs(this->poolLayer2);

    int heightSize = conv2FilterSize;
    int flattenRows = poolLayer2[0].output.rows;
    int flattenCols = poolLayer2[0].output.cols;
    int flattenInputSize = heightSize * flattenCols * flattenRows;

    this->denseLayer1 = Dense(flattenInputSize, 120, this->flattenInput);
    denseLayer1.weights = weights1;
    denseLayer1.forward(aifunc::relu);

    this->denseLayer2 = Dense(120, 84, this->denseLayer1.output);
    denseLayer2.weights = weights2;
    denseLayer2.forward(aifunc::relu);

    this->output = Dense(84, 10, this->denseLayer2.output);
    output.weights = weights3;
    output.forward(aifunc::softmax);

    float maxVal = -1e9f;
    int index = -1;
    for(int i = 0; i < output.outputSize; i++)
    {
        float val = output.output[i];
        if(val > maxVal)
        {
            maxVal = val;
            index = i;
        }
        std::cout << val << "\n";
    }

    return index;
}
