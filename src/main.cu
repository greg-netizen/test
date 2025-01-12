#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <utils.h>
#include <weights.h>

struct Lenet
{
    Convolution2D* convLayer1;
    MaxPooling* poolLayer1;
    Convolution2D* convLayer2;
    MaxPooling* poolLayer2;
    float* flattenInput;

    Dense denseLayer1;
    Dense denseLayer2;
    Dense output;

    Matrix filters;



    int predict(Matrix image);
};

float* flattenMaxPoollLayer(MaxPooling* poolLayer, int size)
{
    const int imageRows = poolLayer[0].output.rows;
    const int imageCols = poolLayer[0].output.cols;
    float *output = new float[size * imageRows * imageCols];
    for(int k = 0; k < size; k++)
    {
        for(int i = 0; i < imageRows; i++)
            for(int j = 0; j < imageCols; j++)
            {
                output[k * imageRows * imageCols + i * imageCols + j] =
                    poolLayer[k].output.data[i * imageCols + j];
            }
    }
    return output;
}





namespace fs = std::filesystem;

int main() {
    try {
        int correctCount = 0;
        int totalCount = 0;

        // Iterate over all image files in the ../media folder
        for (const auto& entry : fs::directory_iterator("../media")) {
            const auto& path = entry.path();

            // Check if the file is a valid image file
            if (path.extension() == ".png" || path.extension() == ".jpg" || path.extension() == ".jpeg") {
                totalCount++;

                // Load the image in grayscale
                auto img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);

                if (img.empty()) {
                    std::cerr << "Failed to load image: " << path << std::endl;
                    continue;
                }

                // Convert to your Matrix class
                Matrix a;
                a.fromCVMAT(img);

                // Create and use the LeNet model
                Lenet lenet{};
                int output = lenet.predict(a);

                // Compare the output with the expected value (5)
                if (output == 5) {
                    correctCount++;
                }

                std::cout << "Image: " << path.filename() << ", Prediction: " << output << std::endl;
            }
        }

        // Display the results
        std::cout << "Total Images Processed: " << totalCount << std::endl;
        std::cout << "Correct Predictions: " << correctCount << std::endl;
        std::cout << "Accuracy: " << (static_cast<double>(correctCount) / totalCount) * 100 << "%" << std::endl;

    } catch (cv::Exception& e) {
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
        cleanWeights();
        throw;
    } catch (std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    return 0;
}
// Convolution2D* convLayer1;
// MaxPooling* poolLayer1;
// Convolution2D* convLayer2;
// MaxPooling* poolLayer2;
// int* flattenInput;
// Dense layer1;
// Dense layer2;
// Dense output;

int Lenet::predict(Matrix image)
{

    int filterConv1Size = 6;
    int filterConv2Size = 16;

    Matrix* filterConv1 = new Matrix[]{ filters::conv1Filter0,filters::conv1Filter1,filters::conv1Filter2,filters::conv1Filter3,filters::conv1Filter4,filters::conv1Filter5,};
    Matrix* filterConv2 = new Matrix[]{filters::conv2Filter0,filters::conv2Filter1,filters::conv2Filter2,filters::conv2Filter3,filters::conv2Filter4,filters::conv2Filter5,filters::conv2Filter6,filters::conv2Filter7,filters::conv2Filter8,filters::conv2Filter9,filters::conv2Filter10,filters::conv2Filter11,filters::conv2Filter12,filters::conv2Filter13,filters::conv2Filter14,filters::conv2Filter15};

    this->convLayer1 = new Convolution2D[filterConv1Size];
    for (int i = 0; i < filterConv1Size; i++)
    {
        this->convLayer1[i] = Convolution2D(image, filterConv1[i]);
        this->convLayer1[i].conv();
    }

    // First pooling layer
    this->poolLayer1 = new MaxPooling[filterConv1Size];
    for (int i = 0; i < filterConv1Size; i++)
    {
        this->poolLayer1[i] = MaxPooling(this->convLayer1[i].output, 2); // Pooling size 2
        this->poolLayer1[i].pool();
    }

    // Sparse connectivity mapping (connectivity between layer 1 and layer 2)
    const int sparseConnections[16][6] = {
        {1, 1, 0, 0, 1, 1}, // Feature map 0 in conv2 connected to 1, 2, 5, 6 in conv1
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

    this->convLayer2 = new Convolution2D[filterConv2Size];
    for (int j = 0; j < filterConv2Size; j++)
    {

        Matrix combinedInput = Matrix::zeros(this->poolLayer1[0].output.rows, this->poolLayer1[0].output.cols);
        for (int i = 0; i < filterConv1Size; i++) // Iterate over the first layer's feature maps
        {
            if (sparseConnections[j][i] == 1) // Only connect if sparse mapping allows
            {
                combinedInput += this->poolLayer1[i].output;
            }
        }

        // Perform convolution on the combined input
        this->convLayer2[j] = Convolution2D(combinedInput, filterConv2[j]);
        this->convLayer2[j].conv();
    }


    this->poolLayer2 = new MaxPooling[ filterConv2Size];
    for(int i=0;i<filterConv2Size;i++)
    {
        this->poolLayer2[i] = MaxPooling(this->convLayer2[i].output,2);
        this->poolLayer2[i].pool();
    }


    int heighSize = filterConv2Size ;
    this->flattenInput =  flattenMaxPoollLayer(this->poolLayer2,heighSize);

    int flattenRows = poolLayer2[0].output.rows;
    int flattenCols = poolLayer2[0].output.cols;
    int flattenInputSize = heighSize * flattenCols * flattenRows;


    this->denseLayer1 = Dense(flattenInputSize,120, this->flattenInput);
    denseLayer1.initWeights(weights1);
    denseLayer1.forward(aifunc::relu);

    this->denseLayer2 = Dense(120,84, this->denseLayer1.output);
    denseLayer2.initWeights(weights2);
    denseLayer2.forward(aifunc::relu);



    this->output = Dense(84,10,this->denseLayer2.output);
    output.initWeights(weights3);


    output.forward(aifunc::softmax);

    auto maxElemIter = std::max_element(output.output, output.output + output.outputSize);

    // Calculate the index of the maximum element
    int index = std::distance(output.output, maxElemIter);

    delete[] filterConv1;
    delete[] filterConv2;

    return index;



}