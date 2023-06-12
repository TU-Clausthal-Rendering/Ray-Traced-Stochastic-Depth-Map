#pragma once
#include "Falcor.h"
#include <sstream>
#include "../Utils/npy.h"

struct ConvolutionNet
{
    struct Matrix
    {
        int rows;
        int columns;
        int channels;
        std::vector<float> data;
    };

    void load(const std::string& baseFilename)
    {
        kernels.resize(0);
        biases.resize(0);

        std::vector<unsigned long> shape;
        int l = 0; // layer
        while (true)
        {
            std::stringstream ss;
            // load weights
            ss << baseFilename << "weight_" << l << ".npy";
            auto kernelFilename = ss.str();
            ss = std::stringstream(); // clear 
            ss << baseFilename << "bias_" << l << ".npy";
            auto biasFilename = ss.str();

            if (!std::filesystem::exists(kernelFilename) || !std::filesystem::exists(biasFilename))
                break;

            kernels.resize(l + 1);
            biases.resize(l + 1);

            npy::LoadArrayFromNumpy<float>(kernelFilename, shape, kernels[l].data);
            kernels[l].rows = shape[0];
            kernels[l].columns = shape.size() >= 1 ? shape[1] : 1;
            kernels[l].channels = shape.size() >= 3 ? shape[2] : 1;
            // load biases

            npy::LoadArrayFromNumpy<float>(biasFilename, shape, biases[l].data);
            biases[l].rows = 1;
            biases[l].columns = shape[0];

            ++l;
        }

        if (kernels.size() == 0)
            throw std::exception("Failed to load neural nets");
    }

    std::vector<Matrix> kernels;
    std::vector<Matrix> biases;
};
