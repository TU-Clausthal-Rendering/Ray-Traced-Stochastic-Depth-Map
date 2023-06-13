#pragma once
#include "Falcor.h"
#include <sstream>
#include "../Utils/npy.h"

struct ConvolutionNet
{
    struct Matrix
    {
        int kernelHeight;
        int kernelWidth;
        int channelsIn;
        int channelsOut;
        std::vector<float> data;

        float get(int kx, int ky, int chIn, int chOut) const
        {
            int index = chOut + chIn * channelsOut + ky * channelsOut * channelsIn + kx * channelsOut * channelsIn * kernelWidth;
            return data.at(index);
        }

        float getBias(int chOut) const
        {
            return data.at(chOut);
        }
    };

    enum class Precision
    {
        Float,
        Half,
        UNorm
    };

    struct LayerFormatInfo
    {
        Falcor::ResourceFormat format;
        unsigned int layers;
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
            assert(shape.size() == 4);
            kernels[l].kernelHeight = shape.at(0);
            kernels[l].kernelWidth = shape.at(1);
            kernels[l].channelsIn = shape.at(2);
            kernels[l].channelsOut = shape.at(3);
            // load biases

            npy::LoadArrayFromNumpy<float>(biasFilename, shape, biases[l].data);
            biases[l].kernelHeight = 1;
            biases[l].kernelWidth = 1;
            biases[l].channelsIn = 1;
            biases[l].channelsOut = shape.at(0);

            ++l;
        }

        if (kernels.size() == 0)
            throw std::exception("Failed to load neural nets");
    }

    /**
     * \brief 
     * \param layer input layer for which to generate the shadercode
     * \param isArrayInput true if the input is an array texture that contains all channels. False if the input are individual single channel textures
     * \return full shader code
     */
    std::string generateShaderCode(size_t layer, bool isArrayInput) const
    {
        std::stringstream ss;

        const auto& k = kernels.at(layer);
        const auto& b = biases.at(layer);

        // input data
        if(isArrayInput)
        {
            ss << "Texture2DArray<float4> channels;\n";
        }
        else
        {
            for(int i = 0; i < k.channelsIn; ++i)
            {
                ss << "Texture2D<float> channel" << i << ";\n";
            }
        }

        // output struct
        ss << "\nstruct ShaderOut {\n";
        auto nChOutQuarter = (k.channelsOut + 3) / 4;
        for (int chOut = 0; chOut < nChOutQuarter; ++chOut)
            ss << "\tfloat4 v" << chOut << " : SV_Target" << chOut << ";\n";
        ss << "};\n";

        // main func
        ss << "\nShaderOut main(float2 uv : TEXCOORD, float4 svPos : SV_POSITION) {\n";
        ss << "\tint2 xy = int2(svPos.xy);\n";
        ss << "\tShaderOut o;\n";
        // initialize ShaderOut with bias
        for(int channel = 0; channel < k.channelsOut; ++channel)
        {
            ss << "\to.v" << (channel / 4) << "[" << (channel % 4) << "] = " << b.getBias(channel) << ";\n";
        }

        // loop over kernel xy
        //ss << "\tfloat chIn; // channel input\n";
        for(int kernely = 0; kernely < k.kernelHeight; ++kernely)
        {
            for(int kernelx = 0; kernelx < k.kernelWidth; ++kernelx)
            {
                int xOff = kernelx - k.kernelWidth / 2;
                int yOff = kernely - k.kernelHeight / 2;

                // obtain texture
                for(int chIn = 0; chIn < k.channelsIn; ++chIn)
                {
                    ss << "\t{\n";
                    // load into chIn variable
                    if(isArrayInput)
                    {
                        ss << "\t\tfloat chIn = channels[int3(xy + int2(" << xOff << "," << yOff << "), " << (chIn / 4) << ")][" << (chIn % 4) << "];\n";
                    }
                    else
                    {
                        ss << "\t\tfloat chIn = channel" << chIn << "[xy + int2(" << xOff << "," << yOff << ")];\n";
                    }

                    // multiply with correct kernels
                    for (int chOut = 0; chOut < k.channelsOut; ++chOut)
                    {
                        ss << "\t\to.v" << (chOut / 4) << "[" << (chOut % 4) << "] += chIn * " << k.get(kernelx, kernely, chIn, chOut) << ";\n";
                    }
                    ss << "\t}\n";
                }
            }
        }

        // apply activation function
        ss << "\t// apply activation function\n";
        for (int chOut = 0; chOut < k.channelsOut; ++chOut)
        {
            ss << "\to.v" << (chOut / 4) << "[" << (chOut % 4) << "] = max(0.0, o.v" << (chOut / 4) << "[" << (chOut % 4) << "]);\n";
        }

        ss << "\treturn o;\n";
        ss << "}\n";

        return ss.str();
    }

    LayerFormatInfo getMatchingLayerOutputFormat(size_t layer, Precision precision) const
    {
        unsigned int count = biases[layer].channelsOut;
        LayerFormatInfo res;
        res.layers = (count + 3u) / 4u;

        if(precision == Precision::Float)
        {
            res.format = Falcor::ResourceFormat::RGBA32Float;
            if (count == 1) res.format = Falcor::ResourceFormat::R32Float;
            if (count == 2) res.format = Falcor::ResourceFormat::RG32Float;
        }
        else if(precision == Precision::Half)
        {
            res.format = Falcor::ResourceFormat::RGBA16Float;
            if (count == 1) res.format = Falcor::ResourceFormat::R16Float;
            if (count == 2) res.format = Falcor::ResourceFormat::RG16Float;
        }
        else if (precision == Precision::UNorm)
        {
            res.format = Falcor::ResourceFormat::RGBA8Unorm;
            if (count == 1) res.format = Falcor::ResourceFormat::R8Unorm;
            if (count == 2) res.format = Falcor::ResourceFormat::RG8Unorm;
        }
        else throw std::runtime_error("invalid precision for getMatchingLayerOutputFormat");

        return res;
    }

    int getLayerCount() const { return int(kernels.size()); }
    int getOutputChannelCount(int layer) const { return kernels.at(layer).channelsOut; }
    int getInputChannelCount(int layer) const { return kernels.at(layer).channelsIn; }
    
    std::vector<Matrix> kernels;
    std::vector<Matrix> biases;
};
