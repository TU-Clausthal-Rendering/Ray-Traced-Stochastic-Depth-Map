#pragma once
#include "Falcor.h"
#include <sstream>
#include "../Utils/npy.h"

// helper class that loads a neural net from a file
struct NeuralNet
{
    struct Matrix
    {
        int rows;
        int columns;
        std::vector<float> data;
    };

    void load(const std::string& baseFilename, int index)
    {
        kernels.resize(0);
        biases.resize(0);

        std::vector<unsigned long> shape;
        int l = 0; // layer
        while(true)
        {
            std::stringstream ss;
            // load weights
            ss << baseFilename << "_weights" << index << "_kernel" << l << ".npy";
            auto kernelFilename = ss.str();
            ss = std::stringstream(); // clear 
            ss << baseFilename << "_weights" << index << "_bias" << l << ".npy";
            auto biasFilename = ss.str();

            if (!std::filesystem::exists(kernelFilename) || !std::filesystem::exists(biasFilename))
                break;

            kernels.resize(l + 1);
            biases.resize(l + 1);

            npy::LoadArrayFromNumpy<float>(kernelFilename, shape, kernels[l].data);
            kernels[l].rows = shape[0];
            kernels[l].columns = shape[1];
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

class NeuralNetCollection
{
    using Texture = Falcor::Texture;
public:
    enum class Type
    {
        Classifier, // last activation function is sigmoid and function returns a bitmask
        Regressor, // last activation function is linear and inputs will be mutable
    };

    NeuralNetCollection(Type type = Type::Classifier)
        :
    mNumNets(1),
    mLayers(0),
    mType(type)
    {}

    void load(const std::string& baseFilename)
    {
        mFilename = baseFilename;
        mNets.resize(mNumNets);
        for (int i = 0; i < mNumNets; ++i)
        {
            mNets[i].load(baseFilename, i);
        }
        mLayers = (int)mNets[0].kernels.size();
    }

    void writeDefinesToFile(const char* filename) const
    {
        std::ofstream file(filename);
        file << getShaderDefine();
        file.close();
    }

    bool renderUI(Falcor::Gui::Widgets& widget)
    {
        widget.separator();

        std::string prefix;
        if (mType == Type::Classifier)
            prefix = "Classifier";
        else
            prefix = "Regressor";
        widget.text(prefix);

        bool changed = false;

        widget.textbox("File " + prefix, mFilename);
        if(widget.button("Load From File"))
        {
            try
            {
                NeuralNetCollection tmp;
                tmp.load(mFilename);
                *this = tmp;
            }
            catch(const std::exception& e)
            {
                Falcor::msgBox("Neural Net Error", "Error loading neural net: " + std::string(e.what()));
            }
            changed = true;
        }

        widget.separator();

        return changed;
    }

private:
    // shader define for NEURAL_NET_DATA
    std::string getShaderDefine() const
    {
        std::stringstream ss;

        // manually unrolled version
        if (mNumNets == 1)
        {
            const auto& net = mNets[0];
            
            // generate function
            if (mType == Type::Classifier)
            {
                ss << "uint evalClassifier";
                ss << "(inout float inputs[" << net.kernels[0].rows << "], float treshold = 0.0){\n";
            }
            else
            {
                ss << "void evalRegressor";
                ss << "(inout float inputs[" << net.kernels[0].rows << "]){\n";
            }
            

            std::string prevInput = "inputs[";
            std::string prevInputEnd = "]";
            int skippedWeights = 0;
            for (size_t l = 0; l < mLayers; ++l)
            {
                auto kernel = net.kernels[l];
                auto bias = net.biases[l];

                float weightThreshold = 0.0001f;

                // load bias
                //ss << "\tfloat layer" << l << "Output[" << bias.columns << "];\n";
                //ss << "\t[unroll] for(uint outIdx = 0; outIdx < " << bias.columns << "; ++outIdx)\n";
                for(int outIdx = 0; outIdx < bias.columns; ++outIdx)
                    ss << "\tfloat layer" << l << "Output" << outIdx << " = " << bias.data[outIdx] << ";\n";

                // multiply with kernel
                for (int outIdx = 0; outIdx < kernel.columns; ++outIdx)
                    for (int inIdx = 0; inIdx < kernel.rows; ++inIdx)
                    {
                        auto k = kernel.data[inIdx * kernel.columns + outIdx];
                        if(abs(k) > weightThreshold)
                            ss << "\tlayer" << l << "Output" << outIdx << " += " << k << " * " << prevInput << inIdx << prevInputEnd << ";\n";
                        else skippedWeights++;
                    }
                        
                ss << '\n';

                // apply activation function
                if (l == mLayers - 1ull)
                {
                    if(mType == Type::Classifier)
                    {
                        // last layer => sigmoid mask
                        ss << "\n\tuint bitmask = 0;\n";
                        for (int outIdx = 0; outIdx < bias.columns; ++outIdx)
                        {
                            ss << "\tif(layer" << l << "Output" << outIdx << " > treshold)";
                            ss << " bitmask = bitmask | " << (1u << outIdx) << ";\n";
                        }
                    }
                    else if (mType == Type::Regressor)
                    {
                        // last layer => linear
                        for (int outIdx = 0; outIdx < bias.columns; ++outIdx)
                            ss << "\tinputs[" << outIdx << "] = layer" << l << "Output" << outIdx << ";\n";
                    }
                }
                else
                {
                    // relu activation
                    //ss << "\n\t[unroll] for(uint outIdx = 0; outIdx < " << bias.columns << "; ++outIdx)\n";
                    for (int outIdx = 0; outIdx < bias.columns; ++outIdx)
                        ss << "\tlayer" << l << "Output" << outIdx << " = max(layer" << l << "Output" << outIdx << ", 0.0);\n";

                    prevInput = "layer" + std::to_string(l) + "Output";
                    prevInputEnd = "";
                }

                ss << '\n';
            }
            
            if (mType == Type::Classifier)
                ss << "\treturn bitmask;\n}\n";
            else
                ss << "\n}\n";

            std::cout << "Skipped " << skippedWeights << " weights" << std::endl;
        }
        

        return ss.str();
    }

private:
    std::vector<NeuralNet> mNets;
    int mNumNets = 0;
    int mLayers = 0;
    std::string mFilename;
    Type mType;
};
