/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ConvolutionalNet.h"
#include "../Utils/GuardBand/guardband.h"

namespace
{
    const std::string kChannel1 = "bright";
    const std::string kChannel2 = "dark";
    const std::string kChannel3 = "importance";
    const std::string kChannel4 = "depth";

    const std::string kInternal = "i";

    const std::string kOutput = "out";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ConvolutionalNet>();
}

ConvolutionalNet::ConvolutionalNet(ref<Device> pDevice, const Properties& dict)
    : RenderPass(pDevice)
{
    std::filesystem::path resPath;
    auto found = findFileInDataDirectories("NeuralNet", resPath);
    assert(found);
    if(!found)
    {
        logError("could not find neural net data path");
    }
    
    //mNet.load(resPath.string() + "/");
    mNets.resize(mSliceCount);
    for(int slice = 0; slice < mSliceCount; ++slice)
    {
        mNets[slice].load(resPath.string() + "/" + std::to_string(slice) + "_");
        if (slice > 0) if (mNets[slice].getLayerCount() != mNets[0].getLayerCount())
            throw std::runtime_error("mismatching layer count in neural nets");
    }
    mLayerCount = mNets[0].getLayerCount();
}

Properties ConvolutionalNet::getProperties() const
{
    return Properties();
}

RenderPassReflection ConvolutionalNet::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kChannel1, "Channel 1").texture2D(0, 0, 1, 1, mSliceCount);
    reflector.addInput(kChannel2, "Channel 2").texture2D(0, 0, 1, 1, mSliceCount);
    reflector.addInput(kChannel3, "Channel 3").texture2D(0, 0, 1, 1, mSliceCount);
    reflector.addInput(kChannel4, "Channel 4").texture2D(0, 0, 1, 1, mSliceCount);

    auto& outField = reflector.addOutput(kOutput, "Output").bindFlags(ResourceBindFlags::AllColorViews)
        //.format(ResourceFormat::R8Unorm)
        .format(ResourceFormat::R32Float)
        .texture2D(0, 0, 0, 1, mSliceCount);
    mReady = false;

    auto edge = compileData.connectedResources.getField(kChannel1);
    if (edge)
    {
        auto srcWidth = edge->getWidth();
        if (srcWidth == 0) srcWidth = compileData.defaultTexDims.x;
        auto srcHeight = edge->getHeight();
        if (srcHeight == 0) srcHeight = compileData.defaultTexDims.y;
        outField.texture2D(srcWidth, srcHeight, 1, 1, 16);

        // add all internal resources
        for(int layer = 0; layer < mNets[0].getLayerCount() - 1; ++layer)
        {
            auto formatInfo = mNets[0].getMatchingLayerOutputFormat(layer, mPrecision);
            for(int slice = 0; slice < mSliceCount; ++slice)
            {
                reflector.addOutput(getInternalName(layer, slice), "internal")
                    .format(formatInfo.format)
                    .texture2D(srcWidth, srcHeight, 1, 1, formatInfo.layers)
                    .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
            }
            
        }
        
        
        mReady = true;
    }

    return reflector;
}

void ConvolutionalNet::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("DeinterleaveTexture::compile - missing incoming reflection information");

    auto edge = compileData.connectedResources.getField(kChannel1);
    if (!edge) throw std::runtime_error("DeinterleaveTexture::compile - missing input information");

    mFbos.clear();
    mPasses.clear();
    mVars.clear();
}

void ConvolutionalNet::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pChannel1 = renderData[kChannel1]->asTexture();
    auto pChannel2 = renderData[kChannel2]->asTexture();
    auto pChannel3 = renderData[kChannel3]->asTexture();
    auto pChannel4 = renderData[kChannel4]->asTexture();
    auto pOut = renderData[kOutput]->asTexture();

    // create resource if they do not exist
    if(mPasses.empty())
    {
        mPasses.resize(mLayerCount * mSliceCount);
        mVars.resize(mLayerCount * mSliceCount);
        if (mLayerCount == 0) throw std::runtime_error("could not load neural network for data");

        auto& dict = renderData.getDictionary();
        auto guardBand = dict.getValue("guardBand", 0);
        auto quarterGuardBand = guardBand / 4;
        uint2 quarterRes = { pOut->getWidth(0), pOut->getHeight(0) };
        
        for (int layer = 0; layer < mLayerCount; ++layer)
        {
            // create a shader graphics var for each slice in each layer (they will all be unique)
            for(int slice = 0; slice < mSliceCount; ++slice)
            {
                // create pass
                auto& pass = getPass(layer, slice);
                pass = createShader(layer, slice);
                setGuardBandScissors(*pass->getState(), quarterRes, quarterGuardBand);

                auto& vars = getVars(layer, slice);
                vars = GraphicsVars::create(mpDevice, pass->getProgram()->getReflector());
                
                // bind input channel texture
                if (layer > 0)
                {
                    auto tex = renderData[getInternalName(layer - 1, slice)]->asTexture();
                    vars->getRootVar()["channels"] = tex;
                }
            }
        }

        // set correct input for first layer, and activation for last layer
        int lastLayer = mLayerCount - 1;
        for (int slice = 0; slice < mSliceCount; ++slice)
        {
            // set input data
            auto& firstVars = getVars(0, slice);
            firstVars->getRootVar()["channel0"].setSrv(pChannel1->getSRV(0, 1, slice));
            firstVars->getRootVar()["channel1"].setSrv(pChannel2->getSRV(0, 1, slice));
            firstVars->getRootVar()["channel2"].setSrv(pChannel3->getSRV(0, 1, slice));
            firstVars->getRootVar()["channel3"].setSrv(pChannel4->getSRV(0, 1, slice));

            if(mClampOutput)
            {
                // set clamp activation data
                auto& lastVars = getVars(lastLayer, slice);
                lastVars->getRootVar()["clampMax"].setSrv(pChannel1->getSRV(0, 1, slice));
                lastVars->getRootVar()["clampMin"].setSrv(pChannel2->getSRV(0, 1, slice));
            }
        }
            
    }

    if(mFbos.empty())
    {
        mFbos.resize(mLayerCount * mSliceCount);
        for(int layer = 0; layer < mLayerCount - 1; ++layer)
        {
            for(int slice = 0; slice < mSliceCount; ++slice)
            {
                auto& fbo = getFbo(layer, slice);
                fbo = Fbo::create(mpDevice);
                auto tex = renderData[getInternalName(layer, slice)]->asTexture();
                auto nOut = mNets[slice].getOutputChannelCount(layer);
                auto nOutQuarter = (nOut + 3) / 4;
                for (int chOut = 0; chOut < nOutQuarter; ++chOut)
                {
                    fbo->attachColorTarget(tex, chOut, 0, chOut, 1);
                }
            }
            
        }

        // set correct fbo output for last layer
        int lastLayer = mLayerCount - 1;
        for (int slice = 0; slice < mSliceCount; ++slice)
        {
            auto& fbo = getFbo(lastLayer, slice);
            fbo = Fbo::create(mpDevice);
            fbo->attachColorTarget(pOut, 0, 0, slice, 1);
        }
    }

    // run all layers
    for (int layer = 0; layer < mNets[0].getLayerCount(); ++layer)
    {
        for (int slice = 0; slice < mSliceCount; ++slice)
        {
            auto& pass = getPass(layer, slice);
            pass->setVars(getVars(layer, slice));
            auto& fbo = getFbo(layer, slice);
            pass->execute(pRenderContext, fbo, false);
        }
    }

    if(mExportLayers)
    {
        for (int layer = 0; layer < mNets[0].getLayerCount() - 1; ++layer)
        {
            for (int slice = 0; slice < mSliceCount; ++slice)
            {
                auto tex = renderData[getInternalName(layer, slice)]->asTexture();
                tex->captureToFile(0, 0, getInternalName(layer, slice) + ".npy", Bitmap::FileFormat::NumpyFile);
            }
        }
        mExportLayers = false;
    }
}

void ConvolutionalNet::renderUI(Gui::Widgets& widget)
{
    const Gui::DropdownList kPrecisionDropdown =
    {
            { (uint32_t)ConvolutionNet::Precision::Float, "Float" },
            { (uint32_t)ConvolutionNet::Precision::Half, "Half" },
            { (uint32_t)ConvolutionNet::Precision::UNorm, "Unorm" },
    };

    if (widget.dropdown("Precision", kPrecisionDropdown, (uint32_t&)mPrecision))
    {
        requestRecompile();
    }

    if(widget.checkbox("Clamp Output", mClampOutput))
    {
        requestRecompile();
    }

    if(widget.button("Debug Export Layers"))
    {
        mExportLayers = true;
    }
}

ref<FullScreenPass> ConvolutionalNet::createShader(int layer, int slice) const
{
    Program::Desc desc;
    auto activation = ConvolutionNet::Activation::ReLU;
    if (layer == mNets[slice].getLayerCount() - 1)
    {
        if(mClampOutput) activation = ConvolutionNet::Activation::Clamp;
        else activation = ConvolutionNet::Activation::None;
    }
    auto shaderCode = mNets[slice].generateShaderCode(layer, layer != 0, activation);
    
    //std::cout << "Convolutional Shader Code for layer " << layer << std::endl;
    //logInfo(shaderCode);
    //std::cout << "------------------------------------------------\n";
    
    desc.addShaderString(shaderCode, "ConvNet").psEntry("main");
    return FullScreenPass::create(mpDevice, desc);
}

std::string ConvolutionalNet::getInternalName(int layer, int slice)
{
    return kInternal + std::to_string(layer) + "_s" + std::to_string(slice);
}

ref<GraphicsVars>& ConvolutionalNet::getVars(int layer, int slice)
{
    return mVars[mNets[slice].getLayerCount() * slice + layer];
}

ref<FullScreenPass>& ConvolutionalNet::getPass(int layer, int slice)
{
    return mPasses[mNets[slice].getLayerCount() * slice + layer];
}

ref<Fbo>& ConvolutionalNet::getFbo(int layer, int slice)
{
    return mFbos[mNets[slice].getLayerCount() * slice + layer];
}
