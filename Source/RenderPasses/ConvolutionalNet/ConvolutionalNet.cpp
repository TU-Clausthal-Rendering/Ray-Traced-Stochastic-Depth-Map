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

    const std::string kInternal = "i";

    const std::string kOutput = "out";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ConvolutionalNet>();
}

ConvolutionalNet::ConvolutionalNet(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    std::filesystem::path resPath;
    auto found = findFileInDataDirectories("NeuralNet/weight_0.npy", resPath);
    assert(found);
    if(!found)
    {
        logError("could not find neural net weights");
    }
    
    mNet.load(resPath.parent_path().string() + "/");
}

Dictionary ConvolutionalNet::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection ConvolutionalNet::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kChannel1, "Channel 1").texture2D(0, 0, 1, 1, 16);
    reflector.addInput(kChannel2, "Channel 2").texture2D(0, 0, 1, 1, 16);

    auto& outField = reflector.addOutput(kOutput, "Output").bindFlags(ResourceBindFlags::AllColorViews).format(ResourceFormat::R8Unorm).texture2D(0, 0, 0, 1, 16);
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
        for(int layer = 0; layer < mNet.getLayerCount() - 1; ++layer)
        {
            auto formatInfo = mNet.getMatchingLayerOutputFormat(layer, mPrecision);
            reflector.addInternal(kInternal + std::to_string(layer), "internal")
                .format(formatInfo.format)
                .texture2D(srcWidth, srcHeight, 1, 1, formatInfo.layers)
                .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
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
}

void ConvolutionalNet::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // create resource if they do not exist
    if(mPasses.empty())
    {
        mPasses.resize(mNet.getLayerCount());
        if (mNet.getLayerCount() == 0) throw std::runtime_error("could not load neural network for data");

        for (int layer = 0; layer < mNet.getLayerCount(); ++layer)
        {
            // create pass
            mPasses[layer] = createShader(layer);
            // bind input channel texture
            if(layer > 0)
            {
                auto tex = renderData[kInternal + std::to_string(layer - 1)]->asTexture();
                mPasses[layer]->getRootVar()["channels"] = tex;
            }
        }
            
    }

    if(mFbos.empty())
    {
        mFbos.resize(mNet.getLayerCount());
        for(int layer = 0; layer < mNet.getLayerCount() - 1; ++layer)
        {
            mFbos[layer] = Fbo::create(mpDevice);
            auto tex = renderData[kInternal + std::to_string(layer)]->asTexture();
            auto nOut = mNet.getOutputChannelCount(layer);
            auto nOutQuarter = (nOut + 3) / 4;
            for(int chOut = 0; chOut < nOutQuarter; ++chOut)
            {
                mFbos[layer]->attachColorTarget(tex, chOut, 0, chOut, 1);
            }
        }

        // last layer
        mFbos.back() = Fbo::create(mpDevice);
        //mFbos.back()->attachColorTarget(renderData[kOutput]->asTexture(), 0);
    }

    auto pChannel1 = renderData[kChannel1]->asTexture();
    auto pChannel2 = renderData[kChannel2]->asTexture();
    auto pOut = renderData[kOutput]->asTexture();

    auto& dict = renderData.getDictionary();
    auto guardBand = dict.getValue("guardBand", 0);
    auto quarterGuardBand = guardBand / 4;
    uint2 quarterRes = { pOut->getWidth(0), pOut->getHeight(0) };
    for(uint slice = 0; slice < 16; ++slice)
    {
        // set inputs for first layer
        auto& pass0 = mPasses.at(0);
        pass0->getRootVar()["channel0"].setSrv(pChannel1->getSRV(0, 1, slice));
        pass0->getRootVar()["channel1"].setSrv(pChannel2->getSRV(0, 1, slice));

        // set clamp values for last pass
        auto& passLast = mPasses.back();
        passLast->getRootVar()["clampMax"].setSrv(pChannel1->getSRV(0, 1, slice));
        passLast->getRootVar()["clampMin"].setSrv(pChannel2->getSRV(0, 1, slice));

        // set output for last layer
        mFbos.back()->attachColorTarget(pOut, 0, 0, slice, 1);

        // run all layers
        for (int layer = 0; layer < mNet.getLayerCount(); ++layer)
        {
            auto& pass = mPasses.at(layer);
            setGuardBandScissors(*pass->getState(), quarterRes, quarterGuardBand);
            auto& fbo = mFbos.at(layer);
            pass->execute(pRenderContext, fbo, false);
        }
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
}

ref<FullScreenPass> ConvolutionalNet::createShader(int layer) const
{
    Program::Desc desc;
    auto activation = ConvolutionNet::Activation::ReLU;
    if (layer == mNet.getLayerCount() - 1) activation = ConvolutionNet::Activation::Clamp;
    auto shaderCode = mNet.generateShaderCode(layer, layer != 0, activation);
    
    std::cout << "Convolutional Shader Code for layer " << layer << std::endl;
    logInfo(shaderCode);
    std::cout << "------------------------------------------------\n";
    
    desc.addShaderString(shaderCode, "ConvNet").psEntry("main");
    return FullScreenPass::create(mpDevice, desc);
}

std::string ConvolutionalNet::getInternalName(int layer, int slize)
{
    return kInternal + std::to_string(layer) + "_s" + std::to_string(slize);
}
