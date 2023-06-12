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

namespace
{
    const std::string kChannel1 = "bright";
    const std::string kChannel2 = "dark";

    const std::string kOutput = "out";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ConvolutionalNet>();
}

ConvolutionalNet::ConvolutionalNet(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
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
        mReady = true;
    }

    return reflector;
}

void ConvolutionalNet::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("DeinterleaveTexture::compile - missing incoming reflection information");

    auto edge = compileData.connectedResources.getField(kChannel1);
    if (!edge) throw std::runtime_error("DeinterleaveTexture::compile - missing input information");
}

void ConvolutionalNet::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pChannel1 = renderData[kChannel1]->asTexture();
    auto pChannel2 = renderData[kChannel2]->asTexture();
    auto pOut = renderData[kOutput]->asTexture();


    for(uint slice = 0; slice < 16; ++slice)
    {
        
    }
}

void ConvolutionalNet::renderUI(Gui::Widgets& widget)
{
}
