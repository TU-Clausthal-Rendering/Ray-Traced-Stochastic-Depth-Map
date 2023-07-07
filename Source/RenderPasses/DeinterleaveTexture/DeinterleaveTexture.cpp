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
#include "DeinterleaveTexture.h"

namespace
{
    const std::string kTexIn = "texIn";
    const std::string kTexOut = "texOut";

    const std::string kProgram = "RenderPasses/DeinterleaveTexture/Deinterleave.slang";
    const std::string kProgramMS = "RenderPasses/DeinterleaveTexture/DeinterleaveMS.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DeinterleaveTexture>();
}

DeinterleaveTexture::DeinterleaveTexture(ref<Device> pDevice, const Properties& dict)
    : RenderPass(pDevice),
    mMaxRenderTargetCount(Fbo::getMaxColorTargetCount())
{
    if (mMaxRenderTargetCount < 8)
        throw std::runtime_error("DeinterleaveTexture: At least 8 rendertargets need to be supported by the graphics card");

    mpFbo = Fbo::create(mpDevice);
    mpPass = FullScreenPass::create(mpDevice, kProgram);
    mpPassMS = FullScreenPass::create(mpDevice, kProgramMS);
}

Properties DeinterleaveTexture::getProperties() const
{
    return Properties();
}

inline ResourceFormat depthToRendertargetFormat(ResourceFormat format)
{
    switch (format)
    {
    case ResourceFormat::D16Unorm:
        return ResourceFormat::R16Unorm;
    case ResourceFormat::D24UnormS8:
        return ResourceFormat::D16Unorm; // this should be ok 
    case ResourceFormat::D32Float:
        return ResourceFormat::R32Float;
    case ResourceFormat::D32FloatS8X24:
        return ResourceFormat::R32Float;
    default:
        FALCOR_ASSERT(isDepthFormat(format) == false);
        return format;
    }
}

RenderPassReflection DeinterleaveTexture::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kTexIn, "texture 2D").bindFlags(ResourceBindFlags::ShaderResource).texture2D(0, 0, 0, 1, 1);
    auto& outField = reflector.addOutput(kTexOut, "texture 2D array").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource).texture2D(0, 0, 0, 1, mSize).format(mLastFormat);
    mReady = false;

    auto edge = compileData.connectedResources.getField(kTexIn);
    if (edge)
    {
        auto inputFormat = edge->getFormat();
        auto srcWidth = edge->getWidth();
        if (srcWidth == 0) srcWidth = compileData.defaultTexDims.x;
        auto srcHeight = edge->getHeight();
        if (srcHeight == 0) srcHeight = compileData.defaultTexDims.y;
        auto sampleCount = edge->getSampleCount();

        // in case the input format is a depth format => convert to render target format
        if (isDepthFormat(inputFormat))
        {
            inputFormat = depthToRendertargetFormat(inputFormat);
        }

        mLastFormat = inputFormat;

        // reduce the size of the target texture
        auto dstWidth = (srcWidth + mWidth - 1) / mWidth;
        auto dstHeight = (srcHeight + mHeight - 1) / mHeight;

        outField.format(inputFormat).texture2D(dstWidth, dstHeight, sampleCount, 1, mSize);
        mReady = true;
    }

    if ((compileData.defaultTexDims.x % 4 != 0) || (compileData.defaultTexDims.y % 4 != 0))
        logWarning("DeinterleaveTexture textures pixels are not a mutliple of size 4, this might results in artifacts!");

    return reflector;
}

void DeinterleaveTexture::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw std::runtime_error("DeinterleaveTexture::compile - missing incoming reflection information");

    auto edge = compileData.connectedResources.getField(kTexIn);
    if (!edge) throw std::runtime_error("DeinterleaveTexture::compile - missing input information");

    auto inFormat = edge->getFormat();
    setInputFormat(inFormat);
}

void DeinterleaveTexture::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pTexIn = renderData[kTexIn]->asTexture();
    auto pTexOut = renderData[kTexOut]->asTexture();

    execute(pRenderContext, pTexIn, pTexOut);
}

void DeinterleaveTexture::execute(RenderContext* pRenderContext, ref<Texture> pTexIn, ref<Texture> pTexOut)
{
    auto pass = mpPass;
    if (pTexIn->getSampleCount() > 1) pass = mpPassMS;

    auto vars = pass->getRootVar();
    vars["src"] = pTexIn;
    for (uint32_t slice = 0; slice < mSize; slice += 8)
    {
        for (uint slot = 0; slot < 8; ++slot)
        {
            // attach single slice of texture
            mpFbo->attachColorTarget(pTexOut, slot, 0, slice + slot, 1);
        }

        vars["PassData"]["offset"] = slice;
        pass->execute(pRenderContext, mpFbo);
    }
}

void DeinterleaveTexture::renderUI(Gui::Widgets& widget)
{
}

void DeinterleaveTexture::setInputFormat(ResourceFormat format)
{
    auto formatDesc = kFormatDesc[(uint32_t)format];

    std::string type;
    // set correct format type

    switch (formatDesc.Type)
    {
    case FormatType::Float:
    case FormatType::Unorm:
    case FormatType::UnormSrgb:
    case FormatType::Snorm:
        type = "float";
        break;
    case FormatType::Uint:
        type = "uint";
        break;
    case FormatType::Sint:
        type = "int";
        break;
    }

    switch (formatDesc.channelCount)
    {
    case 1: break;
    case 2: type += "2"; break;
    case 3: type += "3"; break;
    case 4: type += "4"; break;
    }

    mpPass->addDefine("type", type);
    mpPassMS->addDefine("type", type);
}
