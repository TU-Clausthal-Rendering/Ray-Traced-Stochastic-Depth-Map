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
#include "ImageEquation.h"

namespace
{
    const std::string kFormula = "formula";
    const std::string kFormat = "format";

    const std::string kShaderFilename = "RenderPasses/ImageEquation/ImageEquation.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ImageEquation>();
}

ImageEquation::ImageEquation(ref<Device> pDevice) : RenderPass(std::move(pDevice)),
mFormula("I0[xy]"), // "I0.Sample(s, uv)"
mFormat(ResourceFormat::RGBA32Float)
{
    mpFbo = Fbo::create(mpDevice);
}

ref<ImageEquation> ImageEquation::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<ImageEquation>(std::move(pDevice));
    for (const auto& [key, value] : dict)
    {
        if (key == kFormula) pPass->mFormula = (const std::string&)value;
        else if (key == kFormat) pPass->mFormat = value;
        else Falcor::logError("unknown field " + key);
    }
    return pPass;
}

Properties ImageEquation::getProperties() const
{
    Properties dict;
    dict[kFormula] = mFormula;
    dict[kFormat] = mFormat;
    return dict;
}

RenderPassReflection ImageEquation::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput("I0", "input image").flags(RenderPassReflection::Field::Flags::Optional).bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput("I1", "input image").flags(RenderPassReflection::Field::Flags::Optional).bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput("I2", "input image").flags(RenderPassReflection::Field::Flags::Optional).bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput("I3", "input image").flags(RenderPassReflection::Field::Flags::Optional).bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput("out", "output image").format(mFormat).bindFlags(ResourceBindFlags::RenderTarget);
    return reflector;
}

void ImageEquation::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPass) // reload shaders if dirty
    {
        mValid = false;

        try
        {
            DefineList defines; // = mpScene->getSceneDefines();
            defines.add("FORMULA", mFormula);
            mpPass = FullScreenPass::create(mpDevice, kShaderFilename, defines);
            Sampler::Desc samplerDesc;
            samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
            mpPass->getRootVar()["s"] = Sampler::create(mpDevice, samplerDesc);
            mValid = true;
        }
        catch (const std::exception& e)
        {
            Falcor::logWarning(e.what());
        }
    }

    if (!mValid) return;

    ref<Texture> I0;
    ref<Texture> I1;
    ref<Texture> I2;
    ref<Texture> I3;

    if (renderData["I0"]) I0 = renderData["I0"]->asTexture();
    if (renderData["I1"]) I1 = renderData["I1"]->asTexture();
    if (renderData["I2"]) I2 = renderData["I2"]->asTexture();
    if (renderData["I3"]) I3 = renderData["I3"]->asTexture();

    auto out = renderData["out"]->asTexture();

    mpFbo->attachColorTarget(out, 0);

    mpPass->getRootVar()["I0"] = I0;
    mpPass->getRootVar()["I1"] = I1;
    mpPass->getRootVar()["I2"] = I2;
    mpPass->getRootVar()["I3"] = I3;

    mpPass->execute(pRenderContext, mpFbo);
}

static const Gui::DropdownList kCommonResourceFormats =
{
    { (uint32_t)ResourceFormat::R8Unorm,         "R8Unorm"},
    { (uint32_t)ResourceFormat::R8Snorm,         "R8Snorm"},
    { (uint32_t)ResourceFormat::R16Unorm,        "R16Unorm"},
    { (uint32_t)ResourceFormat::R16Snorm,        "R16Snorm"},
    { (uint32_t)ResourceFormat::RG8Unorm,        "RG8Unorm"},
    { (uint32_t)ResourceFormat::RG8Snorm,        "RG8Snorm"},
    { (uint32_t)ResourceFormat::RG16Unorm,       "RG16Unorm"},
    { (uint32_t)ResourceFormat::RG16Snorm,       "RG16Snorm"},
    //{ (uint32_t)ResourceFormat::RGB16Unorm,      "RGB16Unorm"}, // not supported as render target
    //{ (uint32_t)ResourceFormat::RGB16Snorm,      "RGB16Snorm"}, // not supported as render target
    { (uint32_t)ResourceFormat::RGBA8Unorm,      "RGBA8Unorm"},
    { (uint32_t)ResourceFormat::RGBA8Snorm,      "RGBA8Snorm"},
    { (uint32_t)ResourceFormat::RGB10A2Unorm,    "RGB10A2Unorm"},
    //{ (uint32_t)ResourceFormat::RGB10A2Uint,     "RGB10A2Uint"},
    { (uint32_t)ResourceFormat::RGBA16Unorm,     "RGBA16Unorm"},
    { (uint32_t)ResourceFormat::RGBA8UnormSrgb,  "RGBA8UnormSrgb"},
    { (uint32_t)ResourceFormat::R16Float,        "R16Float"},
    { (uint32_t)ResourceFormat::RG16Float,       "RG16Float"},
    //{ (uint32_t)ResourceFormat::RGB16Float,      "RGB16Float"}, // not supported as render target
    { (uint32_t)ResourceFormat::RGBA16Float,     "RGBA16Float"},
    { (uint32_t)ResourceFormat::R32Float,        "R32Float"},
    { (uint32_t)ResourceFormat::RG32Float,       "RG32Float"},
    // { (uint32_t)ResourceFormat::RGB32Float,      "RGB32Float"}, // not supported as render target
    { (uint32_t)ResourceFormat::RGBA32Float,     "RGBA32Float"},

};

void ImageEquation::renderUI(Gui::Widgets& widget)
{
    widget.textbox(kFormula, mFormula);
    if (widget.button("recompile")) mpPass.reset();
    if (!mValid) widget.text("invalid!", true);

    if (widget.dropdown(kFormat.c_str(), kCommonResourceFormats, *reinterpret_cast<uint32_t*>(&mFormat))) requestRecompile();
}
