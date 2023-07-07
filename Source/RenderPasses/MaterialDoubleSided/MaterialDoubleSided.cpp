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
#include "MaterialDoubleSided.h"

namespace
{
    const std::string kMatIn = "mtlData";
    const std::string kMatOut = "doubleSided";
    const std::string kShaderFilename = "RenderPasses/MaterialDoubleSided/MaterialDoubleSided.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, MaterialDoubleSided>();
}

MaterialDoubleSided::MaterialDoubleSided(ref<Device> pDevice) : RenderPass(std::move(pDevice))
{
    mpPass = FullScreenPass::create(mpDevice, kShaderFilename);
    mpFbo = Fbo::create(mpDevice);
}

ref<MaterialDoubleSided> MaterialDoubleSided::create(ref<Device> pDevice, const Properties& dict)
{
    auto pPass = make_ref<MaterialDoubleSided>(std::move(pDevice));
    return pPass;
}

Properties MaterialDoubleSided::getProperties() const
{
    return Properties();
}

RenderPassReflection MaterialDoubleSided::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kMatIn, "Material data (ID, header)").bindFlags(Resource::BindFlags::ShaderResource).format(ResourceFormat::RGBA32Uint);
    reflector.addOutput(kMatOut, "double sided R8").bindFlags(Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget).format(ResourceFormat::R8Uint);
    return reflector;
}

void MaterialDoubleSided::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pMtlData = renderData[kMatIn]->asTexture();
    auto pDoubleSided = renderData[kMatOut]->asTexture();

    mpFbo->attachColorTarget(pDoubleSided, 0);
    mpPass->getRootVar()["gMtlData"] = pMtlData;

    mpPass->execute(pRenderContext, mpFbo);
}

void MaterialDoubleSided::renderUI(Gui::Widgets& widget)
{
    
}
