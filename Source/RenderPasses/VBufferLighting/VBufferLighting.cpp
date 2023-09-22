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
#include "VBufferLighting.h"

namespace
{
    const char kShaderFile[] = "RenderPasses/VBufferLighting/VBufferLighting.ps.slang";

    const std::string kVBuffer = "vbuffer";
    const std::string kColor = "color";
    const std::string kVisBuffer = "visibilityBuffer";

    const std::string kAmbientIntensity = "ambientIntensity";
    const std::string kEnvMapIntensity = "envMapIntensity";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VBufferLighting>();
}

ref<VBufferLighting> VBufferLighting::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<VBufferLighting>(pDevice, props);
}

VBufferLighting::VBufferLighting(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kEnvMapIntensity) mEnvMapIntensity = value;
        else if (key == kAmbientIntensity) mAmbientIntensity = value;
        else logWarning("Unknown field '{}' in a VBufferLighting dictionary.", key);
    }

    mpFbo = Fbo::create(mpDevice);

}

Properties VBufferLighting::getProperties() const
{
    Properties d;
    d[kEnvMapIntensity] = mEnvMapIntensity;
    d[kAmbientIntensity] = mAmbientIntensity;
    return d;
}

RenderPassReflection VBufferLighting::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kVBuffer, "vbuffer");
    reflector.addInput(kVisBuffer, "Visibility buffer used for shadowing. Range is [0,1] where 0 means the pixel is fully-shadowed and 1 means the pixel is not shadowed at all").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(kColor, "Color texture");

    return reflector;
}

void VBufferLighting::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pColor = renderData[kColor]->asTexture();
    auto pVBuffer = renderData[kVBuffer]->asTexture();
    auto pVisBuffer = renderData[kVisBuffer]->asTexture();

    mpFbo->attachColorTarget(pColor, 0);
    auto vars = mpPass->getRootVar();
    vars["vbuffer"] = pVBuffer;
    vars["visibilityBuffer"] = pVisBuffer;

    if (mDirty)
    {
        vars["ConstantCB"]["gAmbientIntensity"] = mAmbientIntensity;
        vars["ConstantCB"]["gEnvMapIntensity"] = mEnvMapIntensity;
        mDirty = false;
    }

    mpScene->setRaytracingShaderData(pRenderContext, vars);

    mpPass->execute(pRenderContext, mpFbo);
}

void VBufferLighting::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Ambient Intensity", mAmbientIntensity, 0.f, 100.f, 0.1f)) mDirty = true;
    if (widget.var("Env Map Intensity", mEnvMapIntensity, 0.f, 100.f, 0.1f)) mDirty = true;
}

void VBufferLighting::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    if (mpScene)
    {
        // create program
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).psEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setShaderModel("6_5");
        mpPass = FullScreenPass::create(mpDevice, desc, mpScene->getSceneDefines());
        mDirty = true;
    }
}
