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
#include "BinaryDilation.h"

namespace
{
    const std::string kInput = "input";
    const std::string kOutput = "output";

    const std::string kOp = "op";

    const std::string kShaderFile = "RenderPasses/BinaryDilation/BinaryDilation.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, BinaryDilation>();
}

BinaryDilation::BinaryDilation(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kOp) mOp = (const std::string&)value;
        else logWarning("Unknown field `" + key + "` in a BinaryDilation dictionary");
    }

    Program::DefineList defines;
    defines["OP"] = mOp;
    mpPass = FullScreenPass::create(mpDevice, kShaderFile, defines);
    mpFbo = Fbo::create(mpDevice);
}

Dictionary BinaryDilation::getScriptingDictionary()
{
    Dictionary dict;
    dict[kOp] = mOp;
    return dict;
}

RenderPassReflection BinaryDilation::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInput, "binary input").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutput, "dilated binary output").bindFlags(ResourceBindFlags::AllColorViews).format(ResourceFormat::R8Uint);
    return reflector;
}

void BinaryDilation::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pInput = renderData[kInput]->asTexture();
    auto pOutput = renderData[kOutput]->asTexture();

    mpFbo->attachColorTarget(pOutput, 0);
    mpPass->getRootVar()["gInput"] = pInput;
    mpPass->execute(pRenderContext, mpFbo);
}

void BinaryDilation::renderUI(Gui::Widgets& widget)
{
}
