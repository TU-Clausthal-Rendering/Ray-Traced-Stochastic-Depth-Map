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
#include "DirectML.h"
#include "Core/API/NativeHandleTraits.h"
#include "Core/API/NativeFormats.h"
#include <d3d12.h>
//#define DML_TARGET_VERSION 0x2000
#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h>
#include <wrl/client.h>
//#include "DirectMLX.h"


extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DirectML>();
}

DirectML::DirectML(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined (_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

    auto pRawDevice = mpDevice->getNativeHandle().as<ID3D12Device*>();

    D3D12_FEATURE_DATA_D3D12_OPTIONS d3d12Options = {};
    pRawDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &d3d12Options, sizeof(d3d12Options));

    Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice;
    auto hr = DMLCreateDevice1(
        pRawDevice,
        dmlCreateDeviceFlags,
        DML_FEATURE_LEVEL_2_0,
        IID_PPV_ARGS(dmlDevice.GetAddressOf()));
    assert(SUCCEEDED(hr));
    if(FAILED(hr))
    {
        logError("Could not create the IDMLDevice (DirectML)");
    }
    else
    {
        logInfo("Created the IDMLDevice (DirectML)");
    }
}

Dictionary DirectML::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection DirectML::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}

void DirectML::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");
}

void DirectML::renderUI(Gui::Widgets& widget)
{
}
