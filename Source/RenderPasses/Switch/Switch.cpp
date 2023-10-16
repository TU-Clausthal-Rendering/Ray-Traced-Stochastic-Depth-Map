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
#include "Switch.h"

namespace
{
    const std::string kOutput = "out";
    const std::string kCount = "count";
    const std::string kSelected = "selected";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, Switch>();
}

Switch::Switch(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mNames.resize(2);

    // set count
    for (auto [key, value] : props)
    {
        if (key == kCount) mNames.resize(value);
    }

    // set other properties
    for (auto [key, value] : props)
    {
        if(key == kSelected) mSelectedIndex = value;
        else if (key[0] == 'i')
        {
            auto slot = std::stoi(key.substr(1));
            if(slot < 0 || slot >= mNames.size())
                logWarning("Switch::Switch() - slot index " + std::to_string(slot) + " is out of range");
            else mNames[slot] = static_cast<const std::string&>(value);
        }
        else if (key != kCount) logWarning("Unknown field '" + key + "' in a Switch dictionary");
    }
}

Properties Switch::getProperties() const
{
    Properties props;
    props[kCount] = (uint32_t)mNames.size();
    props[kSelected] = mSelectedIndex;

    for(size_t i = 0; i < mNames.size(); ++i)
    {
        props[getInputName(i)] = mNames[i];
    }
    return props;
}

RenderPassReflection Switch::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    mReady = false;
    for(size_t i = 0; i < mNames.size(); ++i)
    {
        reflector.addInput(getInputName(i), mNames[i]).flags(RenderPassReflection::Field::Flags::Optional);
    }
    reflector.addOutput(kOutput, "selected output");

    auto edge = compileData.connectedResources.getField(getInputName(mSelectedIndex));
    if(edge)
    {
        reflector.addOutput(kOutput, edge->getName())
            .resourceType(edge->getType(), edge->getWidth(), edge->getHeight(), edge->getDepth(), edge->getSampleCount(), edge->getMipCount(), edge->getArraySize())
            .bindFlags(edge->getBindFlags())
            .format(edge->getFormat());
        mReady = true;
    }

    return reflector;
}

void Switch::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    //if (!mReady) throw std::runtime_error("Switch::compile - missing incoming reflection information");
}

void Switch::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto inputName = getInputName(mSelectedIndex);
    if (!renderData[inputName]) return;

    auto pInput = renderData[inputName]->asTexture();
    auto pOutput = renderData[kOutput]->asTexture();

    for(uint32_t mip = 0; mip < pInput->getMipCount(); ++mip)
    {
        for(uint32_t layer = 0; layer < pInput->getArraySize(); ++layer)
        {
            pRenderContext->blit(pInput->getSRV(mip, 1, layer, 1), pOutput->getRTV(mip, layer, 1));
        }
    }
}

void Switch::renderUI(Gui::Widgets& widget)
{
    std::vector<Gui::DropdownValue> dropdowns;
    dropdowns.resize(mNames.size());
    for(size_t i = 0; i < mNames.size(); ++i)
    {
        dropdowns[i].label = mNames[i];
        if(dropdowns[i].label.empty()) dropdowns[i].label = getInputName(i);
        dropdowns[i].value = (uint32_t)i;
    }


    if (widget.dropdown("Switch", dropdowns, mSelectedIndex))
    {
        requestRecompile();
    }

    if (auto group = widget.group("Configure", false))
    {
        int count = (int)mNames.size();
        if (group.var("Count", count, 1, 16))
        {
            mNames.resize(count);
            requestRecompile();
        }

        for (size_t i = 0; i < mNames.size(); ++i)
        {
            auto inputName = getInputName(i);
            group.textbox(inputName, mNames[i]);
        }
    }
}

std::string Switch::getInputName(size_t index) const
{
    return "i" + std::to_string(index);
}
