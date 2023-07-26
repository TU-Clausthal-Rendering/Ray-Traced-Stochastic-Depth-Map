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
#include "TemporalDepthPeel.h"

#include "Core/API/RasterizerState.h"
#include "Core/API/VAO.h"
#include "../LinearizeDepth/LinearizeDepth.h"
#include "Core/API/Sampler.h"
//#include "../Utils/GuardBand/guardband.h"

namespace
{
    const std::string kMotionVec = "mvec";
    const std::string kDepth = "linearZ";
    const std::string kDepthOut = "depth2";

    //const std::string kRawDepth = "rawDepth2"; 

    const std::string kIterativeFilename = "RenderPasses/TemporalDepthPeel/TemporalDepthPeel.ps.slang";
    const std::string kRasterFilename = "RenderPasses/TemporalDepthPeel/TemporalDepthPeelRaster.3d.slang";

}
extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, TemporalDepthPeel>();
}

TemporalDepthPeel::TemporalDepthPeel(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpIterPass = FullScreenPass::create(mpDevice, kIterativeFilename);
    mpFbo = Fbo::create(pDevice);
    mpRasterFbo = Fbo::create(pDevice);


    mpRasterState = GraphicsState::create(mpDevice);
    auto rasterProgram = GraphicsProgram::createFromFile(mpDevice, kRasterFilename, "vsMain", "psMain");
    mpRasterState->setProgram(rasterProgram);

    mpRasterVars = GraphicsVars::create(mpDevice, rasterProgram->getReflector());

    RasterizerState::Desc rasterDesc;
    rasterDesc.setCullMode(RasterizerState::CullMode::Front);
    //rasterDesc.setDepthClamp(false);
    //rasterDesc.setFillMode(RasterizerState::FillMode::Wireframe);

    mpRasterState->setRasterizerState(RasterizerState::create(rasterDesc));

    { // depth sampler
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Border, Sampler::AddressMode::Border, Sampler::AddressMode::Border);
        samplerDesc.setBorderColor(float4(0.0f));
        mpIterPass->getRootVar()["gLinearSampler"] = Sampler::create(pDevice, samplerDesc);
        mpRasterVars->getRootVar()["gLinearSampler"] = Sampler::create(pDevice, samplerDesc);
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        mpIterPass->getRootVar()["gPointSampler"] = Sampler::create(pDevice, samplerDesc);
        mpRasterVars->getRootVar()["gPointSampler"] = Sampler::create(pDevice, samplerDesc);
    }
}

Properties TemporalDepthPeel::getProperties() const
{
    return {};
}

RenderPassReflection TemporalDepthPeel::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kDepth, "linear depths").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addInput(kMotionVec, "Motion vectors").bindFlags(Resource::BindFlags::ShaderResource);
    reflector.addOutput(kDepthOut, "depthOut").format(ResourceFormat::R32Float).bindFlags(ResourceBindFlags::AllColorViews);
    //reflector.addInternal(kRawDepth, "non-linear depth 2").format(ResourceFormat::D32Float).bindFlags(ResourceBindFlags::AllDepthViews);
    return reflector;
}

void TemporalDepthPeel::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mpPrevDepth.reset();
    mpPrevDepth2.reset();

    mRasterIndexBuffer = genIndexBuffer(compileData.defaultTexDims);
    auto vao = Vao::create(Vao::Topology::TriangleList, VertexLayout::create(), Vao::BufferVec(), mRasterIndexBuffer, ResourceFormat::R32Uint);
    mpRasterState->setVao(vao);
}

void TemporalDepthPeel::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pDepth = renderData[kDepth]->asTexture();
    auto pMotionVec = renderData[kMotionVec]->asTexture();
    auto pDepthOut = renderData[kDepthOut]->asTexture();
    //auto pRawDepth = renderData[kRawDepth]->asTexture();



    // check if resource dimensions changed and allocate texture accordingly
    mpPrevDepth = allocatePrevFrameTexture(pDepth, std::move(mpPrevDepth));
    mpPrevDepth2 = allocatePrevFrameTexture(pDepth, std::move(mpPrevDepth2));

    //auto& dict = renderData.getDictionary();
    //auto guardBand = dict.getValue("guardBand", 0);
    //setGuardBandScissors(*mpIterPass->getState(), renderData.getDefaultTextureDims(), guardBand);

    ShaderVar vars;
    if (mImplementation == Implementation::Iterative) vars = mpIterPass->getRootVar();
    else if(mImplementation == Implementation::Raster) vars = mpRasterVars->getRootVar();

    // set common vars
    vars["gDepth"] = pDepth;

    mpScene->getCamera()->setShaderData(vars["PerFrameCB"]["gCamera"]);

    auto conversionMat = math::mul(mpScene->getCamera()->getViewMatrix(), math::inverse(mpScene->getCamera()->getPrevViewMatrix()));
    vars["PerFrameCB"]["prevViewToCurView"] = conversionMat;
    conversionMat = math::mul(mpScene->getCamera()->getPrevViewMatrix(), math::inverse(mpScene->getCamera()->getViewMatrix()));
    vars["PerFrameCB"]["curViewToPrevView"] = conversionMat;

    if(mImplementation == Implementation::Iterative)
    {
        vars["gMotionVec"] = pMotionVec;
        vars["gPrevDepth"] = mpPrevDepth;
        vars["gPrevDepth2"] = mpPrevDepth2;

        mpFbo->attachColorTarget(pDepthOut, 0);
        mpIterPass->execute(pRenderContext, mpFbo, true);
    }
    else if(mImplementation == Implementation::Raster)
    {
        // clear pDepthOut before drawing
        //pRenderContext->clearTexture(pDepthOut.get(), float4(0.0f));
        pRenderContext->blit(pDepth->getSRV(), pDepthOut->getRTV());
        //pRenderContext->clearDsv(pRawDepth->getDSV().get(), 1.0f, 0);

        vars["PerFrameCB"]["resolution"] = renderData.getDefaultTextureDims();

        mpRasterFbo->attachColorTarget(pDepthOut, 0);
        mpRasterState->setFbo(mpRasterFbo, true);

        // draw with prev2
        vars["gPrevDepth"] = mpPrevDepth2;
        pRenderContext->drawIndexed(mpRasterState.get(), mpRasterVars.get(), mRasterIndexBuffer->getElementCount(), 0, 0);

        // draw with prev
        vars["gPrevDepth"] = mpPrevDepth;
        pRenderContext->drawIndexed(mpRasterState.get(), mpRasterVars.get(), mRasterIndexBuffer->getElementCount(), 0, 0);
    }

    // save depth and ao from this frame for next frame
    pRenderContext->blit(pDepth->getSRV(), mpPrevDepth->getRTV());
    pRenderContext->blit(pDepthOut->getSRV(), mpPrevDepth2->getRTV());

    if (!mEnabled)
    {
        pRenderContext->blit(pDepth->getSRV(), pDepthOut->getRTV());
    }
}

void TemporalDepthPeel::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enable", mEnabled);

    widget.dropdown("Implementation", mImplementation);
}

void TemporalDepthPeel::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

ref<Texture> TemporalDepthPeel::allocatePrevFrameTexture(const ref<Texture>& original, ref<Texture> prev) const
{
    assert(original);
    bool allocate = prev == nullptr;
    allocate = allocate || (prev->getWidth() != original->getWidth());
    allocate = allocate || (prev->getHeight() != original->getHeight());
    allocate = allocate || (prev->getFormat() != original->getFormat());

    if (!allocate) return prev;

    return Texture::create2D(mpDevice, original->getWidth(), original->getHeight(), original->getFormat(), 1, 1, nullptr, original->getBindFlags());
}

ref<Buffer> TemporalDepthPeel::genIndexBuffer(uint2 res) const
{
    size_t numQuads = (res.x - 1) * (res.y - 1);
    size_t numVertices = numQuads * 6;

    std::vector<uint32_t> indices(numVertices);

    auto calcVertexIndex = [&](size_t x, size_t y)
    {
        return uint32_t(y * res.x + x);
    };

    for(size_t y = 0; y < res.y - 1; ++y)
    {
        for(size_t x = 0; x < res.x - 1; ++x)
        {
            size_t quadIndex = y * (res.x - 1) + x;
            size_t vertexIndex = quadIndex * 6;
            // triangle 1 (cw)
            indices[vertexIndex + 0] = calcVertexIndex(x + 0, y + 0);
            indices[vertexIndex + 2] = calcVertexIndex(x + 0, y + 1);
            indices[vertexIndex + 1] = calcVertexIndex(x + 1, y + 0);

            // triangle 2 (cw)
            indices[vertexIndex + 3] = calcVertexIndex(x + 1, y + 0);
            indices[vertexIndex + 5] = calcVertexIndex(x + 0, y + 1);
            indices[vertexIndex + 4] = calcVertexIndex(x + 1, y + 1);
        }
    }
    
    //return Buffer::create(mpDevice, numVertices * sizeof(indices[0]), Resource::BindFlags::Index, Buffer::CpuAccess::None, indices.data());
    return Buffer::createStructured(mpDevice, sizeof(indices[0]), (uint32_t)numVertices, Resource::BindFlags::Index, Buffer::CpuAccess::None, indices.data());
}
