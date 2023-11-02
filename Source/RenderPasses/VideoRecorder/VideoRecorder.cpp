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
#include "VideoRecorder.h"

#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VideoRecorder>();
}

VideoRecorder::VideoRecorder(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
}

Properties VideoRecorder::getProperties() const
{
    return {};
}

RenderPassReflection VideoRecorder::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}

void VideoRecorder::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto renderDict = renderData.getDictionary();
    mpRenderGraph = (RenderGraph*)renderDict[kRenderGraph];

}

void VideoRecorder::renderUI(Gui::Widgets& widget)
{
    saveFrame();

    if (mState == State::Recording)
    {
        if (widget.button("Record Stop"))
        {
            mState = State::Idle;
            // remove duplicate points from start and end
            auto newStart = std::find_if(mPathPoints.begin(), mPathPoints.end(), [&](const PathPoint& p) { return any(p.pos != mPathPoints.begin()->pos); });
            if (newStart != mPathPoints.end())
            {
                if (newStart != mPathPoints.begin()) --newStart;
                auto newEnd = std::find_if(mPathPoints.rbegin(), mPathPoints.rend(), [&](const PathPoint& p) { return any(p.pos != mPathPoints.rbegin()->pos); }).base();
                if (newEnd != mPathPoints.end()) ++newEnd;
                mPathPoints = std::vector<PathPoint>(newStart, newEnd);
            }
        }
    }
    else // (mState == State::Idle)
    {
        if (widget.button("Record Start"))
        {
            mState = State::Recording;
            mPathPoints.clear();
        }
    }

    
    if(mState == State::Replaying)
    {
        if(widget.button("Replay Stop"))
        {
            mState = State::Idle;
        }
        widget.slider("", mReplayIndex, size_t(0), mPathPoints.size() - 1, true);
    }
    else // (mState == State::Idle)
    {
        if(widget.button("Replay Start") && mPathPoints.size())
        {
            mState = State::Replaying;
            mReplayIndex = 0;
        }
    }

    if(widget.checkbox("Replay Save To File", mSaveToFile))
    {
        // restart replay
        if(mState == State::Replaying) mReplayIndex = 0;
    }
    if(mSaveToFile && mOutputs.empty()) widget.tooltip("No outputs selected. Nothing will be saved to file!");

    widget.text("Path Points: " + std::to_string(mPathPoints.size()));

    // list all outputs
    if(auto g = widget.group("Outputs"))
    {
        bool selectAll = false;

        if (g.button("All")) selectAll = true;
        if (g.button("None", true))
        {
            mOutputs.clear();
        }
        for(uint32_t i = 0; i < mpRenderGraph->getOutputCount(); i++)
        {
            auto name = mpRenderGraph->getOutputName(i);
            bool selected = mOutputs.count(name) != 0;
            if(g.checkbox(name.c_str(), selected))
            {
                if(selected) mOutputs.insert(name);
                else mOutputs.erase(name);
            }
            if(selectAll) mOutputs.insert(name);
        }
    }

    // logic
    updateCamera();
}

PathPoint VideoRecorder::createFromCamera()
{
    assert(mpScene);
    auto cam = mpScene->getCamera();
    PathPoint p;
    p.pos = cam->getPosition();
    p.dir = normalize(cam->getTarget() - p.pos);

    return p;
}

void VideoRecorder::saveFrame()
{
    if (!mSaveToFile) return;
    if (mState != State::Replaying) return;
    assert(mpRenderGraph);

    for(const auto& target : mOutputs)
    {
        auto output = mpRenderGraph->getOutput(target);
        if(!output) continue;

        auto tex = output->asTexture();
        assert(tex);
        if(!tex) continue;

        auto filename = "frame" + output->getName() + std::to_string(mReplayIndex) + ".bmp";
        tex->captureToFile(0, 0, filename, Bitmap::FileFormat::BmpFile);
    }
}

void VideoRecorder::updateCamera()
{
    if (!mpScene) return;

    switch (mState)
    {
    case State::Recording:
        mPathPoints.push_back(createFromCamera());
        break;
    case State::Replaying:
        if (mReplayIndex >= mPathPoints.size())
        {
            // transfer to idle
            mState = State::Idle;
            // go back to initial position
            if (mPathPoints.size())
            {
                auto cam = mpScene->getCamera();
                cam->setPosition(mPathPoints[0].pos);
                cam->setTarget(mPathPoints[0].pos + mPathPoints[0].dir);
            }
        }
        else
        {
            auto p = mPathPoints[mReplayIndex++];
            auto cam = mpScene->getCamera();
            cam->setPosition(p.pos);
            cam->setTarget(p.pos + p.dir);

        }
        break;
    }
}
