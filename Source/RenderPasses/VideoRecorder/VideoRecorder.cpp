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

#include <cstdio>

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
    auto pRenderGraph = (RenderGraph*)renderDict[kRenderGraph];
    if(mpRenderGraph != pRenderGraph)
    {
        // clear old outputs and add the primary output as default target
        mOutputs.clear();
        if(pRenderGraph && pRenderGraph->getOutputCount() > 0) mOutputs.insert(pRenderGraph->getOutputName(0));
    }
    mpRenderGraph = pRenderGraph;
    mClock.tick();
}

void VideoRecorder::renderUI(Gui::Widgets& widget)
{
    saveFrame();

    if (mState == State::Record)
    {
        if (widget.button("Record Stop"))
        {
            stopRecording();
        }
    }
    else // (mState == State::Idle)
    {
        if (widget.button("Record Start") && mState == State::Idle)
        {
            startRecording();
        }
    }
    
    if(mState == State::Preview)
    {
        if(widget.button("Preview Stop"))
        {
            stopPreview();
        }
        //widget.slider("", mRenderIndex, size_t(0), mPathPoints.size() - 1, true);
    }
    else // (mState == State::Idle)
    {
        if(widget.button("Preview Start") && mPathPoints.size() && mState == State::Idle)
        {
            startPreview();
        }
    }

    if(mState == State::Render)
    {
        if (widget.button("Render Stop"))
        {
            stopRender();
        }
    }
    else
    {
        if(widget.button("Render Start") && mPathPoints.size() && mState == State::Idle && mOutputs.size())
        {
            startRender();
        }
        if (mOutputs.empty()) widget.tooltip("No outputs selected. Nothing will be saved to file!");
    }

    widget.var("FPS", mFps, 1, 240);

    if(auto group = widget.group("Smooth Path", true))
    {

        if(group.button("Apply") && mPathPoints.size() > 1)
        {
            smoothPath();
        }
        if(group.button("Clear"))
        {
            mSmoothPoints.clear();
        }
    }

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
    assert(!mClock.isPaused());
    p.time = (float)mClock.getTime();

    return p;
}

namespace fs = std::filesystem;

// file helper functions
bool folderExists(const std::string& folderPath) {
    return fs::is_directory(folderPath);
}

bool createFolder(const std::string& folderPath) {
    try {
        fs::create_directories(folderPath);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

bool deleteFolder(const std::string& folderPath) {
    try {
        fs::remove_all(folderPath);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

void deleteFile(const std::string& filePath) {
    try {
        if (fs::exists(filePath) && fs::is_regular_file(filePath)) {
            fs::remove(filePath);
        }
    }
    catch (const std::exception&) { }
}

void VideoRecorder::saveFrame()
{
    if (mState != State::Render) return;
    assert(mpRenderGraph);
    mRenderIndex++;

    for(const auto& target : mOutputs)
    {
        auto output = mpRenderGraph->getOutput(target);
        if(!output) continue;

        auto tex = output->asTexture();
        assert(tex);
        if(!tex) continue;

        const auto& outputName = output->getName();

        if(mRenderIndex <= 1) // replay index 1 == first frame, because this function is called after the camera update
        {
            // delete old content in the tmp output folder
            deleteFolder(outputName);
            createFolder(outputName);
        }

        auto filenameBase = outputName + "/frame" + outputName;
        std::stringstream filename;
        filename << filenameBase << std::setfill('0') << std::setw(4) << mRenderIndex << ".bmp";
        tex->captureToFile(0, 0, filename.str(), Bitmap::FileFormat::BmpFile);
    }
}

void VideoRecorder::updateCamera()
{
    if (!mpScene) return;

    float time = (float)mClock.getTime();
    auto cam = mpScene->getCamera();

    // helper function to get the interpolated path point based on time
    auto getInterpolatedPathPoint = [&]()
    {
        assert(mPathPoints.size());
        const auto& path = mSmoothPoints.empty() ? mPathPoints : mSmoothPoints;

        auto step2 = std::find_if(path.begin(), path.end(), [time](const PathPoint& p) { return p.time >= time; });
        if (step2 == path.end())
        {
            return path.back();
        }

        auto step1 = step2;
        if (step1 != path.begin()) --step1; // move to previous step

        // interpolate position
        float t = (time - step1->time) / (step2->time - step1->time);
        if (step1->time >= step2->time) t = 1.0; // in case step1 == step2

        PathPoint res;
        res.time = time;
        res.pos = lerp(step1->pos, step2->pos, t);
        res.dir = lerp(step1->dir, step2->dir, t);
        return res;
    };


    switch (mState)
    {
    case State::Record:
        mPathPoints.push_back(createFromCamera());
        break;
    case State::Preview:
    {
        auto p = getInterpolatedPathPoint();
        cam->setPosition(p.pos);
        cam->setTarget(p.pos + p.dir);
        if(p.time >= mPathPoints.back().time)
        {
            // stop animation
            stopPreview();
        }
    }  break;

    case State::Render:
        auto p = getInterpolatedPathPoint();
        cam->setPosition(p.pos);
        cam->setTarget(p.pos + p.dir);
        if (p.time >= mPathPoints.back().time)
        {
            // stop animation
            stopRender();
        }
        break;
    }
}

void VideoRecorder::startRecording()
{
    assert(mState == State::Idle);
    if(mState != State::Idle) return;

    mState = State::Record;
    mPathPoints.clear();
    mSmoothPoints.clear();
    mClock.play();
    mClock.setTime(0);
    mClock.setFramerate(0); // disabled framerate for record
}

void VideoRecorder::startPreview()
{
    assert(mState == State::Idle);
    if(mState != State::Idle) return;

    mState = State::Preview;
    mClock.play();
    mClock.setTime(0);
    mClock.setFramerate(0); // disabled framerate for preview
}

void VideoRecorder::startRender()
{
    assert(mState != State::Record);
    if(mState == State::Record) return;

    mState = State::Render;
    mClock.play();
    mClock.setTime(0);
    mClock.setFramerate(mFps); // use framerate for render
    mRenderIndex = 0;
}

void VideoRecorder::stopRecording()
{
    assert(mState == State::Record);
    if(mState != State::Record) return;

    mState = State::Idle;
    // remove duplicate points from start and end
    auto newStart = std::find_if(mPathPoints.begin(), mPathPoints.end(), [&](const PathPoint& p) { return any(p.pos != mPathPoints.begin()->pos); });
    if (newStart != mPathPoints.end())
    {
        if (newStart != mPathPoints.begin()) --newStart;
        auto newEnd = std::find_if(mPathPoints.rbegin(), mPathPoints.rend(), [&](const PathPoint& p) { return any(p.pos != mPathPoints.rbegin()->pos); }).base();
        if (newEnd != mPathPoints.end()) ++newEnd;
        mPathPoints = std::vector<PathPoint>(newStart, newEnd);

        // fix times
        if(mPathPoints.size())
        {
            const auto startTime = mPathPoints[0].time;
            for(auto& p : mPathPoints)
            {
                p.time -= startTime;
            }
        }
    }
}

void VideoRecorder::stopPreview()
{
    assert(mState == State::Preview);
    if (mState != State::Preview) return;

    mState = State::Idle;
}

void VideoRecorder::stopRender()
{
    assert(mState == State::Render);
    if(mState != State::Render) return;

    mState = State::Idle;

    // create video files for each output
    for (const auto& target : mOutputs)
    {
        auto output = mpRenderGraph->getOutput(target);
        if (!output) continue;

        auto tex = output->asTexture();
        assert(tex);
        if (!tex) continue;

        const auto& outputName = output->getName();

        auto filenameBase = outputName + "/frame" + outputName;
        char buffer[2048];

        std::string outputFilename = outputName + ".mp4";
        deleteFile(outputFilename); // delete old file (otherwise ffmpeg will not write anything)
        sprintf_s(buffer, "ffmpeg -r %d -i %s%%04d.bmp -c:v libx264 -preset medium -crf 12 -vf \"fps=%d,format=yuv420p\" \"%s\" 2>&1", mFps, filenameBase.c_str(), mFps, outputFilename.c_str());

        // last frame, convert to video
        FILE* ffmpeg = _popen(buffer, "w");
        if (!ffmpeg)
        {
            logError("Cannot use popen to execute ffmpeg!");
            continue;
        }

        auto err = _pclose(ffmpeg);
        deleteFolder(outputName); // delete the temporary files
        if (err)
        {
            logError("Error while executing ffmpeg:\n");
        }
    }
}

void VideoRecorder::smoothPath()
{
    if(mPathPoints.size() < 2) return;

    mSmoothPoints.clear();
    // apply gaussian blur to path
    mSmoothPoints.resize(mPathPoints.size());

    float timeRadius = 0.5; // 0.5 seconds

    for(size_t i = 0; i < mPathPoints.size(); ++i)
    {
        const auto& p = mPathPoints[i];
        auto& sp = mSmoothPoints[i];
        sp = p; // initialize with p
        float wsum = 1.0; // weight sum

        auto addPoint = [&](const PathPoint& p)
        {
            float w = expf(-powf(p.time - sp.time, 2) / (2 * powf(timeRadius, 2)));
            sp.pos += w * p.pos;
            sp.dir += w * p.dir;
            wsum += w;
        };

        for(int j = int(i - 1); j >= 0 && mPathPoints[j].time > p.time - timeRadius; --j)
        {
            addPoint(mPathPoints[j]);
        }
        for(int j = int(i + 1); j < int(mPathPoints.size()) && mPathPoints[j].time < p.time + timeRadius; ++j)
        {
            addPoint(mPathPoints[j]);
        }

        // normalize
        sp.pos /= wsum;
        sp.dir /= wsum;
    }
}

void VideoRecorder::forceIdle()
{
    switch(mState)
    {
    case State::Record:
        stopRecording();
        break;
    case State::Preview:
        stopPreview();
        break;
    case State::Render:
        stopRender();
        break;
    }

    assert(mState == State::Idle);
}
