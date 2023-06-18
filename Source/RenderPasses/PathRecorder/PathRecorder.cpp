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
#include "PathRecorder.h"
#include "Utils/Image/npy.h"
#include <random>

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PathRecorder>();
}

PathRecorder::PathRecorder(ref<Device> pDevice, const Dictionary& dict)
    : RenderPass(pDevice)
{
}

Dictionary PathRecorder::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection PathRecorder::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}

void PathRecorder::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;
    if (mAutoRecord)
    {
        auto pos = mpScene->getCamera()->getPosition();
        if (math::length(mLastAutoRecordPoint - pos) > mAutoRecordDistance)
        {
            mLastAutoRecordPoint = pos;
            mSaveNext = true;
        }
    }

    if (mSaveNext)
    {
        auto p = createFromCamera();
        mPoints.push_back(p);
        mSaveNext = false;
    }


}

void PathRecorder::renderUI(Gui::Widgets& widget)
{
    if(widget.button("Add"))
    {
        mSaveNext = true;
    }

    widget.separator();

    if(widget.checkbox("Auto Record", mAutoRecord))
    {
        if (!mpScene)
        {
            mAutoRecord = false;
        }
        else if(mAutoRecord)
        {
            mSaveNext = true;
            mLastAutoRecordPoint = mpScene->getCamera()->getPosition();
        }
    }

    widget.var("Auto Record Distance", mAutoRecordDistance, 1.0f, 100.0f);

    widget.checkbox("Random Direction", mRandomDirection);

    widget.separator();

    widget.text("Length: " + std::to_string(mPoints.size()));

    if(widget.button("Reset"))
    {
        mPoints.clear();
    }

    if(widget.button("Save Numpy") && mPoints.size())
    {
        // save as numpy file
        const float* floatData = reinterpret_cast<const float*>(mPoints.data());
        const unsigned long floatCount = mPoints.size() * (sizeof(PathPoint)/sizeof(float));

        npy::SaveArrayAsNumpy("path.npy", false, 1, &floatCount, floatData);
    }

    if(widget.button("Load Numpy"))
    {
        mPoints = loadPathPoints();
    }
    
    if(widget.button("Preview Points"))
    {
        mPreviewPointIndex = 0;
        mAutoRecord = false;
    }

    if (mPreviewPointIndex >= 0)
    {
        widget.text("Index: " + std::to_string(mPreviewPointIndex));
        if (mPreviewPointIndex >= int(mPoints.size()))
        {
            mPreviewPointIndex = -1;
        }
        else if(mpScene)
        {
            auto p = mPoints[mPreviewPointIndex];
            mpScene->getCamera()->setPosition(p.pos);
            mpScene->getCamera()->setTarget(p.pos + p.dir);
            mPreviewPointIndex += 1;

            // sleep
            //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
}

void PathRecorder::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

std::vector<PathPoint> PathRecorder::loadPathPoints()
{
    std::vector<PathPoint> res;

    std::vector<unsigned long> shape;
    std::vector<float> data;
    npy::LoadArrayFromNumpy("path.npy", shape, data);
    if (shape.empty()) return res; // could not find?

    assert(shape.size() == 1);
    PathPoint* pFirst = reinterpret_cast<PathPoint*>(data.data());
    PathPoint* pLast = reinterpret_cast<PathPoint*>(data.data() + shape[0]);
    res.assign(pFirst, pLast);
    return res;
}

PathPoint PathRecorder::createFromCamera()
{
    assert(mpScene);
    auto cam = mpScene->getCamera();
    PathPoint p;
    p.pos = cam->getPosition();
    p.dir = normalize(cam->getTarget() - p.pos);

    if(mRandomDirection)
    {
        // generate random 3d direction
        // create random generator with seed 89
        static std::mt19937 rng(89);
        // create random distribution in 0 1
        std::uniform_real_distribution<float> dist(0.f, 1.f);

        std::srand(89); // always use the same seed for the noise texture (linear rand uses std rand)
        
        float2 uv;
        uv.x = dist(rng);
        uv.y = dist(rng);
        // Map to radius 1 sphere
        float phi = uv.y * 2.0f * (float)M_PI;
        //float theta = uv.x * (float)M_PI;
        float theta = (uv.x * 0.4f + 0.4f) * (float)M_PI; // force looking down
        p.dir = float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    }

    return p;
}
