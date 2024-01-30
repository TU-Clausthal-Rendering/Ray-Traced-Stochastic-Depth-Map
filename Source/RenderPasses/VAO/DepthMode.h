#pragma once

namespace Falcor
{
    enum class DepthMode : uint32_t
    {
        SingleDepth,
        DualDepth,
        StochasticDepth,
        Raytraced,
        MachineClassify,
        MachinePredict,
        PerfectClassify, // for performance analysis (will be expensive)
    };

    FALCOR_ENUM_INFO(
        DepthMode,
        {
            {DepthMode::SingleDepth, "SingleDepth"},
            {DepthMode::DualDepth, "DualDepth"},
            {DepthMode::StochasticDepth, "StochasticDepth"},
            {DepthMode::Raytraced, "Raytraced"},
            {DepthMode::MachineClassify, "MachineClassify"},
            {DepthMode::MachinePredict, "MachinePredict"},
            {DepthMode::PerfectClassify, "PerfectClassify"},
        }
    );

    FALCOR_ENUM_REGISTER(DepthMode);
}
