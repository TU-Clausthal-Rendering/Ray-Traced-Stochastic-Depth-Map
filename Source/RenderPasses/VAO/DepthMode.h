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
        Mipmaps, // depth mipmaps
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
            {DepthMode::Mipmaps, "Mipmaps"},
        }
    );

    FALCOR_ENUM_REGISTER(DepthMode);
}
