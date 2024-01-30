#pragma once

namespace Falcor
{
    enum class DepthMode : uint32_t
    {
        SingleDepth,
        DualDepth,
        StochasticDepth,
        Raytraced,
    };

    FALCOR_ENUM_INFO(
        DepthMode,
        {
            {DepthMode::SingleDepth, "SingleDepth"},
            {DepthMode::DualDepth, "DualDepth"},
            {DepthMode::StochasticDepth, "StochasticDepth"},
            {DepthMode::Raytraced, "Raytraced"},
        }
    );

    FALCOR_ENUM_REGISTER(DepthMode);
}
