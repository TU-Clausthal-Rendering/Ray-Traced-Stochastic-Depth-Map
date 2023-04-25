#pragma once

namespace Falcor
{
    enum class DepthMode
    {
        SingleDepth,
        DualDepth,
        StochasticDepth,
        Raytraced,
        MachineClassify,
        MachinePredict,
        PerfectClassify, // for performance analysis (will be expensive)
    };
}
