#pragma once
#include "Core/Enum.h"

namespace Falcor
{
    enum class StochasticDepthImplementation : unsigned int
    {
        Default = 0,
        CoverageMask,
        ReservoirSampling,
        KBuffer
    };

    FALCOR_ENUM_INFO(
        StochasticDepthImplementation,
        {
            {StochasticDepthImplementation::Default, "Default"},
            {StochasticDepthImplementation::CoverageMask, "CoverageMask"},
            {StochasticDepthImplementation::ReservoirSampling, "ReservoirSampling"},
            {StochasticDepthImplementation::KBuffer, "KBuffer"},
        }
    );

    FALCOR_ENUM_REGISTER(StochasticDepthImplementation);
}
