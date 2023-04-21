#pragma once
#include "Falcor.h"

inline void setGuardBandScissors(Falcor::GraphicsState& state, Falcor::uint2 resolution, int guardBand)
{
    Falcor::GraphicsState::Scissor sc;
    sc.left = guardBand;
    sc.top = guardBand;
    sc.right = resolution.x - guardBand;
    sc.bottom = resolution.y - guardBand;

    GraphicsState::Viewport vp(0.0f, 0.0f, float(resolution.x), float(resolution.y), 0.0f, 1.0f);
    state.setScissors(0, sc);
    state.setViewport(0, vp, false);
}
