#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct RTAOData
{
    float maxAORayTHit = 0.5f;
    float maxTheoreticalTHit = 0.5f;
    uint spp = 1;
    bool applyExponentialFalloff = true;

    float exponentialFalloffDecayConstant = 2.f; //0-20
    float minimumAmbientIllumination = 0.07f; //0-1
    float normalScale = 0.01f;

};

END_NAMESPACE_FALCOR
