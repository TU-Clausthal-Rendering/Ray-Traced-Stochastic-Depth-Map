#pragma once
#include "Core/Enum.h"

namespace Falcor
{
    enum class AOKernel : unsigned int
    {
        VAO,
        HBAO
    };

    FALCOR_ENUM_INFO(
        AOKernel,
        {
            {AOKernel::VAO, "VAO"},
            {AOKernel::HBAO, "HBAO"},
        }
    );

    FALCOR_ENUM_REGISTER(AOKernel);
}
