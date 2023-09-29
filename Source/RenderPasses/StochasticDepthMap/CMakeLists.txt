add_plugin(StochasticDepthMap)

target_sources(StochasticDepthMap PRIVATE
    StochasticDepthMap.cpp
    StochasticDepthMap.h
    StochasticDepth.ps.slang
    Stencil.ps.slang
    StochasticDepthImplementation.h
)

target_copy_shaders(StochasticDepthMap RenderPasses/StochasticDepthMap)

target_source_group(StochasticDepthMap "RenderPasses")
