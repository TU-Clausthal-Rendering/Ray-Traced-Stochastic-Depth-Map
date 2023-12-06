add_plugin(StochasticDepthMapRT)

target_sources(StochasticDepthMapRT PRIVATE
    StochasticDepthMapRT.cpp
    StochasticDepthMapRT.h
    StochasticDepthMapRT.rt.slang
    StochasticDepthMapRT.ps.slang
    Common.slangh
    Stencil.ps.slang
    Jitter.slangh
)

target_copy_shaders(StochasticDepthMapRT RenderPasses/StochasticDepthMapRT)

target_source_group(StochasticDepthMapRT "RenderPasses")
