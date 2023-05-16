add_plugin(StochasticDepthMapRT)

target_sources(StochasticDepthMapRT PRIVATE
    StochasticDepthMapRT.cpp
    StochasticDepthMapRT.h
    StochasticDepthMapRT.rt.slang
)

target_copy_shaders(StochasticDepthMapRT RenderPasses/StochasticDepthMapRT)

target_source_group(StochasticDepthMapRT "RenderPasses")
