add_plugin(DebugStochasticDepth)

target_sources(DebugStochasticDepth PRIVATE
    DebugStochasticDepth.cpp
    DebugStochasticDepth.h
    DebugStochasticDepth.slang
)

target_copy_shaders(DebugStochasticDepth RenderPasses/DebugStochasticDepth)

target_source_group(DebugStochasticDepth "RenderPasses")
