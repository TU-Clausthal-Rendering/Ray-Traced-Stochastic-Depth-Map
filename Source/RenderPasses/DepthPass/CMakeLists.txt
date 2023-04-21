add_plugin(DepthPass)

target_sources(DepthPass PRIVATE
    DepthPass.cpp
    DepthPass.h
    DepthPass.ps.slang
)

target_copy_shaders(DepthPass RenderPasses/DepthPass)

target_source_group(DepthPass "RenderPasses")
