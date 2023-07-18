add_plugin(DepthPeelSinglePass)

target_sources(DepthPeelSinglePass PRIVATE
    DepthPeelSinglePass.cpp
    DepthPeelSinglePass.h
    DepthPeelSinglePass.3d.slang
)

target_copy_shaders(DepthPeelSinglePass RenderPasses/DepthPeelSinglePass)

target_source_group(DepthPeelSinglePass "RenderPasses")
