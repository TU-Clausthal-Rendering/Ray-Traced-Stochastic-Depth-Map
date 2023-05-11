add_plugin(DepthPeeling)

target_sources(DepthPeeling PRIVATE
    DepthPeeling.cpp
    DepthPeeling.h
    DepthPeeling.3d.slang
)

target_copy_shaders(DepthPeeling RenderPasses/DepthPeeling)

target_source_group(DepthPeeling "RenderPasses")
