add_plugin(VBufferLighting)

target_sources(VBufferLighting PRIVATE
    VBufferLighting.cpp
    VBufferLighting.h
    VBufferLighting.ps.slang
)

target_copy_shaders(VBufferLighting RenderPasses/VBufferLighting)

target_source_group(VBufferLighting "RenderPasses")
