add_plugin(ForwardLighting)

target_sources(ForwardLighting PRIVATE
    ForwardLighting.cpp
    ForwardLighting.h
    ForwardLighting.3d.slang
)

target_copy_shaders(ForwardLighting RenderPasses/ForwardLighting)

target_source_group(ForwardLighting "RenderPasses")
