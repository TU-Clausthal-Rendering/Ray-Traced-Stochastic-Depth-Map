add_plugin(RayMinMaxLength)

target_sources(RayMinMaxLength PRIVATE
    RayMinMaxLength.cpp
    RayMinMaxLength.h
    RayMinMaxLength.ps.slang
)

target_copy_shaders(RayMinMaxLength RenderPasses/RayMinMaxLength)

target_source_group(RayMinMaxLength "RenderPasses")
