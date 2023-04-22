add_plugin(RayShadow)

target_sources(RayShadow PRIVATE
    RayShadow.cpp
    RayShadow.h
    RayShadow.ps.slang
)

target_copy_shaders(RayShadow RenderPasses/RayShadow)

target_source_group(RayShadow "RenderPasses")
