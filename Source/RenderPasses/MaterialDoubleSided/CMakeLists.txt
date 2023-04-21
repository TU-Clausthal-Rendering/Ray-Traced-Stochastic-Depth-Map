add_plugin(MaterialDoubleSided)

target_sources(MaterialDoubleSided PRIVATE
    MaterialDoubleSided.cpp
    MaterialDoubleSided.h
    MaterialDoubleSided.ps.slang
)

target_copy_shaders(MaterialDoubleSided RenderPasses/MaterialDoubleSided)

target_source_group(MaterialDoubleSided "RenderPasses")
