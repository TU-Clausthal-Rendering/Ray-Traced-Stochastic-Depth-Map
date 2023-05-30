add_plugin(BinaryDilation)

target_sources(BinaryDilation PRIVATE
    BinaryDilation.cpp
    BinaryDilation.h
    BinaryDilation.ps.slang
)

target_copy_shaders(BinaryDilation RenderPasses/BinaryDilation)

target_source_group(BinaryDilation "RenderPasses")
