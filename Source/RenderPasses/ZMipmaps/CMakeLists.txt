add_plugin(ZMipmaps)

target_sources(ZMipmaps PRIVATE
    ZMipmaps.cpp
    ZMipmaps.h
    Mip.ps.slang
)

target_copy_shaders(ZMipmaps RenderPasses/ZMipmaps)

target_source_group(ZMipmaps "RenderPasses")
