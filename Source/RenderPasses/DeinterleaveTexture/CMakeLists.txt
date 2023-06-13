add_plugin(DeinterleaveTexture)

target_sources(DeinterleaveTexture PRIVATE
    DeinterleaveTexture.cpp
    DeinterleaveTexture.h
    Deinterleave.slang
    DeinterleaveMS.slang
)

target_copy_shaders(DeinterleaveTexture RenderPasses/DeinterleaveTexture)

target_source_group(DeinterleaveTexture "RenderPasses")
