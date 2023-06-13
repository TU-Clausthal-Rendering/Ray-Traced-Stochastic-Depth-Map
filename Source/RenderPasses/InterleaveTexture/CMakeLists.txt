add_plugin(InterleaveTexture)

target_sources(InterleaveTexture PRIVATE
    InterleaveTexture.cpp
    InterleaveTexture.h
    Interleave.slang
)

target_copy_shaders(InterleaveTexture RenderPasses/InterleaveTexture)

target_source_group(InterleaveTexture "RenderPasses")
