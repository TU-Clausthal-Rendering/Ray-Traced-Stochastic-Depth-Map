add_plugin(AOGuidedBlur)

target_sources(AOGuidedBlur PRIVATE
    AOGuidedBlur.cpp
    AOGuidedBlur.h
    AOGuidedBlur.ps.slang
)

target_copy_shaders(AOGuidedBlur RenderPasses/AOGuidedBlur)

target_source_group(AOGuidedBlur "RenderPasses")
