add_plugin(AOVarianceFix)

target_sources(AOVarianceFix PRIVATE
    AOVarianceFix.cpp
    AOVarianceFix.h
    AOVarianceFix.ps.slang
)

target_copy_shaders(AOVarianceFix RenderPasses/AOVarianceFix)

target_source_group(AOVarianceFix "RenderPasses")
