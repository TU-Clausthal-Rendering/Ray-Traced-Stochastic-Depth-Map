add_plugin(VarianceMap)

target_sources(VarianceMap PRIVATE
    VarianceMap.cpp
    VarianceMap.h
    DepthInit.ps.slang
    NormalInit.ps.slang
    Mip.ps.slang
)

target_copy_shaders(VarianceMap RenderPasses/VarianceMap)

target_source_group(VarianceMap "RenderPasses")
