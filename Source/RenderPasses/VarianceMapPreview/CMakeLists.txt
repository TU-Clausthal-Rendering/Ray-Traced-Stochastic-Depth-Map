add_plugin(VarianceMapPreview)

target_sources(VarianceMapPreview PRIVATE
    VarianceMapPreview.cpp
    VarianceMapPreview.h
    VarianceMapPreview.ps.slang
)

target_copy_shaders(VarianceMapPreview RenderPasses/VarianceMapPreview)

target_source_group(VarianceMapPreview "RenderPasses")
