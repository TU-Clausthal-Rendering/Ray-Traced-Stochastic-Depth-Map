add_plugin(ReconstructNormals)

target_sources(ReconstructNormals PRIVATE
    ReconstructNormals.cpp
    ReconstructNormals.h
    ReconstructNormals.ps.slang
)

target_copy_shaders(ReconstructNormals RenderPasses/ReconstructNormals)

target_source_group(ReconstructNormals "RenderPasses")
