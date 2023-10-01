add_plugin(CompressNormals)

target_sources(CompressNormals PRIVATE
    CompressNormals.cpp
    CompressNormals.h
    CompressNormals.ps.slang
)

target_copy_shaders(CompressNormals RenderPasses/CompressNormals)

target_source_group(CompressNormals "RenderPasses")
