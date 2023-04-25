add_plugin(VAO)

target_sources(VAO PRIVATE
    VAO.cpp
    VAO.h
    VAO.ps.slang
    VAOData.slang
    DepthMode.h
)

target_copy_shaders(VAO RenderPasses/VAO)

target_source_group(VAO "RenderPasses")
