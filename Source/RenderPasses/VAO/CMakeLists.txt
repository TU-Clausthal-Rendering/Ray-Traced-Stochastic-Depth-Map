add_plugin(VAO)

target_sources(VAO PRIVATE
    VAO.cpp
    VAO.h
    VAO.ps.slang
    VAOData.slang
    DepthMode.h
    vao_to_numpy.h
    npy.h
)

target_copy_shaders(VAO RenderPasses/VAO)

target_source_group(VAO "RenderPasses")

target_link_libraries(VAO PRIVATE gli)
