add_plugin(RTAO)

target_sources(RTAO PRIVATE
    RTAO.cpp
    RTAO.h
    Ray.rt.slang
    RTAOData.slang
)

target_copy_shaders(RTAO RenderPasses/RTAO)

target_source_group(RTAO "RenderPasses")
