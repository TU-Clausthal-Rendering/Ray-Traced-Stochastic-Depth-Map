add_plugin(HBAO)

target_sources(HBAO PRIVATE
    HBAO.cpp
    HBAO.h
    HBAO.ps.slang
    HBAOData.slang
)

target_copy_shaders(HBAO RenderPasses/HBAO)

target_source_group(HBAO "RenderPasses")
