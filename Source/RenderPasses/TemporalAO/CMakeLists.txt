add_plugin(TemporalAO)

target_sources(TemporalAO PRIVATE
    TemporalAO.cpp
    TemporalAO.h
    TemporalAO.ps.slang
)

target_copy_shaders(TemporalAO RenderPasses/TemporalAO)

target_source_group(TemporalAO "RenderPasses")
