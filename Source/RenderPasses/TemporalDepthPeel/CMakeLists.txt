add_plugin(TemporalDepthPeel)

target_sources(TemporalDepthPeel PRIVATE
    TemporalDepthPeel.cpp
    TemporalDepthPeel.h
    TemporalDepthPeel.ps.slang
)

target_copy_shaders(TemporalDepthPeel RenderPasses/TemporalDepthPeel)

target_source_group(TemporalDepthPeel "RenderPasses")
