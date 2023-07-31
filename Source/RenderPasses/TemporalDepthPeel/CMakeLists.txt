add_plugin(TemporalDepthPeel)

target_sources(TemporalDepthPeel PRIVATE
    TemporalDepthPeel.cpp
    TemporalDepthPeel.h
    TemporalDepthPeel.ps.slang
    TemporalDepthPeelRaster.3d.slang
    TemporalDepthPeelPoints.3d.slang
    TemporalDepthPeelPointsFix.ps.slang
)

target_copy_shaders(TemporalDepthPeel RenderPasses/TemporalDepthPeel)

target_source_group(TemporalDepthPeel "RenderPasses")
