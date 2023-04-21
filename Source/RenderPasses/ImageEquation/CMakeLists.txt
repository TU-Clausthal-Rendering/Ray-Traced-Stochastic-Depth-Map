add_plugin(ImageEquation)

target_sources(ImageEquation PRIVATE
    ImageEquation.cpp
    ImageEquation.h
    ImageEquation.ps.slang
)

target_copy_shaders(ImageEquation RenderPasses/ImageEquation)

target_source_group(ImageEquation "RenderPasses")
