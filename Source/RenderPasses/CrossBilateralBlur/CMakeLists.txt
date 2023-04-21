add_plugin(CrossBilateralBlur)

target_sources(CrossBilateralBlur PRIVATE
    CrossBilateralBlur.cpp
    CrossBilateralBlur.h
    CrossBilateralBlur.ps.slang
)

target_copy_shaders(CrossBilateralBlur RenderPasses/CrossBilateralBlur)

target_source_group(CrossBilateralBlur "RenderPasses")
