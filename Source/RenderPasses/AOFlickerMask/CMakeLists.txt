add_plugin(AOFlickerMask)

target_sources(AOFlickerMask PRIVATE
    AOFlickerMask.cpp
    AOFlickerMask.h
    AOFlickerMask.ps.slang
)

target_copy_shaders(AOFlickerMask RenderPasses/AOFlickerMask)

target_source_group(AOFlickerMask "RenderPasses")
