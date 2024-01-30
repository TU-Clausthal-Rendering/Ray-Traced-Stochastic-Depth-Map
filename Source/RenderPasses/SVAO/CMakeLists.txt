add_plugin(SVAO)

target_sources(SVAO PRIVATE
    SVAO.cpp
    SVAO.h
    Common.slang
    NeuralNet.h
    AOKernel.h
    CopyStencil.ps.slang
    SVAORaster.ps.slang
    SVAORaster2.ps.slang
    Ray.rt.slang
    VAOData.slang
)

target_copy_shaders(SVAO RenderPasses/SVAO)

target_source_group(SVAO "RenderPasses")
