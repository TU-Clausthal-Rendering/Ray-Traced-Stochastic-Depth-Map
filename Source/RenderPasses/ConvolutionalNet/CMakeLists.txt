add_plugin(ConvolutionalNet)

target_sources(ConvolutionalNet PRIVATE
    ConvolutionalNet.cpp
    ConvolutionalNet.h
    ConvolutionNet.h
)

target_source_group(ConvolutionalNet "RenderPasses")
