add_plugin(DirectML)

target_sources(DirectML PRIVATE
    DirectML.cpp
    DirectML.h
    DirectMLX.h
)

target_source_group(DirectML "RenderPasses")
