add_plugin(VAOExport)

target_sources(VAOExport PRIVATE
    VAOExport.cpp
    VAOExport.h
)

target_source_group(VAOExport "RenderPasses")
