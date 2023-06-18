add_plugin(PathRecorder)

target_sources(PathRecorder PRIVATE
    PathRecorder.cpp
    PathRecorder.h
)

target_source_group(PathRecorder "RenderPasses")
