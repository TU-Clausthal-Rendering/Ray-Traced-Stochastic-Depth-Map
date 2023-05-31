add_plugin(GuardBand)

target_sources(GuardBand PRIVATE
    GuardBand.cpp
    GuardBand.h
)

target_source_group(GuardBand "RenderPasses")
