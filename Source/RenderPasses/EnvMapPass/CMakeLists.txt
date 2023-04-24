add_plugin(EnvMapPass)

target_sources(EnvMapPass PRIVATE
    EnvMapPass.cpp
    EnvMapPass.h
    EnvMapPass.ps.slang
)

target_copy_shaders(EnvMapPass RenderPasses/EnvMapPass)

target_source_group(EnvMapPass "RenderPasses")
