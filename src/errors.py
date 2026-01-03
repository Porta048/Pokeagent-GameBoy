class PokemonAIError(Exception):
    pass
class ROMLoadError(PokemonAIError):
    pass
class MemoryReadError(PokemonAIError):
    pass
class CheckpointLoadError(PokemonAIError):
    pass
class GameEnvironmentError(PokemonAIError):
    pass
class ModelSaveError(PokemonAIError):
    pass