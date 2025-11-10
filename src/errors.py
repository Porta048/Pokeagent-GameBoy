class PokemonAIError(Exception): 
    """Base custom exception for Pokemon AI errors."""
    pass


class ROMLoadError(PokemonAIError):
    """Raised when there's an error loading the ROM file."""
    pass


class MemoryReadError(PokemonAIError):
    """Raised when there's an error reading game memory."""
    pass


class CheckpointLoadError(PokemonAIError):
    """Raised when there's an error loading a checkpoint."""
    pass


class GameEnvironmentError(PokemonAIError):
    """Raised when there's an error with the game environment."""
    pass


class ModelSaveError(PokemonAIError):
    """Raised when there's an error saving the model."""
    pass