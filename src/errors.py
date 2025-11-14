class PokemonAIError(Exception):
    """Eccezione base personalizzata per errori AI Pokemon."""
    pass


class ROMLoadError(PokemonAIError):
    """Sollevata quando c'è un errore nel caricamento del file ROM."""
    pass


class MemoryReadError(PokemonAIError):
    """Sollevata quando c'è un errore nella lettura della memoria di gioco."""
    pass


class CheckpointLoadError(PokemonAIError):
    """Sollevata quando c'è un errore nel caricamento di un checkpoint."""
    pass


class GameEnvironmentError(PokemonAIError):
    """Sollevata quando c'è un errore con l'ambiente di gioco."""
    pass


class ModelSaveError(PokemonAIError):
    """Sollevata quando c'è un errore nel salvataggio del modello."""
    pass