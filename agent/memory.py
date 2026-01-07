# memory.py
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    logging.getLogger("pokeagent.memory").warning(
        "chromadb non installato: memoria a lungo termine disabilitata"
    )

from config import config as CFG

class HybridMemory:
    """Gestisce memoria ibrida."""

    def __init__(self):
        self.working_memory = {
            "current_goal": None,
            "last_action": None,
            "current_screen_state": None
        }
        self.short_term_memory = []
        self.long_term_memory_client = None
        self.long_term_memory_collection = None
        self._init_long_term_memory()

    def _init_long_term_memory(self):
        """Init DB vettoriale."""
        if chromadb:
            self.long_term_memory_client = chromadb.PersistentClient(
                path=CFG.LONG_TERM_MEMORY_PATH,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            self.long_term_memory_collection = self.long_term_memory_client.get_or_create_collection(
                name="pokemon_game_facts"
            )

    def update_working_memory(self, goal=None, action=None, screen_state=None):
        """Update working memory."""
        if goal:
            self.working_memory["current_goal"] = goal
        if action:
            self.working_memory["last_action"] = action
        if screen_state:
            self.working_memory["current_screen_state"] = screen_state

    def add_event(self, event: Dict[str, Any]):
        """Add event."""
        event["timestamp"] = datetime.now().isoformat()
        self.short_term_memory.append(event)

        if len(self.short_term_memory) % CFG.MEMORY_SUMMARIZE_EVERY_N_STEPS == 0:
            self._summarize_old_events()

    def get_recent_history(self, num_events: int = 10) -> str:
        """Get recent events."""
        recent = self.short_term_memory[-num_events:]
        return "\n".join([f"{e['timestamp']}: {e['description']}" for e in recent])

    def _summarize_old_events(self):
        """Summarize old events."""
        if not chromadb or not self.long_term_memory_collection:
            return

        events_to_summarize = self.short_term_memory[:len(self.short_term_memory)//2]
        if not events_to_summarize:
            return

        summary_text = "Implementare la chiamata a Qwen3-VL per il riassunto."

        self.store_fact("summary", summary_text)

        self.short_term_memory = self.short_term_memory[len(self.short_term_memory)//2:]

    def store_fact(self, fact_type: str, content: str, metadata: Optional[Dict] = None):
        """Store permanent fact."""
        if self.long_term_memory_collection:
            doc_id = f"fact_{datetime.now().timestamp()}"
            self.long_term_memory_collection.add(
                documents=[content],
                metadatas=[{"type": fact_type, **(metadata or {})}],
                ids=[doc_id]
            )

    def query_facts(self, query: str, n_results: int = 3) -> List[str]:
        """Query relevant facts."""
        if not self.long_term_memory_collection:
            return []
        results = self.long_term_memory_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

memory = HybridMemory()
