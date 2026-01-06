# agent.py
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from config import config as CFG
from agent.memory import memory
from agent.emulator import EmulatorHarness

class AgentPhase(Enum):
    PLANNING = 1
    EXECUTION = 2
    CRITIQUE = 3

class PlanningAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def formulate_goal(self, game_state: Dict, long_term_goal: str, knowledge_context: str = "") -> str:
        prompt = CFG.PLANNER_PROMPT_TEMPLATE.format(
            game_state=game_state,
            long_term_goal=long_term_goal,
            recent_history=memory.get_recent_history(10),
            relevant_facts=knowledge_context + "\n" + "\n".join(memory.query_facts(long_term_goal, n_results=3))
        )
        next_goal = None
        if hasattr(self.llm, "generate_text"):
            next_goal = self.llm.generate_text(prompt)
        if not next_goal:
            next_goal = "Esplora l'area, interagisci con NPC, gestisci menu e battaglie."
        candidate = next_goal.strip().splitlines()[0].strip()
        candidate = candidate.strip("*`_ ").strip()
        candidate = re.sub(r"^[#\-\s]+", "", candidate).strip()
        candidate = re.sub(r"(?i)^prossimo\s+obiettivo\s*:\s*", "", candidate).strip()
        if len(candidate) < 8:
            candidate = "Esplora l'area, interagisci con NPC, gestisci menu e battaglie."
        next_goal = candidate[:240]
        memory.store_fact("planned_goal", next_goal, {"phase": "planning"})
        return next_goal

class ExecutionAgent:
    def __init__(self, emulator: EmulatorHarness, llm_client):
        self.emulator = emulator
        self.llm = llm_client
        self.step_count = 0
        if hasattr(self.llm, 'reset_state'):
            self.llm.reset_state()
            print(f"[Agente] Stato LLM iniziale: {self.llm.get_stats()}")

    def _would_loop(self, candidate_action: str, action_history: List[str]) -> bool:
        if not CFG.ANTI_LOOP_ENABLED:
            return False
        recent = [a for a in action_history[-8:] if isinstance(a, str) and a]
        if len(recent) < 4:
            return False
        if all(a == candidate_action for a in recent[-4:]):
            return True
        if len(recent) >= 5:
            prev = recent[-1]
            if recent[-5] == candidate_action and recent[-4] == prev and recent[-3] == candidate_action and recent[-2] == prev:
                return True
        return False

    def _select_fallback_action(self, game_mode: str, state: Dict, action_history: List[str]) -> str:
        print(f"[Fallback] Modalità: {game_mode}, Stato: {state}")
        
        if game_mode == "menu":
            if len(action_history) >= 2 and action_history[-1] == "a" and action_history[-2] == "a":
                return "b"
            candidates = ["b", "down", "up", "a"]
        elif game_mode == "battle":
            candidates = ["a", "up", "down", "b"]
        else:
            if len(action_history) >= 3:
                recent = action_history[-3:]
                if recent.count("up") >= 2:
                    candidates = ["right", "left", "down", "a"]
                elif recent.count("down") >= 2:
                    candidates = ["right", "left", "up", "a"]
                elif recent.count("right") >= 2:
                    candidates = ["up", "down", "left", "a"]
                elif recent.count("left") >= 2:
                    candidates = ["up", "down", "right", "a"]
                else:
                    candidates = ["up", "right", "down", "left", "a", "start"]
            else:
                candidates = ["up", "right", "a", "start", "down", "left", "b"]

        last = action_history[-1] if action_history else None
        for cand in candidates:
            if cand != last and not self._would_loop(cand, action_history):
                print(f"[Fallback] Scelta intelligente: {cand}")
                return cand
        
        final_choice = "right" if last != "right" else "up"
        print(f"[Fallback] Scelta finale: {final_choice}")
        return final_choice

    def execute_task(self, task_description: str, game_state: Dict, screen_image_np, action_history: List[str]) -> Dict[str, Any]:
        self.step_count = 0
        outcome = {"success": False, "steps": [], "reason": ""}

        if "vai a" in task_description.lower() and CFG.ENABLE_PATHFINDER:
            outcome = self._use_navigation_tool(task_description, game_state)
        elif "puzzle" in task_description.lower():
            outcome = self._use_puzzle_solver(task_description)
        else:
            outcome = self._use_llm_for_basic_actions(task_description, game_state, screen_image_np, action_history)

        memory.add_event({
            "description": f"Eseguito: {task_description}",
            "outcome": outcome
        })
        return outcome

    def _use_navigation_tool(self, task: str, state: Dict) -> Dict:
        print(f"[Execution] Uso il tool di navigazione per: {task}")
        success = self.emulator.navigate_to(5, 10, 8)
        return {"success": success, "steps": ["navigate_tool_called"], "reason": "Tool di navigazione usato."}

    def _use_puzzle_solver(self, task: str) -> Dict:
        success = self.emulator.solve_boulder_puzzle("default")
        return {"success": success, "steps": ["puzzle_solver_called"], "reason": f"Tool puzzle usato per: {task}"}

    def _use_llm_for_basic_actions(self, task: str, state: Dict, screen_image_np, action_history: List[str]) -> Dict:
        game_mode = "exploring"
        if state.get("in_battle"):
            game_mode = "battle"
        elif state.get("menu_open"):
            game_mode = "menu"

        action_index = None
        llm_stats = None
        
        if hasattr(self.llm, "get_action"):
            game_context = {"goal": task, **state}
            action_index = self.llm.get_action(game_mode, game_context, screen_image_np, action_history[-20:])
            if hasattr(self.llm, 'get_stats'):
                llm_stats = self.llm.get_stats()
                print(f"[Agente] LLM stats: {llm_stats}")

        action = None
        if isinstance(action_index, int) and 0 <= action_index < len(CFG.ACTIONS):
            action = CFG.ACTIONS[action_index]
            print(f"[Agente] LLM ha suggerito azione: {action} (index: {action_index})")
        else:
            print(f"[Agente] LLM non ha prodotto azione valida (returned: {action_index})")

        used_fallback = False
        needs_fallback = (not action) or (isinstance(action, str) and self._would_loop(action, action_history))
        if needs_fallback:
            if not CFG.LLM_FALLBACK_ON_ERROR:
                self.step_count += 1
                return {"success": False, "steps": [], "reason": "LLM non ha prodotto un'azione valida.", "action_index": action_index}
            action = self._select_fallback_action(game_mode, state, action_history)
            used_fallback = True
            print(f"[Agente] Usando fallback perché: {'nessuna azione LLM' if not action else 'azione causerebbe loop'}")

        self.emulator.press_button(action.upper())
        self.step_count += 1
        if used_fallback:
            return {"success": True, "steps": [action], "reason": f"Azione fallback: {action}", "action_index": action_index}
        return {"success": True, "steps": [action], "reason": f"Azione LLM: {action}", "action_index": action_index}

class CritiqueAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def review_outcome(self, goal: str, outcome: Dict, new_state: Dict) -> Dict:
        if not CFG.CRITIQUE_ENABLED:
            return {"needs_retry": False, "feedback": "Critica disabilitata."}

        prompt = f"""
        Obiettivo: {goal}
        Risultato segnalato: {outcome}
        Nuovo stato del gioco: {new_state}
        L'obiettivo è stato completamente raggiunto? Rispondi SI o NO.
        Se NO, fornisci una breve ragione.
        """
        critique_response = None
        if hasattr(self.llm, "generate_text"):
            critique_response = self.llm.generate_text(prompt)
        if not critique_response:
            critique_response = "NO"

        critique_upper = critique_response.upper()
        needs_retry = "SI" not in critique_upper and "YES" not in critique_upper
        feedback = critique_response

        memory.add_event({
            "description": f"Critica: {feedback}",
            "type": "critique"
        })
        return {"needs_retry": needs_retry, "feedback": feedback}

class MasterAgent:
    def __init__(self, emulator: EmulatorHarness, llm_client):
        self.phase = AgentPhase.PLANNING
        self.emulator = emulator
        self.llm_client = llm_client
        self.planner = PlanningAgent(llm_client)
        self.executor = ExecutionAgent(emulator, llm_client)
        self.critic = CritiqueAgent(llm_client)
        self.long_term_goal = "Sconfiggi la Lega Pokemon e diventa Campione."
        self.current_goal: Optional[str] = None
        self.action_history: List[str] = []
        self.step_index = 0
        self.logger = logging.getLogger("pokeagent")

    def _infer_game_mode(self, parsed_state: Dict[str, Any]) -> str:
        if parsed_state.get("in_battle"):
            return "battle"
        if parsed_state.get("menu_open"):
            return "menu"
        return "exploring"

    def run_step(self):
        self.step_index += 1
        current = self.emulator.get_current_state()
        parsed = current.get("parsed") or {}
        memory.update_working_memory(screen_state=parsed)

        map_id = current.get("ram", {}).get("current_map", 0)
        kb_context = self.emulator.get_knowledge_context(map_id)
        
        parsed["semantic_context"] = kb_context

        if not self.current_goal or self.step_index % 25 == 1:
            self.current_goal = self.planner.formulate_goal(parsed, self.long_term_goal, knowledge_context=kb_context)
            memory.update_working_memory(goal=self.current_goal)

        game_mode = self._infer_game_mode(parsed)
        self.phase = AgentPhase.EXECUTION
        outcome = self.executor.execute_task(self.current_goal, parsed, current.get("visual_np"), self.action_history)

        chosen_action = None
        if outcome.get("steps"):
            chosen_action = outcome["steps"][-1]
            self.action_history.append(chosen_action)
            if len(self.action_history) > 200:
                self.action_history = self.action_history[-200:]
            memory.update_working_memory(action=chosen_action)

        log_payload = {
            "step": self.step_index,
            "phase": str(self.phase),
            "mode": game_mode,
            "goal": self.current_goal,
            "action": chosen_action,
            "outcome": outcome,
            "state": parsed
        }
        self.logger.info(json.dumps(log_payload, ensure_ascii=False))

        if self.step_index % 50 == 0:
            self.phase = AgentPhase.CRITIQUE
            critique = self.critic.review_outcome(self.current_goal, outcome, parsed)
            self.logger.info(json.dumps({"step": self.step_index, "phase": str(self.phase), "critique": critique}, ensure_ascii=False))
            if not critique.get("needs_retry", False):
                self.long_term_goal = self._update_long_term_goal(parsed)
        self.phase = AgentPhase.PLANNING

    def _update_long_term_goal(self, state: Dict) -> str:
        if "palestra" in state.get('location', '').lower():
            return f"Sconfiggi il Capopalestra in {state['location']}."
        return self.long_term_goal
