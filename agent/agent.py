# agent.py
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from config import config as CFG
from agent.memory import memory
from agent.emulator import EmulatorHarness

logger = logging.getLogger("pokeagent.agent")

class AgentPhase(Enum):
    PLANNING = 1
    EXECUTION = 2
    CRITIQUE = 3

class PlanningAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def _planning_budget(self, game_state: Dict, long_term_goal: str, knowledge_context: str) -> int:
        if not bool(CFG.ADAPTIVE_COMPUTE_ENABLED):
            return int(CFG.LLM_PLANNING_BUDGET_DEFAULT)

        combined = f"{long_term_goal}\n{knowledge_context}\n{game_state}".lower()
        strategic_markers = (
            "elite", "lega", "indigo", "campione", "champion", "e4", "pre-elite", "pre elite", "plateau"
        )
        if any(m in combined for m in strategic_markers):
            return int(CFG.LLM_PLANNING_BUDGET_STRATEGIC)
        return int(CFG.LLM_PLANNING_BUDGET_DEFAULT)

    def formulate_goal(self, game_state: Dict, long_term_goal: str, knowledge_context: str = "") -> str:
        prompt = CFG.PLANNER_PROMPT_TEMPLATE.format(
            game_state=game_state,
            long_term_goal=long_term_goal,
            recent_history=memory.get_recent_history(10),
            relevant_facts=knowledge_context + "\n" + "\n".join(memory.query_facts(long_term_goal, n_results=3))
        )
        next_goal = None
        if hasattr(self.llm, "generate_text"):
            budget = self._planning_budget(game_state, long_term_goal, knowledge_context)
            next_goal = self.llm.generate_text(prompt, num_predict=budget)
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
            logger.debug("Stato LLM iniziale: %s", self.llm.get_stats())

    def _await_llm_with_ui_ticks(self, fn, max_wait_s: float) -> Any:
        if (not bool(CFG.RENDER_ENABLED)) or bool(CFG.HEADLESS) or (not hasattr(self.emulator, "tick_idle")):
            return fn()
        max_wait_s_f = float(max_wait_s)
        if max_wait_s_f <= 0:
            return None

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            deadline = time.monotonic() + max_wait_s_f
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    return future.result(timeout=min(0.05, remaining))
                except FutureTimeoutError:
                    try:
                        self.emulator.tick_idle(1)
                    except Exception:
                        pass

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
        logger.debug("Fallback: mode=%s state=%s", game_mode, state)
        
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
                logger.debug("Fallback scelta: %s", cand)
                return cand
        
        final_choice = "right" if last != "right" else "up"
        logger.debug("Fallback scelta finale: %s", final_choice)
        return final_choice

    def _difficulty_score(self, game_mode: str, state: Dict, task: str) -> float:
        base = 0.0
        if game_mode == "battle":
            base += 2.0
        elif game_mode == "menu":
            base += 1.0
        else:
            base += 0.0

        semantic = str(state.get("semantic_context", "") or "").lower()
        location = str(state.get("location", "") or "").lower()
        task_l = str(task or "").lower()
        text_box = str(state.get("text_box", "") or "").lower()

        markers = ("capopalestra", "palestra", "gym", "giovanni", "elite", "lega", "rival", "champion", "campione")
        if any(m in semantic for m in markers) or any(m in location for m in markers) or any(m in task_l for m in markers):
            base += 2.0
        if any(m in text_box for m in ("leader", "capopalestra", "elite", "lega", "campione")):
            base += 1.0

        try:
            opponent_level = int(state.get("opponent_level", 0) or 0)
        except Exception:
            opponent_level = 0
        if opponent_level >= 40:
            base += 1.5
        elif opponent_level >= 20:
            base += 0.5

        try:
            badges = int(state.get("badges", 0) or 0)
        except Exception:
            badges = 0
        if badges >= 6:
            base += 1.0

        return base

    def _compute_profile(self, game_mode: str, state: Dict, task: str) -> Dict[str, Any]:
        if not bool(CFG.ADAPTIVE_COMPUTE_ENABLED):
            return {"num_predict": None, "temperature": None, "n_candidates": 1}

        score = self._difficulty_score(game_mode, state, task)

        num_predict = int(CFG.LLM_ACTION_BUDGET_EXPLORING)
        temperature = float(CFG.LLM_TEMPERATURE)
        n_candidates = 1
        min_candidates = 1
        max_candidates = int(CFG.PARALLEL_SAMPLING_MAX_CANDIDATES)

        if game_mode == "menu":
            num_predict = int(CFG.LLM_ACTION_BUDGET_MENU)
            n_candidates = 2
            min_candidates = 2
        elif game_mode == "battle":
            num_predict = int(CFG.LLM_ACTION_BUDGET_BATTLE)
            temperature = max(0.4, float(CFG.LLM_TEMPERATURE))
            n_candidates = int(CFG.PARALLEL_SAMPLING_MIN_CANDIDATES)
            min_candidates = int(CFG.PARALLEL_SAMPLING_MIN_CANDIDATES)

        if score >= 4.0:
            num_predict = int(CFG.LLM_ACTION_BUDGET_BOSS)
            temperature = max(0.5, float(CFG.LLM_TEMPERATURE))
            n_candidates = int(CFG.PARALLEL_SAMPLING_MAX_CANDIDATES)

        target = 0.75 + min(0.20, 0.05 * float(score))
        base_p = {"exploring": 0.55, "menu": 0.65, "battle": 0.70}.get(game_mode, 0.55)
        p = max(0.12, min(0.90, float(base_p - 0.06 * min(float(score), 6.0))))
        n_scaled = 1
        try:
            if 0.0 < target < 1.0 and 0.0 < p < 1.0:
                denom = math.log(1.0 - p)
                if denom < 0:
                    n_scaled = int(math.ceil(math.log(1.0 - target) / denom))
        except Exception:
            n_scaled = 1

        n_candidates = max(int(n_candidates), int(min_candidates), int(n_scaled))
        n_candidates = min(int(max_candidates), int(n_candidates))

        if hasattr(self.llm, "get_stats"):
            stats = self.llm.get_stats() or {}
            latency = float(stats.get("latency_ema_s", 0.0) or 0.0)
            degraded = bool(stats.get("degraded", False))
            available = bool(stats.get("available", True))
            if (not available) or degraded or latency >= 2.5:
                n_candidates = 1
                num_predict = int(min(num_predict, int(CFG.LLM_ACTION_BUDGET_EXPLORING)))

        return {
            "num_predict": num_predict,
            "temperature": temperature,
            "n_candidates": max(1, int(n_candidates))
        }

    def _score_action(self, action: Optional[str], game_mode: str, state: Dict, action_history: List[str]) -> float:
        if not action:
            return -1e9
        if self._would_loop(action, action_history):
            return -1e6

        score = 0.0
        last = action_history[-1] if action_history else None
        if action == last:
            score -= 1.0

        if game_mode == "battle":
            if action == "a":
                score += 3.0
            elif action in ("up", "down"):
                score += 1.0
            elif action in ("start", "select"):
                score -= 2.0
        elif game_mode == "menu":
            if action in ("up", "down", "left", "right"):
                score += 1.0
            elif action == "a":
                score += 1.5
            elif action == "b":
                score += 1.0
        else:
            if action in ("up", "down", "left", "right"):
                score += 1.0
            elif action == "a":
                score += 0.5
            elif action == "start":
                score -= 0.5

        text_box = str(state.get("text_box", "") or "").lower()
        if text_box and "non implementato" not in text_box:
            if action == "a":
                score += 0.5
            if action == "b":
                score -= 0.2

        return score

    def _select_best_candidate(
        self,
        candidate_indices: List[int],
        game_mode: str,
        state: Dict,
        action_history: List[str]
    ) -> Optional[int]:
        best_idx: Optional[int] = None
        best_score = -1e18
        seen: set[int] = set()
        accept_score = 1e18
        if game_mode == "battle":
            accept_score = 3.0
        elif game_mode == "menu":
            accept_score = 1.5

        for idx in candidate_indices:
            if not isinstance(idx, int) or idx < 0 or idx >= len(CFG.ACTIONS):
                continue
            if idx in seen:
                continue
            seen.add(idx)
            action = CFG.ACTIONS[idx]
            s = self._score_action(action, game_mode, state, action_history)
            if s > best_score:
                best_score = s
                best_idx = idx
                if s >= accept_score:
                    return best_idx

        return best_idx

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
        logger.info("Uso tool navigazione per: %s", task)
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
        
        profile = self._compute_profile(game_mode, state, task)
        num_predict = profile.get("num_predict", None)
        temperature = profile.get("temperature", None)
        n_candidates = int(profile.get("n_candidates", 1) or 1)

        if bool(CFG.PARALLEL_SAMPLING_ENABLED) and n_candidates > 1 and hasattr(self.llm, "get_action_candidates"):
            game_context = {"goal": task, **state}
            max_wait_s = 4.0 if game_mode in ("battle", "menu") else 6.0
            candidates = self._await_llm_with_ui_ticks(
                lambda: self.llm.get_action_candidates(
                    game_mode,
                    game_context,
                    screen_image_np,
                    action_history[-20:],
                    n_candidates=n_candidates,
                    num_predict=num_predict,
                    temperature=temperature
                ),
                max_wait_s=max_wait_s
            ) or []
            chosen = self._select_best_candidate(candidates, game_mode, state, action_history)
            if chosen is None and candidates:
                chosen = candidates[0]
            action_index = chosen
            logger.debug("Candidate actions: %s -> chosen=%s", candidates, action_index)
        elif hasattr(self.llm, "get_action"):
            game_context = {"goal": task, **state}
            max_wait_s = 3.0 if game_mode in ("battle", "menu") else 5.0
            action_index = self._await_llm_with_ui_ticks(
                lambda: self.llm.get_action(
                    game_mode,
                    game_context,
                    screen_image_np,
                    action_history[-20:],
                    num_predict=num_predict,
                    temperature=temperature
                ),
                max_wait_s=max_wait_s
            )
            if hasattr(self.llm, 'get_stats'):
                llm_stats = self.llm.get_stats()
                logger.debug("LLM stats: %s", llm_stats)

        action = None
        if isinstance(action_index, int) and 0 <= action_index < len(CFG.ACTIONS):
            action = CFG.ACTIONS[action_index]
            logger.debug("LLM azione: %s (index=%s)", action, action_index)
        else:
            logger.debug("LLM azione non valida (returned=%s)", action_index)

        used_fallback = False
        needs_fallback = (not action) or (isinstance(action, str) and self._would_loop(action, action_history))
        if needs_fallback:
            if not CFG.LLM_FALLBACK_ON_ERROR:
                self.step_count += 1
                return {"success": False, "steps": [], "reason": "LLM non ha prodotto un'azione valida.", "action_index": action_index}
            action = self._select_fallback_action(game_mode, state, action_history)
            used_fallback = True
            logger.debug(
                "Uso fallback: %s",
                "nessuna azione LLM" if not action else "azione causerebbe loop"
            )

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
            critique_response = self.llm.generate_text(prompt, num_predict=int(CFG.LLM_CRITIQUE_BUDGET))
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
