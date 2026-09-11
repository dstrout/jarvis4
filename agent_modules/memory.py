"""
Memory Agent — extracts, consolidates, and retrieves long-term memories.

Runs asynchronously after each conversation turn to extract observations
about the user (preferences, habits, facts) and consolidate them into
the memory store. Also handles periodic personality evolution.

NOT a user-facing agent — this is internal infrastructure called by the
orchestrator client, not delegated to via tool calls.
"""

import json
import logging
import time
from pathlib import Path

from agents import Runner

from agents import ModelSettings

from core.agents_init import (
    MODEL_ORCHESTRATOR,
    AGENT_MODEL_SETTINGS,
    make_model,
    register_reasoning_settings,
)
from core.agent_base import make_agent
from core.memory_store import MemoryStore

log = logging.getLogger("jarvis.memory_agent")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_PERSONALITY_FILE = _PROJECT_ROOT / "personality.md"
_MEMORY_DB = _PROJECT_ROOT / "memory.db"

# ---------------------------------------------------------------------------
# Memory model settings — use gpt-oss for nuance detection
# ---------------------------------------------------------------------------

_MEMORY_MODEL = MODEL_ORCHESTRATOR  # gpt-oss-120b

# Low reasoning effort: extraction is a classification task, not a reasoning
# one. Registered so the Cerebras-only fields are stripped for other providers.
_MEMORY_SETTINGS = register_reasoning_settings(
    ModelSettings(temperature=0.3, top_p=0.9),
    {"reasoning_effort": "low", "reasoning_format": "hidden"},
)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a memory extraction system. Analyze the following conversation turn \
between a user and their AI assistant "Jarvis". Extract any observations worth \
remembering long-term.

Categories:
- preference: Things the user likes, dislikes, or prefers (e.g. "prefers dim lights after 9pm")
- habit: Patterns of behavior (e.g. "usually checks email first thing in the morning")
- fact: Concrete facts about the user (e.g. "works from a home office", "has two aquariums")
- personality: Observations about how the user communicates or what interaction style \
they respond well to (e.g. "appreciates dry humor", "wants brief answers not essays")

Rules:
- Only extract things clearly demonstrated by THIS exchange — don't speculate.
- Be specific and concrete, not vague.
- If the user corrects Jarvis, that's a strong signal — capture the correction.
- If the exchange is trivial small talk with no learnable signal, return NONE.
- Rate importance 1-10: mundane preferences (3-4), strong preferences or corrections (6-7), \
  critical facts or repeated patterns (8-10).

Return a JSON array of objects, or the string NONE if nothing worth remembering.
Each object has keys: "content", "category", "importance"

Conversation turn:
USER: {user_message}
JARVIS: {assistant_response}
"""


# ---------------------------------------------------------------------------
# Personality reflection prompt
# ---------------------------------------------------------------------------

_REFLECT_PROMPT = """\
You are maintaining a personality profile for an AI assistant named Jarvis. \
Below is the current personality file and a list of accumulated memories about \
the user and their interaction preferences.

Your job: rewrite the personality file to subtly incorporate what you've learned. \
Don't change the core character (dry wit, competent, slightly snarky). Instead, \
fine-tune the behavioral notes based on observed patterns.

For example, if memories show the user prefers very brief answers, add a note about \
brevity. If they respond well to specific types of humor, note that. If they hate \
certain phrasings, note that as an avoidance.

Keep the file under 2000 characters. Maintain the existing structure and voice. \
Changes should be subtle — this is evolution, not revolution.

CURRENT PERSONALITY FILE:
{personality}

ACCUMULATED MEMORIES ({count} total):
{memories}

Return ONLY the new personality file content — no commentary, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Memory Manager (not an Agent — a utility class)
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Manages the memory extraction pipeline and personality evolution.

    Called by OrchestratorClient after each conversation turn.
    """

    def __init__(self, store: MemoryStore | None = None):
        self._store = store or MemoryStore(db_path=_MEMORY_DB)
        self._turn_count = 0
        self._reflect_interval = 25  # Reflect every N turns
        self._last_reflect = 0
        self._personality_cache: str | None = None

        # Build extraction agent
        self._extractor = make_agent(
            name="memory_extractor",
            instructions="You extract memories from conversations. Always respond with valid JSON array or the word NONE.",
            model_id=_MEMORY_MODEL,
            model_settings=_MEMORY_SETTINGS,
        )

        # Build reflection agent
        self._reflector = make_agent(
            name="memory_reflector",
            instructions="You evolve AI personality profiles based on accumulated observations. Return only the updated file content.",
            model_id=_MEMORY_MODEL,
            model_settings=_MEMORY_SETTINGS,
        )

        log.info(
            "MemoryManager initialized: %d memories, reflect every %d turns",
            self._store.count(),
            self._reflect_interval,
        )

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def process_turn(self, user_message: str, assistant_response: str):
        """
        Process a conversation turn: extract memories, consolidate, and
        optionally trigger personality reflection.

        This is called asynchronously — it should not block the voice pipeline.
        """
        if not user_message or not assistant_response:
            return

        # Skip trivial exchanges
        if len(user_message) < 10 and len(assistant_response) < 50:
            return

        self._turn_count += 1
        t0 = time.perf_counter()

        try:
            await self._extract_and_store(user_message, assistant_response)
        except Exception as e:
            log.error("Memory extraction failed: %s", e, exc_info=True)

        # Periodic reflection
        if (
            self._turn_count - self._last_reflect >= self._reflect_interval
            and self._store.count() >= 5
        ):
            try:
                await self._reflect()
                self._last_reflect = self._turn_count
            except Exception as e:
                log.error("Personality reflection failed: %s", e, exc_info=True)

        elapsed = time.perf_counter() - t0
        log.debug("Memory processing took %.2fs", elapsed)

    async def _extract_and_store(self, user_message: str, assistant_response: str):
        """Extract memories from a conversation turn and consolidate."""
        prompt = _EXTRACT_PROMPT.format(
            user_message=user_message[:2000],
            assistant_response=assistant_response[:2000],
        )

        result = await Runner.run(self._extractor, prompt)
        output = (result.final_output or "").strip()

        if not output or output.upper() == "NONE":
            log.debug("No memories extracted from turn")
            return

        # Parse JSON — handle markdown fences the model might add
        output = output.strip("`")
        if output.startswith("json"):
            output = output[4:].strip()

        try:
            memories = json.loads(output)
        except json.JSONDecodeError:
            log.warning("Failed to parse extraction output: %s", output[:200])
            return

        if not isinstance(memories, list):
            memories = [memories]

        for mem_data in memories:
            content = mem_data.get("content", "").strip()
            category = mem_data.get("category", "fact")
            importance = float(mem_data.get("importance", 5))

            if not content:
                continue

            if category not in ("preference", "habit", "fact", "personality"):
                category = "fact"

            action, mem = self._store.consolidate(content, category, importance)
            log.info("Memory %s [%s]: %s", action, category, content[:80])

    async def _reflect(self):
        """Evolve the personality file based on accumulated memories."""
        personality = self.get_personality()
        if not personality:
            log.warning("No personality file found, skipping reflection")
            return

        all_memories = self._store.get_all()
        if len(all_memories) < 5:
            return

        # Format memories for the prompt
        mem_lines = []
        for mem in all_memories:
            mem_lines.append(
                f"[{mem.category}] (importance={mem.importance:.0f}) {mem.content}"
            )

        prompt = _REFLECT_PROMPT.format(
            personality=personality,
            count=len(all_memories),
            memories="\n".join(mem_lines),
        )

        result = await Runner.run(self._reflector, prompt)
        new_personality = (result.final_output or "").strip()

        if new_personality and len(new_personality) > 200:
            _PERSONALITY_FILE.write_text(new_personality)
            self._personality_cache = new_personality
            log.info("Personality file evolved (%d chars)", len(new_personality))
        else:
            log.warning("Reflection returned too-short output, skipping update")

    def get_personality(self) -> str | None:
        """Load the personality file (cached)."""
        if self._personality_cache is None:
            if _PERSONALITY_FILE.exists():
                self._personality_cache = _PERSONALITY_FILE.read_text()
        return self._personality_cache

    def get_relevant_memories(
        self,
        query: str,
        top_k: int = 8,
    ) -> str | None:
        """
        Retrieve relevant memories formatted for injection into the
        orchestrator's system prompt.

        Returns a formatted string or None if no memories exist.
        """
        if self._store.count() == 0:
            return None

        results = self._store.search(query, top_k=top_k)
        if not results:
            return None

        lines = []
        for mem, score in results:
            lines.append(f"- [{mem.category}] {mem.content}")

        return "\n".join(lines)

    def get_personality_block(self) -> str | None:
        """
        Return the personality file content for injection into the
        orchestrator's system prompt.
        """
        return self.get_personality()

    def close(self):
        self._store.close()
