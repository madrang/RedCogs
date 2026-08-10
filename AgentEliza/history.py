import asyncio
import time

# Assumed provider prompt-cache lifetime when the provider documents none.
# 5 minutes gives a user time to type a reply. A provider overrides this
# with its `cache_ttl` class attribute once the real value is known.
DEFAULT_CACHE_TTL = 300
# Smallest agent context among the providers: 256K tokens.
CONTEXT_TOKENS = 256_000
CHARS_PER_TOKEN = 4
# The history gets about half of the context. The real token count is the
# trigger when the API reports one. The character count is the fallback.
HISTORY_MAX_TOKENS = CONTEXT_TOKENS // 2
HISTORY_MAX_CHARS = CONTEXT_TOKENS // 2 * CHARS_PER_TOKEN
# Cap of the rolling summary text.
SUMMARY_MAX_CHARS = 4000

COMPACT_PROMPT = (
    "Condense this conversation into a short factual summary. "
    "Keep names, facts, decisions, and open questions. Drop small talk. "
    "Merge it with the summary so far when one is given."
)


class Session:
    """One conversation context: a guild session, or the DM session of a user.

    messages[0] is the system message of the context: the system prompt,
    the memory blocks, and the conversation summary. It is written when the
    context starts and rebuilt only when the context expires (idle past the
    cache lifetime) or after a compaction. The rest of messages holds the
    turns kept verbatim. size counts the characters of all messages, not
    their number: many small messages and a few giant ones do not cost the
    same context. scope is the Memory scope the session belongs to: guild,
    or user for a DM. summary is the condensed form of the compacted turns;
    it persists in the Memory store of the scope, so a reboot loses only
    the verbatim turns, never the summary. seen_users tracks which users
    got their memory injected in this context, so it happens once per user
    per context. last_prompt_tokens is the real prompt size of the last
    API answer, 0 when unknown.
    """

    def __init__(self, scope: str):
        self.scope = scope
        self.messages = []
        self.size = 0
        self.last_active = 0.0
        self.summary = ""
        self.seen_users = set()
        # The real prompt size of the last API answer, 0 when unknown.
        self.last_prompt_tokens = 0
        # One chat request per session at a time: replies, compactions, and the sweeper queue on it.
        self.lock = asyncio.Lock()
        # Stamp of the last compaction, so the sweeper does not re-fire on an idle session.
        self.last_compaction = 0.0
        # The last compaction error, None when the last compaction worked.
        self.error = None

    def start_context(self, system_text: str) -> None:
        """Open or refresh the context: replace the system message, keep the turns."""
        turns = self.messages[1:] if self.messages else []
        self.messages = [{"role": "system", "content": system_text}, *turns]
        self.size = len(system_text) + sum(len(message["content"]) for message in turns)
        self.seen_users.clear()
        self.touch()

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.size += len(content)
        self.touch()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def idle(self) -> float:
        """Seconds since the last activity."""
        return time.monotonic() - self.last_active

    def plan_compaction(self):
        """The block to summarize: every turn after the system message."""
        return self.messages[1:]

    def apply_compaction(self, summary: str, dropped: int) -> None:
        """Drop the summarized turns and store the summary.

        Turns that arrived after the plan stay in the session. The real
        prompt token count is unknown again until the next answer.
        """
        dropped = min(dropped, len(self.messages) - 1)
        self.messages = self.messages[:1] + self.messages[1 + dropped:]
        self.size = sum(len(message["content"]) for message in self.messages)
        self.summary = summary
        self.seen_users.clear()
        self.last_compaction = time.monotonic()
        self.last_prompt_tokens = 0


class History:
    """The conversation sessions of the AgentEliza cog, in memory only.

    Holds the Memory reference so a new session reads back its persisted
    summary on creation.
    """

    def __init__(self, memory):
        self.sessions: dict[int, Session] = {}
        self.memory = memory

    async def get(self, session_id: int, scope: str) -> Session:
        """The session of an id, creating it empty. scope: guild, or user for a DM."""
        session = self.sessions.get(session_id)
        if session is None:
            session = Session(scope)
            self.sessions[session_id] = session
            # A reboot loses the RAM session, not the summary: read it back on creation.
            session.summary = await self.memory.read_summary(scope, session_id)
        return session

    def needs_compaction(self, session: Session, cache_ttl: int) -> bool:
        """True when the session should be compacted.

        Size trigger: the real prompt token count of the last answer when
        the API reports one, the character estimate otherwise. Idle trigger:
        the session went idle past half the provider cache lifetime, so
        compacting still hits the warm prompt cache for the summarization
        call. Used by the reply path and by the background sweeper.
        """
        if len(session.messages) <= 1:
            return False
        if session.last_prompt_tokens:
            if session.last_prompt_tokens >= HISTORY_MAX_TOKENS:
                return True
        elif session.size >= HISTORY_MAX_CHARS:
            return True
        return session.idle() >= cache_ttl / 2
