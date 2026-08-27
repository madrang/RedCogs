import asyncio
import contextlib
import time

# Assumed provider prompt-cache lifetime when the provider documents none.
# 5 minutes gives a user time to type a reply. A provider overrides this
# with its `cache_ttl` class attribute once the real value is known.
DEFAULT_CACHE_TTL = 300
# The sweeper compacts a session at this fraction of the cache lifetime,
# while the provider prompt cache is still warm for the summarization call.
# 0.8 of 5 minutes is 4 minutes.
COMPACTION_AT = 0.8
# Fallback context assumption for a model with no known context size:
# 256K tokens, of which the history gets about half. A known context size
# (provider context_lengths) replaces this with the CONTEXT_FILL fraction.
CONTEXT_TOKENS = 256_000
CHARS_PER_TOKEN = 4
# The real token count is the trigger when the API reports one. The
# character count is the fallback.
HISTORY_MAX_TOKENS = CONTEXT_TOKENS // 2
HISTORY_MAX_CHARS = CONTEXT_TOKENS // 2 * CHARS_PER_TOKEN
# Fraction of a known model context the session may fill before compaction.
CONTEXT_FILL = 0.8
# Cap of the rolling summary text.
SUMMARY_MAX_CHARS = 4000

# The harness request that asks for the summary. It goes to the API as a
# user message after the untouched session, and it introduces the summary
# inside the session as the compaction exchange: the harness request,
# answered by the agent with the summary as its own reply. Harness messages
# use the same square-bracket tags as the memory notes.
COMPACT_REQUEST = (
    "[harness] condense this conversation into a short summary. "
    "Keep names, facts, decisions, and open questions. Drop small talk. "
    "Keep the character of the exchange too: the tone, the mood of each person, "
    "the state of the relationship, and the shared references that give the conversation continuity. "
    "When the conversation holds a summary from an earlier condense, merge its content into the new summary. [/harness]"
)
# Context backfill target of a fresh session: recent channel messages from
# Discord. eliza.py scans the channel history for them.
BACKFILL_MESSAGES = 64
# Cap of the wait for the session lock. 30 minutes is far above the
# longest reply or compaction: a wait that long means something is
# seriously wrong, and the waiter fails with TimeoutError instead of
# deadlocking.
LOCK_ACQUIRE_TIMEOUT = 1800


@contextlib.asynccontextmanager
async def _acquire(lock: asyncio.Lock):
    """One lock acquisition with the timeout. The held section is not timed."""
    async with asyncio.timeout(LOCK_ACQUIRE_TIMEOUT):
        await lock.acquire()
    try:
        yield
    finally:
        lock.release()
# Recent turns kept verbatim through a compaction: as many as the backfill
# restores on a fresh session.
COMPACTION_KEEP_TURNS = BACKFILL_MESSAGES


class Session:
    """One conversation context: a guild session, or the DM session of a user.

    messages[0] is the system message of the context: the system prompt
    and the memory blocks. It is written when the context starts and
    rebuilt only when the context expires (idle past the cache lifetime)
    or after a compaction. The rest of messages holds the turns kept
    verbatim. Tool exchanges (the assistant calls and the tool results) and
    the reasoning fields stay in the turns until a compaction summarizes
    them. The summary joins the context as a turn, not in the system
    message: inject_summary places it after the system message as the
    compaction exchange (harness request, agent answer), so a rebuilt
    context keeps it without repeating it. size counts the characters of
    all messages, not their number: many small messages and a few giant
    ones do not cost the same context. scope is the Memory scope the
    session belongs to: guild, or user for a DM. summary is the condensed
    form of the compacted turns; it persists in the Memory store of the
    scope, so a reboot loses only the verbatim turns, never the summary.
    seen_users tracks the users with an injected memory note in this
    context, so the note appears once per user per context. last_prompt_tokens
    is the real prompt size of the last API answer, 0 when unknown.
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

    def acquire(self):
        """The session lock with a timeout on the wait.

        The wait caps at LOCK_ACQUIRE_TIMEOUT (30 minutes), far above the
        longest reply or compaction: a timeout means something is
        seriously wrong, and the waiter fails with TimeoutError instead of
        deadlocking. The held section is not timed.
        """
        return _acquire(self.lock)

    def start_context(self, system_text: str) -> None:
        """Open or refresh the context: replace the system message, keep the turns."""
        turns = self.messages[1:] if self.messages else []
        self.messages = [{"role": "system", "content": system_text}, *turns]
        self.size = len(system_text) + sum(self._size_of(message) for message in turns)
        self.seen_users.clear()
        self.touch()

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.size += len(content)
        self.touch()

    @staticmethod
    def _size_of(message: dict) -> int:
        """The size of one turn: the content, plus the reasoning when the turn keeps it."""
        return len(message.get("content") or "") + len(message.get("reasoning_content") or "") + len(message.get("reasoning") or "")

    def append_message(self, message: dict) -> None:
        """Store one turn as the API sent it: role, content, and the extra fields (tool calls, reasoning)."""
        self.messages.append(message)
        self.size += self._size_of(message)
        self.touch()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def idle(self) -> float:
        """Seconds since the last activity."""
        return time.monotonic() - self.last_active

    def plan_compaction(self, keep: int = COMPACTION_KEEP_TURNS):
        """The block to summarize: the turns after the system message, minus the recent tail.

        A session that unloads compacts every turn: keep=0. The boundary
        never splits a tool exchange: a tool result without its call breaks
        the next request. The boundary moves back to the exchange start.
        """
        if len(self.messages) <= 1 + keep:
            return []
        if not keep:
            return self.messages[1:]
        cut = len(self.messages) - keep
        while cut > 1 and self.messages[cut].get("role") == "tool":
            cut -= 1
        if cut <= 1:
            return []
        return self.messages[1:cut]

    def inject_summary(self) -> None:
        """Place the summary into the turns as the compaction exchange, after the system message."""
        trace = [
            {"role": "user", "content": COMPACT_REQUEST},
            {"role": "assistant", "content": self.summary},
        ]
        self.messages[1:1] = trace
        self.size += sum(self._size_of(turn) for turn in trace)

    def apply_compaction(self, summary: str, dropped: int) -> None:
        """Drop the summarized turns and store the summary.

        Turns that arrived after the plan stay in the session. The real
        prompt token count is unknown again until the next answer.
        """
        dropped = min(dropped, len(self.messages) - 1)
        self.messages = self.messages[:1] + self.messages[1 + dropped:]
        self.size = sum(self._size_of(message) for message in self.messages)
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

    def needs_compaction(self, session: Session, cache_ttl: int, context_tokens: int | None = None) -> bool:
        """True when the session should be compacted.

        Size trigger: the real prompt token count of the last answer when
        the API reports one, the character estimate otherwise. The budget is
        CONTEXT_FILL of the model context when context_tokens is known, the
        HISTORY_MAX fallback otherwise. Idle trigger: the session went idle
        past COMPACTION_AT of the provider cache lifetime, so compacting
        still hits the warm prompt cache for the summarization call. Used by
        the reply path and by the background sweeper.
        """
        if len(session.messages) <= 1:
            return False
        if context_tokens:
            max_tokens = int(context_tokens * CONTEXT_FILL)
            max_chars = max_tokens * CHARS_PER_TOKEN
        else:
            max_tokens = HISTORY_MAX_TOKENS
            max_chars = HISTORY_MAX_CHARS
        if session.last_prompt_tokens:
            if session.last_prompt_tokens >= max_tokens:
                return True
        elif session.size >= max_chars:
            return True
        return session.idle() >= cache_ttl * COMPACTION_AT
