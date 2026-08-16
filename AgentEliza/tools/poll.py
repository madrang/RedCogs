"""The propose_choices tool: propose choices for an answer to a question."""

from ..polls import POLL_ANSWERS_MAX, POLL_ANSWER_MAX_CHARS, POLL_QUESTION_MAX_CHARS, POLL_VIEW_IDLE


class PollTools:
    """The propose_choices tool."""

    def poll_tools(self) -> list:
        """The OpenAI function schema of the choices tool."""
        return [
            {
                "type": "function"
                , "function": {
                    "name": "propose_choices"
                    , "description": (
                        "Propose choices for an answer to a question: a message with one button per choice. "
                        "The status arrives prepended to the next messages."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "question": {
                                "type": "string"
                                , "description": f"The question to answer, up to {POLL_QUESTION_MAX_CHARS} characters."
                            }
                            , "choices": {
                                "type": "array"
                                , "items": {"type": "string"}
                                , "description": f"The choices, 2 to {POLL_ANSWERS_MAX}, each up to {POLL_ANSWER_MAX_CHARS} characters."
                            }
                            , "multiple": {
                                "type": "boolean"
                                , "description": "Set true to allow more than one choice per person. Default: false."
                            }
                        }
                        , "required": ["question", "choices"]
                    }
                }
            }
        ]

    async def _tool_propose_choices(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return "Error: the question must be a non-empty string."
        question = " ".join(question.split())
        if len(question) > POLL_QUESTION_MAX_CHARS:
            return f"Error: the question is over {POLL_QUESTION_MAX_CHARS} characters."
        choices = arguments.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= POLL_ANSWERS_MAX:
            return f"Error: the choices must be a list of 2 to {POLL_ANSWERS_MAX} strings."
        cleaned = []
        for choice in choices:
            if not isinstance(choice, str) or not choice.strip():
                return "Error: every choice must be a non-empty string."
            choice = " ".join(choice.split())
            if len(choice) > POLL_ANSWER_MAX_CHARS:
                return f"Error: the choice {choice!r} is over {POLL_ANSWER_MAX_CHARS} characters."
            cleaned.append(choice)
        multiple = bool(arguments.get("multiple"))
        if self.polls is None:
            return "Error: the choices are not available."
        channel = await self.channel_getter(channel_id) if self.channel_getter else None
        if channel is None:
            return "Error: the current channel is unknown."
        session_id = channel_id if guild_id is not None else user_id
        error = await self.polls.create(session_id, channel, question, cleaned, multiple)
        if error:
            return error
        return (
            f"The choices for {question!r} have been posted with {len(cleaned)} options. "
            "The status arrives prepended to the next messages."
        )
