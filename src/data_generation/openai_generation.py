from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str = "gpt-5.5"

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = ".env", model: str | None = None) -> "OpenAISettings":
        if env_file is not None:
            load_dotenv(env_file)
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for live OpenAI generation")
        return cls(api_key=api_key, model=model or os.environ.get("OPENAI_MODEL", "gpt-5.5"))

    def __repr__(self) -> str:
        return f"OpenAISettings(api_key='<redacted>', model={self.model!r})"


@dataclass(frozen=True)
class GenerationResult:
    examples: list[dict[str, Any]] | str
    raw_response: dict[str, Any]


class OpenAIGenerationClient:
    def __init__(
        self,
        *,
        sdk_client: Any | None = None,
        model: str = "gpt-5.5",
        api_key: str | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(api_key=api_key)
        self.sdk_client = sdk_client
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_batch(self, prompt_request: dict[str, Any]) -> GenerationResult:
        completion = self.sdk_client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_request["system_prompt"]},
                {"role": "user", "content": prompt_request["user_prompt"]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": f"{prompt_request['category']}_batch",
                    "strict": True,
                    "schema": _structured_output_schema(prompt_request["response_schema"]),
                },
            },
            timeout=self.timeout_seconds,
        )
        message = completion.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, dict) and "examples" in parsed:
            examples = parsed["examples"]
        elif parsed is not None:
            examples = parsed
        else:
            examples = _examples_from_content(getattr(message, "content", ""))
        raw_response = completion.model_dump() if hasattr(completion, "model_dump") else {"model": self.model}
        return GenerationResult(examples=examples, raw_response=raw_response)


def _structured_output_schema(category_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["examples"],
        "additionalProperties": False,
        "properties": {
            "examples": category_schema,
        },
    }


def _examples_from_content(content: str) -> list[dict[str, Any]] | str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict) and isinstance(parsed.get("examples"), list):
        return parsed["examples"]
    if isinstance(parsed, list):
        return parsed
    return content
