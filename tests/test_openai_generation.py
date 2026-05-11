import pytest

from data_generation.openai_generation import OpenAIGenerationClient, OpenAISettings


def test_openai_settings_loads_api_key_without_exposing_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    settings = OpenAISettings.from_env()

    assert settings.api_key == "sk-secret-value"
    assert settings.model == "gpt-5.5"
    assert "sk-secret-value" not in repr(settings)


def test_openai_settings_accepts_existing_open_ai_api_key_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_AI_API_KEY", "sk-alt-secret-value")

    settings = OpenAISettings.from_env(env_file=None)

    assert settings.api_key == "sk-alt-secret-value"


def test_openai_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAISettings.from_env(env_file=None)


def test_openai_generation_client_sends_structured_prompt_to_sdk():
    calls = []

    class FakeCompletions:
        def parse(self, **kwargs):
            calls.append(kwargs)

            class Message:
                parsed = {"examples": [{"prompt": "Say hi", "code": "with entry { print(\"hi\"); }", "complexity": "simple"}]}
                refusal = None
                content = '[{"prompt":"Say hi"}]'

            class Choice:
                message = Message()

            class Completion:
                choices = [Choice()]
                id = "completion-id"
                model = "gpt-5.5"

                def model_dump(self):
                    return {"id": self.id, "model": self.model}

            return Completion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeSdk:
        chat = FakeChat()

    prompt_request = {
        "system_prompt": "system",
        "user_prompt": "user",
        "response_schema": {"type": "array"},
        "category": "code_gen",
    }

    result = OpenAIGenerationClient(sdk_client=FakeSdk(), model="gpt-5.5").generate_batch(prompt_request)

    assert result.examples == [{"prompt": "Say hi", "code": "with entry { print(\"hi\"); }", "complexity": "simple"}]
    assert result.raw_response["id"] == "completion-id"
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["timeout"] == 90.0
    assert calls[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "code_gen_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "required": ["examples"],
                "additionalProperties": False,
                "properties": {"examples": {"type": "array"}},
            },
        },
    }


def test_openai_generation_client_unwraps_examples_from_content_when_sdk_parse_is_null():
    class FakeCompletions:
        def parse(self, **kwargs):
            class Message:
                parsed = None
                content = '{"examples":[{"prompt":"Say hi","code":"valid jac","complexity":"simple"}]}'

            class Choice:
                message = Message()

            class Completion:
                choices = [Choice()]

                def model_dump(self):
                    return {"id": "completion-id"}

            return Completion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeSdk:
        chat = FakeChat()

    result = OpenAIGenerationClient(sdk_client=FakeSdk(), timeout_seconds=12.0).generate_batch(
        {
            "system_prompt": "system",
            "user_prompt": "user",
            "response_schema": {"type": "array"},
            "category": "code_gen",
        }
    )

    assert result.examples == [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]


def test_openai_generation_client_uses_custom_timeout():
    calls = []

    class FakeCompletions:
        def parse(self, **kwargs):
            calls.append(kwargs)

            class Message:
                parsed = {"examples": []}
                content = ""

            class Choice:
                message = Message()

            class Completion:
                choices = [Choice()]

                def model_dump(self):
                    return {}

            return Completion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeSdk:
        chat = FakeChat()

    OpenAIGenerationClient(sdk_client=FakeSdk(), timeout_seconds=12.0).generate_batch(
        {
            "system_prompt": "system",
            "user_prompt": "user",
            "response_schema": {"type": "array"},
            "category": "conversion",
        }
    )

    assert calls[0]["timeout"] == 12.0


def test_openai_generation_client_retries_transient_failures_and_records_metadata():
    attempts = 0

    class FakeCompletions:
        def parse(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("temporary timeout")

            class Usage:
                prompt_tokens = 10
                completion_tokens = 5
                total_tokens = 15

            class Message:
                parsed = {"examples": [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]}
                content = ""
                refusal = None

            class Choice:
                message = Message()
                finish_reason = "stop"

            class Completion:
                choices = [Choice()]
                id = "completion-id"
                usage = Usage()

                def model_dump(self):
                    return {"id": self.id}

            return Completion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeSdk:
        chat = FakeChat()

    result = OpenAIGenerationClient(sdk_client=FakeSdk(), max_retries=1).generate_batch(
        {
            "system_prompt": "system",
            "user_prompt": "user",
            "response_schema": {"type": "array"},
            "category": "code_gen",
        }
    )

    assert attempts == 2
    assert result.raw_response["generation_metadata"]["retry_count"] == 1
    assert result.raw_response["generation_metadata"]["usage"]["total_tokens"] == 15
    assert result.raw_response["generation_metadata"]["finish_reason"] == "stop"


def test_openai_generation_client_stops_after_max_retries():
    class FakeCompletions:
        def parse(self, **kwargs):
            raise TimeoutError("temporary timeout")

    class FakeChat:
        completions = FakeCompletions()

    class FakeSdk:
        chat = FakeChat()

    with pytest.raises(TimeoutError):
        OpenAIGenerationClient(sdk_client=FakeSdk(), max_retries=1).generate_batch(
            {
                "system_prompt": "system",
                "user_prompt": "user",
                "response_schema": {"type": "array"},
                "category": "code_gen",
            }
        )
