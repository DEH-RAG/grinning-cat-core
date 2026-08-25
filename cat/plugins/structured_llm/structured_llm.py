import json
from dataclasses import dataclass
from typing import Any

import jsonschema
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from cat.utils import parse_json


class StructuredLLMError(Exception):
    """Raised when the LLM output cannot be parsed/validated into the output model."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class StructuredLLMResult:
    text: str
    model: Any | None


class StructuredLLM:
    async def run(
        self,
        prompt: str,
        llm: Any,
        output_model: dict[str, Any] | type[BaseModel],  # accepts raw JSON-schema dict or Pydantic class
        system_prompt: str | None = None,
    ) -> StructuredLLMResult:
        if hasattr(llm, "with_structured_output"):
            return await self._run_native(prompt, llm, output_model)
        return await self._run_fallback(prompt, llm, output_model, system_prompt)

    @staticmethod
    def _is_pydantic(obj) -> bool:
        return hasattr(obj, "model_json_schema")

    async def _run_native(self, prompt: str, llm: Any, output_model: Any) -> StructuredLLMResult:
        try:
            bounded = llm.with_structured_output(output_model)
            res = await bounded.ainvoke(prompt, config=RunnableConfig())

            if hasattr(res, "model_dump_json"):
                text = res.model_dump_json()
                model = res
            elif isinstance(res, dict):
                text = json.dumps(res)
                if self._is_pydantic(output_model):
                    model = output_model(**res)
                else:
                    model = res
            else:
                text = str(res)
                model = None
        except Exception as e:
            raw = getattr(e, "raw_text", "")
            raise StructuredLLMError(f"Native structured output failed: {e}", raw)

        return StructuredLLMResult(text=text, model=model)

    async def _run_fallback(
        self, prompt: str, llm: Any, output_model: Any, system_prompt: str | None
    ) -> StructuredLLMResult:
        if self._is_pydantic(output_model):
            schema = output_model.model_json_schema()
        else:
            schema = output_model
        full_prompt = self._build_json_prompt(schema, prompt, system_prompt)

        raw = await llm.ainvoke(full_prompt, config=RunnableConfig())
        text = getattr(raw, "content", str(raw))

        try:
            if self._is_pydantic(output_model):
                model = parse_json(text, output_model)
            else:
                model = json.loads(text)
                jsonschema.validate(model, output_model)
        except jsonschema.exceptions.ValidationError as e:
            raise StructuredLLMError(f"Failed to validate LLM output against JSON schema: {e}", text)
        except Exception as e:
            raise StructuredLLMError(f"Failed to parse LLM output as structured JSON: {e}", text)

        return StructuredLLMResult(text=text, model=model)

    @staticmethod
    def _build_json_prompt(schema: dict, prompt: str, system_prompt: str | None) -> str:
        schema_json = json.dumps(schema)
        if system_prompt:
            return (
                f"{system_prompt}\n\n"
                f"Return ONLY a single valid JSON object conforming to this JSON schema:\n"
                f"```json\n{schema_json}\n```\n\n"
                f"Input:\n{prompt}"
            )
        return (
            "Return ONLY a single valid JSON object conforming to this JSON schema:\n"
            f"```json\n{schema_json}\n```\n\n"
            f"Input:\n{prompt}"
        )