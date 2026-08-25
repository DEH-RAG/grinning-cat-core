if __name__ == "__main__":
    import asyncio
    import json

    import jsonschema
    from pydantic import BaseModel

    from cat.plugins.structured_llm.models import SentimentResult
    from cat.plugins.structured_llm.structured_llm import (
        StructuredLLM,
        StructuredLLMError,
    )

    NATIVE_JSON = '{"sentiment": "positive", "confidence": 0.95, "reason": "Great experience."}'
    BAD_TEXT = "this is definitely not json"

    PERSON_SCHEMA = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    PERSON_JSON = '{"name": "Alice", "age": 30}'
    PERSON_BAD_JSON = '{"name": "Alice", "age": "thirty"}'


    class StubNativeLLM:
        def with_structured_output(self, output_model):
            if not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
                raise TypeError("output_model is not a valid Pydantic model")
            bound = self._Bounded(output_model)
            return bound

        class _Bounded:
            def __init__(self, output_model):
                self._model = output_model

            async def ainvoke(self, input, **kwargs):
                return self._model(sentiment="positive", confidence=0.95, reason="Great experience.")


    class StubNativeDictLLM:
        def with_structured_output(self, output_model):
            bound = self._Bounded()
            return bound

        class _Bounded:
            async def ainvoke(self, input, **kwargs):
                return {"name": "Alice", "age": 30}


    class StubTextLLM:
        def __init__(self, response):
            self._response = response

        async def ainvoke(self, input, **kwargs):
            return self._response


    async def scenario_native_happy():
        stub = StubNativeLLM()
        result = await StructuredLLM().run("Classify: I love it", stub, SentimentResult)
        assert isinstance(result.model, SentimentResult), "model is not SentimentResult"
        parsed = json.loads(result.text)
        assert parsed == json.loads(NATIVE_JSON), "text is not the expected JSON"
        return "PASS: native happy path (with_structured_output -> valid model)"


    async def scenario_fallback_happy():
        stub = StubTextLLM(NATIVE_JSON)
        result = await StructuredLLM().run("Classify: I love it", stub, SentimentResult)
        assert isinstance(result.model, SentimentResult), "model is not SentimentResult"
        json.loads(result.text)
        return "PASS: fallback happy path (ainvoke JSON string -> parsed model)"


    async def scenario_failure_non_json():
        stub = StubTextLLM(BAD_TEXT)
        try:
            await StructuredLLM().run("Classify: whatever", stub, SentimentResult)
        except StructuredLLMError as e:
            assert e.raw_text == BAD_TEXT, "raw_text does not match the bad string"
            return "PASS: non-JSON output raises StructuredLLMError with raw_text"
        raise AssertionError("expected StructuredLLMError was not raised")


    async def scenario_invalid_model():
        stub = StubNativeLLM()
        try:
            await StructuredLLM().run("Classify: whatever", stub, int)
        except StructuredLLMError:
            return "PASS: invalid output_model=int raises StructuredLLMError (no uncaught TypeError)"
        raise AssertionError("expected StructuredLLMError was not raised")


    async def scenario_native_happy_dict():
        stub = StubNativeDictLLM()
        result = await StructuredLLM().run("Extract the person", stub, PERSON_SCHEMA)
        assert result.model == {"name": "Alice", "age": 30}, "model is not the raw dict"
        parsed = json.loads(result.text)
        jsonschema.validate(parsed, PERSON_SCHEMA)
        return "PASS: native happy path with raw dict schema (returns conforming dict)"


    async def scenario_fallback_happy_dict():
        stub = StubTextLLM(PERSON_JSON)
        result = await StructuredLLM().run("Extract the person", stub, PERSON_SCHEMA)
        assert result.model == {"name": "Alice", "age": 30}, "model is not the parsed dict"
        jsonschema.validate(result.model, PERSON_SCHEMA)
        return "PASS: fallback happy path with raw dict schema (parsed + validated)"


    async def scenario_fallback_failure_dict():
        stub = StubTextLLM(PERSON_BAD_JSON)
        try:
            await StructuredLLM().run("Extract the person", stub, PERSON_SCHEMA)
        except StructuredLLMError as e:
            assert e.raw_text == PERSON_BAD_JSON, "raw_text does not match the bad json"
            return "PASS: fallback dict violating schema raises StructuredLLMError with raw_text"
        raise AssertionError("expected StructuredLLMError was not raised")


    async def main():
        results = await asyncio.gather(
            scenario_native_happy(),
            scenario_fallback_happy(),
            scenario_failure_non_json(),
            scenario_invalid_model(),
            scenario_native_happy_dict(),
            scenario_fallback_happy_dict(),
            scenario_fallback_failure_dict(),
        )
        for line in results:
            print(line)
        return 0


    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001 - test harness must turn ANY failure into a non-zero exit
        print(f"FAIL: {exc}")
        raise SystemExit(1)