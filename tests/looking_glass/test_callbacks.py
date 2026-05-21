import json
from unittest.mock import AsyncMock
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from cat.looking_glass.callbacks import WebSocketCallbackManager, ThinkingMessage
from cat.startup import patch_reasoning_content_monkeypatch


async def test_monkeypatch_propagates_reasoning_content():
    """_convert_delta_to_message_chunk preserves reasoning_content in additional_kwargs
    after patch_reasoning_content_monkeypatch() is applied."""
    from langchain_community.chat_models import openai as openai_llm

    # Apply monkeypatch in this test directly (no app lifespan needed)
    patch_reasoning_content_monkeypatch()

    delta = {
        "id": "test",
        "role": "assistant",
        "content": "Hello",
        "reasoning_content": "Let me think about this",
    }
    result = openai_llm._convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert result.additional_kwargs.get("reasoning_content") == "Let me think about this"


async def test_on_llm_new_token_forwards_reasoning_content():
    """on_llm_new_token sends llm_thinking when chunk.message.additional_kwargs
    contains reasoning_content."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    msg = AIMessageChunk(
        content="Hello",
        additional_kwargs={"reasoning_content": "I should check the docs"},
    )
    chunk = ChatGenerationChunk(message=msg)

    await cb.on_llm_new_token("irrelevant", chunk=chunk)

    # Should have sent llm_thinking, NOT chat_token
    assert mock_notifier.send_llm_thinking.called
    assert not mock_notifier.send_chat_token.called

    sent_json = mock_notifier.send_llm_thinking.call_args[0][0]
    sent = json.loads(sent_json)
    assert sent["content"] == "I should check the docs"
    assert sent["step"] == 0
    assert cb._thinking_streamed is True


async def test_on_llm_new_token_reasoning_takes_priority_over_think_tags():
    """reasoning_content from additional_kwargs is handled BEFORE DeepSeek  thinking tags."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    msg = AIMessageChunk(
        content="Hello",
        additional_kwargs={"reasoning_content": "deep analysis"},
    )
    chunk = ChatGenerationChunk(message=msg)

    await cb.on_llm_new_token(" thinking", chunk=chunk)

    # reasoning_content path should fire, not the  thinking tag path
    assert mock_notifier.send_llm_thinking.called
    sent_json = mock_notifier.send_llm_thinking.call_args[0][0]
    assert "deep analysis" in sent_json


async def test_on_llm_new_token_no_reasoning_passes_through():
    """Without reasoning_content, the callback falls through to normal token handling."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    msg = AIMessageChunk(content="Hello", additional_kwargs={})
    chunk = ChatGenerationChunk(message=msg)

    await cb.on_llm_new_token("Hello", chunk=chunk)

    assert not mock_notifier.send_llm_thinking.called
    assert mock_notifier.send_chat_token.called


async def test_on_llm_new_token_no_chunk_passes_through():
    """When no chunk is provided, the callback falls through (Anthropic-style)."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    await cb.on_llm_new_token("Hello", chunk=None)

    assert not mock_notifier.send_llm_thinking.called
    assert mock_notifier.send_chat_token.called


async def test_on_llm_new_token_empty_reasoning_string():
    """Empty reasoning_content string should not trigger llm_thinking."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    msg = AIMessageChunk(
        content="Hello",
        additional_kwargs={"reasoning_content": ""},
    )
    chunk = ChatGenerationChunk(message=msg)

    await cb.on_llm_new_token("Hello", chunk=chunk)

    assert not mock_notifier.send_llm_thinking.called
    assert mock_notifier.send_chat_token.called


async def test_reasoning_content_increments_step():
    """Each reasoning chunk should get the same step (not incremented per chunk)."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    # First call
    msg1 = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "Step 1 reasoning"},
    )
    await cb.on_llm_new_token("", chunk=ChatGenerationChunk(message=msg1))

    # Second call — step should still be 0 (not incremented per chunk)
    msg2 = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "more reasoning"},
    )
    await cb.on_llm_new_token("", chunk=ChatGenerationChunk(message=msg2))

    assert mock_notifier.send_llm_thinking.call_count == 2
    first_call = json.loads(mock_notifier.send_llm_thinking.call_args_list[0][0][0])
    second_call = json.loads(mock_notifier.send_llm_thinking.call_args_list[1][0][0])
    assert first_call["step"] == 0
    assert second_call["step"] == 0


async def test_reasoning_content_resets_across_llm_calls():
    """Step should reset on a new chat model start."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    # First LLM call
    msg = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "first reasoning"},
    )
    await cb.on_llm_new_token("", chunk=ChatGenerationChunk(message=msg))
    assert cb._thinking_step == 0

    # New LLM call resets
    await cb.on_chat_model_start({}, [[]])
    assert cb._thinking_step == 0
    assert cb._thinking_streamed is False

    # Second LLM call reasoning
    msg2 = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "second reasoning"},
    )
    await cb.on_llm_new_token("", chunk=ChatGenerationChunk(message=msg2))
    sent = json.loads(mock_notifier.send_llm_thinking.call_args_list[-1][0][0])
    assert sent["step"] == 0


async def test_on_llm_end_skipped_when_streamed():
    """on_llm_end should NOT re-send reasoning_content when _thinking_streamed is True."""
    mock_notifier = AsyncMock()
    cb = WebSocketCallbackManager(mock_notifier)

    msg = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "streamed reasoning"},
    )
    await cb.on_llm_new_token("", chunk=ChatGenerationChunk(message=msg))

    assert cb._thinking_streamed is True

    from langchain_core.outputs import LLMResult

    # Now on_llm_end is called — should skip because already streamed
    await cb.on_llm_end(LLMResult(generations=[]))

    # Only the streaming call
    assert mock_notifier.send_llm_thinking.call_count == 1
