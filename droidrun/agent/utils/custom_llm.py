import asyncio
import logging
import requests
from typing import Any, List, Optional, AsyncGenerator
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
)

from pydantic import Field

logger = logging.getLogger("droidrun")

class SimpleCustomLLM(CustomLLM):
    """
    A simple wrapper for custom APIs that take a prompt and return text.
    """
    api_url: str = Field(description="The endpoint URL of the custom API")
    api_key: Optional[str] = Field(default=None, description="Optional API Key for Authorization header")
    model_name: str = Field(default="custom-api-model", description="The name of the model")
    prompt_key: str = Field(default="prompt", description="The JSON key for the prompt. If set to 'messages', it will wrap the prompt in OpenAI format.")
    response_key: str = Field(default="text", description="The JSON key for the response text. Supports dots for nesting and numbers for indices (e.g., 'choices.0.message.content')")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            is_chat_model=False,
        )


    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            # Prepare Headers
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Prepare Payload
            if self.prompt_key == "messages":
                # Special handling for OpenAI-like local formats
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}]
                }
            else:
                payload = {self.prompt_key: prompt}
            
            # Merge additional model parameters (temperature, max_tokens, etc.)
            payload.update(kwargs)
            
            logger.debug(f"Calling Custom API: {self.api_url}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Precise parsing of nested keys and list indices
            text = data
            for key in self.response_key.split('.'):
                if isinstance(text, dict):
                    text = text.get(key, "")
                elif isinstance(text, list) and key.isdigit():
                    idx = int(key)
                    text = text[idx] if idx < len(text) else ""
                else:
                    break
            
            return CompletionResponse(text=str(text))
        except Exception as e:
            logger.error(f"Error calling Custom API: {e}")
            return CompletionResponse(text=f"Error: {str(e)}")


    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Convert chat messages to a single flattened prompt
        flattened_prompt = ""
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            flattened_prompt += f"{role.upper()}: {msg.content}\n"
        
        # Append assistant trigger
        if not flattened_prompt.strip().endswith("ASSISTANT:"):
            flattened_prompt += "ASSISTANT: "
            
        completion = self.complete(flattened_prompt, **kwargs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=completion.text),
            raw=completion
        )

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Convert chat messages to a single flattened prompt
        flattened_prompt = ""
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            flattened_prompt += f"{role.upper()}: {msg.content}\n"
        
        # Append assistant trigger
        if not flattened_prompt.strip().endswith("ASSISTANT:"):
            flattened_prompt += "ASSISTANT: "
            
        completion = await self.acomplete(flattened_prompt, **kwargs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=completion.text),
            raw=completion
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, **kwargs)
        yield CompletionResponse(text=response.text, delta=response.text, raw=response.raw)

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        response = self.chat(messages, **kwargs)
        yield ChatResponse(
            message=response.message,
            delta=response.message.content,
            raw=response.raw
        )

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        response = await self.achat(messages, **kwargs)
        yield ChatResponse(
            message=response.message,
            delta=response.message.content,
            raw=response.raw
        )

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        response = await self.acomplete(prompt, **kwargs)
        yield CompletionResponse(
            text=response.text,
            delta=response.text,
            raw=response.raw
        )

class GroqSDKLLM(CustomLLM):
    """
    LLM wrapper using the official Groq SDK.
    Supports advanced parameters like reasoning_effort.
    """
    model: str = Field(description="The Groq model name")
    api_key: str = Field(description="Groq API Key")
    temperature: float = Field(default=1.0)
    max_tokens: Optional[int] = Field(default=None)
    reasoning_effort: Optional[str] = Field(default=None)
    additional_kwargs: dict = Field(default_factory=dict)

    _client: Any = None
    _aclient: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        from groq import Groq, AsyncGroq
        self._client = Groq(api_key=self.api_key)
        self._aclient = AsyncGroq(api_key=self.api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            is_chat_model=True,
        )

    def _prepare_params(self, messages: List[ChatMessage], **kwargs) -> dict:
        msgs = [{"role": m.role.value if hasattr(m.role, 'value') else str(m.role), "content": m.content} for m in messages]
        params = {
            "model": self.model,
            "messages": msgs,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": kwargs.get("stream", False),
        }

        # Add optional params
        if self.max_tokens: params["max_completion_tokens"] = self.max_tokens
        if self.reasoning_effort: params["reasoning_effort"] = self.reasoning_effort
        
        # Merge additional kwargs
        params.update(self.additional_kwargs)
        params.update(kwargs)
        return params


    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        params = self._prepare_params(messages, stream=False, **kwargs)
        response = self._client.chat.completions.create(**params)
        content = response.choices[0].message.content
        return ChatResponse(message=ChatMessage(role="assistant", content=content), raw=response)


    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        params = self._prepare_params(messages, stream=False, **kwargs)
        response = await self._aclient.chat.completions.create(**params)
        content = response.choices[0].message.content
        return ChatResponse(message=ChatMessage(role="assistant", content=content), raw=response)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        res = self.chat(messages, **kwargs)
        return CompletionResponse(text=res.message.content, raw=res.raw)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        res = await self.achat(messages, **kwargs)
        return CompletionResponse(text=res.message.content, raw=res.raw)

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        params = self._prepare_params(messages, stream=True, **kwargs)
        response = await self._aclient.chat.completions.create(**params)
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=delta),
                delta=delta,
                raw=chunk
            )

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        params = self._prepare_params(messages, stream=True, **kwargs)
        response = self._client.chat.completions.create(**params)
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=delta),
                delta=delta,
                raw=chunk
            )

    async def astream_complete(self, prompt: str, **kwargs: Any) -> Any:
        messages = [ChatMessage(role="user", content=prompt)]
        async for response in self.astream_chat(messages, **kwargs):
            yield CompletionResponse(text=response.message.content, delta=response.delta, raw=response.raw)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        messages = [ChatMessage(role="user", content=prompt)]
        for response in self.stream_chat(messages, **kwargs):
            yield CompletionResponse(text=response.message.content, delta=response.delta, raw=response.raw)


