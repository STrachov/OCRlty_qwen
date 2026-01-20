import os
from typing import Any, Dict, List, Optional

import httpx


class VLLMClient:
    """Minimal async client for vLLM OpenAI-compatible API."""

    def __init__(self) -> None:
        self.base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
        self.api_key = os.getenv("VLLM_API_KEY", "")
        self.timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "120"))

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            r.raise_for_status()
            return r.json()

    async def models(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
            r.raise_for_status()
            return r.json()
