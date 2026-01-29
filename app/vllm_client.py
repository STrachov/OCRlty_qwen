import os
from typing import Any, Dict, List, Optional

import httpx


def _trim(s: str, n: int = 4000) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "…<trimmed>")


class VLLMClient:
    """Minimal async client for vLLM OpenAI-compatible API."""

    def __init__(self) -> None:
        self.base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
        self.api_key = os.getenv("VLLM_API_KEY", "")
        self.timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "120"))

    def _headers(self, request_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
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
                headers=self._headers(request_id=request_id),
                json=payload,
            )

            if r.status_code >= 400:
                try:
                    body = r.json()
                    body_s = _trim(str(body))
                except Exception:
                    body_s = _trim(r.text)
                raise RuntimeError(f"vLLM HTTP {r.status_code}: {body_s}")

            return r.json()

    async def models(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
            if r.status_code >= 400:
                try:
                    body = r.json()
                    body_s = _trim(str(body))
                except Exception:
                    body_s = _trim(r.text)
                raise RuntimeError(f"vLLM HTTP {r.status_code}: {body_s}")
            return r.json()
