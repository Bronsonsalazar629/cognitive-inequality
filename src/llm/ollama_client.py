"""
Ollama Local LLM Client

Implements BaseLLMClient for locally-running Ollama models.
Default model: qwen2.5-coder:7b (confirmed available via `ollama list`).

Usage:
    client = OllamaClient(model='qwen2.5-coder:7b')
    response = client.call_with_retry(prompt, system_instruction="...")
"""

import json
import logging
import time
from typing import Optional

import requests

from src.llm.llm_client_base import BaseLLMClient

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = 'http://localhost:11434'


class OllamaClient(BaseLLMClient):
    """LLM client for locally-running Ollama models."""

    def __init__(self, model: str = 'qwen2.5-coder:7b',
                 base_url: str = OLLAMA_BASE_URL,
                 enable_cache: bool = False):
        super().__init__(model=model, enable_cache=enable_cache)
        self.base_url = base_url
        self.provider_name = f'Ollama({model})'

    def call_with_retry(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Call local Ollama model via /api/chat endpoint.

        Combines system_instruction and prompt into a chat message pair.
        Retries up to max_retries times on connection/timeout errors.
        """
        messages = []
        if system_instruction:
            messages.append({'role': 'system', 'content': system_instruction})
        messages.append({'role': 'user', 'content': prompt})

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {'temperature': temperature},
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f'{self.base_url}/api/chat',
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data['message']['content']
                logger.info(f"Ollama response ({len(content)} chars) on attempt {attempt}")
                return content

            except requests.exceptions.ConnectionError:
                logger.error(
                    f"Ollama not reachable at {self.base_url}. "
                    f"Start with: ollama serve"
                )
                raise RuntimeError("Ollama server not running. Run: ollama serve")

            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout on attempt {attempt}/{max_retries}")
                if attempt == max_retries:
                    raise RuntimeError("Ollama timed out after max retries")
                time.sleep(2 ** attempt)

            except (requests.exceptions.HTTPError, KeyError) as e:
                logger.error(f"Ollama error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Ollama failed: {e}")
                time.sleep(1)

        raise RuntimeError("Ollama: exceeded max retries")

    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is loaded."""
        try:
            resp = requests.get(f'{self.base_url}/api/tags', timeout=5)
            if resp.status_code != 200:
                return False
            models = [m['name'] for m in resp.json().get('models', [])]
            return any(self.model in m for m in models)
        except Exception:
            return False
