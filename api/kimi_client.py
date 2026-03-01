"""Kimi Code Plan (K2.5) ModelClient integration.

Uses OpenAI SDK compatibility to talk to Kimi's Coding API.
Falls back to Anthropic Messages format via AnthropicAPIClient
if OpenAI-compatible mode is not supported.

Environment variables:
    KIMI_API_KEY:  Required. Your Kimi API key (sk-kimi-...).
    KIMI_BASE_URL: Optional. Override the default base URL.
"""

import os
import logging
from typing import Optional, Any, Callable, Dict, Literal, Union

from adalflow.core.types import ModelType
from openai_client import OpenAIClient, get_first_message_content, parse_stream_response

log = logging.getLogger(__name__)

# Default Kimi Coding API base URL
DEFAULT_KIMI_BASE_URL = "https://api.kimi.com/coding/v1"


class KimiCodingClient(OpenAIClient):
    """Kimi Code Plan client using OpenAI SDK compatibility.

    Kimi's Coding API (K2.5) is compatible with the OpenAI Chat Completions
    format. This client reuses the OpenAIClient implementation with a
    different base URL and API key environment variable.

    Supported models:
        - k2p5: Kimi K2.5 (262K context, 32K output, reasoning, text+image)

    Usage:
        client = KimiCodingClient()
        # Ensure KIMI_API_KEY is set in the environment
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "KIMI_BASE_URL",
        env_api_key_name: str = "KIMI_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url or os.getenv(env_base_url_name, DEFAULT_KIMI_BASE_URL),
            env_base_url_name=env_base_url_name,
            env_api_key_name=env_api_key_name,
        )
