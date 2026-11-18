"""Local Qwen runtime has no hosted AI authentication."""
from .openai import create_subscription_llm_from_config

__all__ = ["create_subscription_llm_from_config"]
