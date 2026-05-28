"""LLM generation module supporting OpenAI and Anthropic."""

import logging

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
) -> str:
    """Generate an answer using the configured LLM."""
    if provider == "openai":
        return _generate_openai(system_prompt, user_prompt, model)
    elif provider == "anthropic":
        return _generate_anthropic(system_prompt, user_prompt, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _generate_openai(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    return response.choices[0].message.content or ""


def _generate_anthropic(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate using Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text
