import json
import os
import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google import genai
from google.genai import types  # or import the config class

load_dotenv()


def save_jsonl(result, output_file):
    """Write result to file and update processed IDs set"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


# --- Async Clients ---
client_openai_async = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_deepseek_async = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

client_anthropic_async = anthropic.AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client_gemini_async = client_genai.aio


# --- Client Factory ---
def get_client_async(model: str = "gpt-4.1"):
    """Return the correct async client for a model family."""
    if model.startswith("deepseek"):
        return client_deepseek_async
    elif model.startswith("gpt-") or model.startswith("o"):
        return client_openai_async
    elif model.startswith("claude-"):
        return client_anthropic_async
    elif model.startswith("gemini-"):
        return client_gemini_async
    else:
        raise ValueError(
            f"Unsupported model: {model}. "
            "Use 'gpt-*', 'deepseek-*', 'claude-*', or 'gemini-*'."
        )


# --- Unified API ---
async def run_completion(
        model: str,
        messages: list[dict],
        json_mode: bool = False,
        max_tokens: int = 150,
        **kwargs,
) -> str | dict:
    # For GPT / DeepSeek using new Responses API
    if model.startswith(("gpt-", "o", "deepseek")):
        client = get_client_async(model)

        resp = await client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=max_tokens,
            **kwargs,
        )
        text_out = resp.output_text.strip()

        if json_mode:
            try:
                return json.loads(text_out)
            except json.JSONDecodeError as e:
                # Fallback: return raw text if invalid JSON
                raise ValueError(f"Response was not valid JSON: {text_out!r}") from e
        else:
            return text_out

    # For Anthropic
    elif model.startswith("claude-"):
        client = get_client_async(model)
        resp = await client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens or 1000,
            **kwargs,
        )
        text_out = resp.content[0].text.strip()

        if json_mode:
            try:
                return json.loads(text_out)
            except json.JSONDecodeError as e:
                raise ValueError(f"Response was not valid JSON: {text_out!r}") from e
        else:
            return text_out

    # Gemini
    elif model.startswith("gemini-"):
        client = get_client_async(model)
        text_input = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        resp = await client.models.generate_content(
            model=model,
            contents=text_input,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                    # include_thoughts=True
                ),
                max_output_tokens=max_tokens,
                **kwargs
            )
        )
        text_out = resp.text.strip()

        if json_mode:
            try:
                return json.loads(text_out)
            except json.JSONDecodeError as e:
                raise ValueError(f"Response was not valid JSON: {text_out!r}") from e
        else:
            return text_out

    else:
        raise ValueError(f"Unsupported model: {model}")
