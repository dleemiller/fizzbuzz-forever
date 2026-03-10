import random

from datasets import Dataset
from num2words import num2words


def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)


def get_tool_definitions() -> list[dict]:
    """Get tool definitions for the modulo tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": "modulo",
                "description": "Compute the remainder of dividing two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "The dividend."},
                        "b": {"type": "integer", "description": "The divisor."},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]


def _build_tool_calling_messages(target: int, system_prompt: str, verbalize: bool = False) -> list[dict]:
    """Build a tool-calling conversation showing correct fizzbuzz reasoning."""
    user_content = num2words(target) if verbalize else str(target)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Step 1: check modulo 3
    mod3 = target % 3
    messages.append({
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {
            "name": "modulo", "arguments": {"a": target, "b": 3}
        }}],
    })
    messages.append({"role": "tool", "name": "modulo", "content": str(mod3)})

    # Step 2: check modulo 5
    mod5 = target % 5
    messages.append({
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {
            "name": "modulo", "arguments": {"a": target, "b": 5}
        }}],
    })
    messages.append({"role": "tool", "name": "modulo", "content": str(mod5)})

    # Step 3: final answer as text
    result = fizzbuzz(target)
    answer = f"{target} {result}"
    messages.append({"role": "assistant", "content": answer})

    return messages


def create_tool_sft_dataset(
    num_samples: int,
    max_number: int,
    system_prompt: str,
    verbalize_prob: float = 0.0,
) -> Dataset:
    """Create SFT dataset with tool-calling examples.

    Uses prompt-completion format with tools column.
    Supplements with every number 1-200 for coverage.
    """
    tools = get_tool_definitions()
    records = []
    # Supplement with every number 1-200
    for target in range(1, 201):
        for verb in ([False, True] if verbalize_prob > 0 else [False]):
            messages = _build_tool_calling_messages(target, system_prompt, verbalize=verb)
            records.append({
                "prompt": messages[:2],
                "completion": messages[2:],
                "tools": tools,
            })
    # Fill the rest with random large numbers
    for _ in range(num_samples - len(records)):
        target = random.randint(1, max_number)
        verbalize = verbalize_prob > 0 and random.random() < verbalize_prob
        messages = _build_tool_calling_messages(target, system_prompt, verbalize=verbalize)
        records.append({
            "prompt": messages[:2],
            "completion": messages[2:],
            "tools": tools,
        })
    random.shuffle(records)
    return Dataset.from_list(records)
