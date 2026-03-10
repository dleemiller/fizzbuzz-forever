"""Test the trained fizzbuzz agent model."""
import os
import sys

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import yaml
from num2words import num2words

from fizzbuzz_forever.agent import FizzBuzzAgent
from fizzbuzz_forever.dataset import fizzbuzz


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_path = f"{cfg['training']['output_dir']}/final_model"
    print(f"Loading model from {model_path}")
    agent = FizzBuzzAgent.load(model_path)

    test_cases = [
        (1, "neither"), (3, "div by 3"), (5, "div by 5"), (15, "div by both"),
        (7, "neither"), (33, "div by 3"), (100, "div by 5"), (150, "div by both"),
        (999, "div by 3"), (10000, "div by 5"), (99990, "div by both"), (99991, "neither"),
    ]

    print("\n=== Digit format tests ===")
    correct = 0
    for number, desc in test_cases:
        expected = f"{number} {fizzbuzz(number)}"
        answer = agent(number)
        match = answer is not None and answer.strip() == expected
        correct += match
        status = "OK" if match else "FAIL"
        print(f"  [{status}] {number:>6} ({desc:>12}): got '{answer}', expected '{expected}'")
    print(f"\nDigit accuracy: {correct}/{len(test_cases)}")

    print("\n=== Verbalized format tests ===")
    verb_correct = 0
    for number, desc in test_cases:
        expected = f"{number} {fizzbuzz(number)}"
        # Verbalized input: override the agent call with num2words
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": num2words(number)},
        ]
        # Run agent loop manually with verbalized input
        import json, torch
        answer = None
        for _ in range(3):
            ids = agent.tokenizer.apply_chat_template(
                messages, tools=agent.tools, tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
            )
            if not isinstance(ids, torch.Tensor):
                ids = ids["input_ids"]
            ids = ids.to(agent.model.device)
            attention_mask = torch.ones_like(ids)
            with torch.no_grad():
                out = agent.model.generate(ids, attention_mask=attention_mask, max_new_tokens=200, use_cache=True)
            resp = agent.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
            if "<tool_call>" in resp:
                tc = json.loads(resp[resp.index("<tool_call>") + 11:resp.index("</tool_call>")].strip())
                result = getattr(agent.env, tc["name"])(**tc["arguments"])
                messages.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                ]})
                messages.append({"role": "tool", "name": tc["name"], "content": str(result)})
            else:
                answer = resp.split("<|im_end|>")[0].strip()
                break

        match = answer is not None and answer.strip() == expected
        verb_correct += match
        status = "OK" if match else "FAIL"
        word = num2words(number)
        print(f"  [{status}] {word:>30} ({desc:>12}): got '{answer}', expected '{expected}'")

    print(f"\nVerbalized accuracy: {verb_correct}/{len(test_cases)}")
    print(f"\nTotal accuracy: {correct + verb_correct}/{len(test_cases) * 2}")


if __name__ == "__main__":
    main()
