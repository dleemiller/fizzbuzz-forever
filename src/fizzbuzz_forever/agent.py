import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fizzbuzz_forever.env import FizzBuzzEnv
from fizzbuzz_forever.dataset import get_tool_definitions
from fizzbuzz_forever.train import patch_no_thinking

CONFIG_FILENAME = "fizzbuzz_config.json"


class FizzBuzzAgent:
    def __init__(self, model, tokenizer, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.tools = get_tool_definitions()
        self.env = FizzBuzzEnv()

    @classmethod
    def load(cls, path: str):
        """Load a trained fizzbuzz agent from a local path or HF model ID."""
        tokenizer = AutoTokenizer.from_pretrained(path)
        patch_no_thinking(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()

        config_path = Path(path) / CONFIG_FILENAME
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            system_prompt = cfg["system_prompt"]
        else:
            raise FileNotFoundError(
                f"No {CONFIG_FILENAME} found at {path}. "
                "Was this model saved with fizzbuzz-train?"
            )

        return cls(model, tokenizer, system_prompt)

    def save(self, path: str):
        """Save the agent (model + tokenizer + config) to a directory."""
        self.model.config.tie_word_embeddings = False
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(Path(path) / CONFIG_FILENAME, "w") as f:
            json.dump({"system_prompt": self.system_prompt}, f)

    def __call__(self, n: int, max_iterations: int = 3) -> str:
        """Run fizzbuzz on a number. Returns e.g. '15 FizzBuzz'."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(n)},
        ]
        for _ in range(max_iterations):
            ids = self.tokenizer.apply_chat_template(
                messages, tools=self.tools, tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
            )
            if not isinstance(ids, torch.Tensor):
                ids = ids["input_ids"]
            ids = ids.to(self.model.device)
            attention_mask = torch.ones_like(ids)
            with torch.no_grad():
                out = self.model.generate(
                    ids, attention_mask=attention_mask,
                    max_new_tokens=200, use_cache=True,
                )
            resp = self.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
            if "<tool_call>" in resp:
                tc_start = resp.index("<tool_call>") + len("<tool_call>")
                tc_end = resp.index("</tool_call>")
                tc = json.loads(resp[tc_start:tc_end].strip())
                result = getattr(self.env, tc["name"])(**tc["arguments"])
                messages.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                ]})
                messages.append({"role": "tool", "name": tc["name"], "content": str(result)})
            else:
                return resp.split("<|im_end|>")[0].strip()
        return None
