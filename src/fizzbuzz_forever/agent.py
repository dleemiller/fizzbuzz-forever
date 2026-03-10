import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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
        self._prefix_cache = None
        self._prefix_len = 0

    def _ensure_prefix_cache(self):
        """Compute and cache KV for the system prompt + tool definitions prefix."""
        if self._prefix_cache is not None:
            return
        prefix_ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}],
            tools=self.tools, tokenize=True,
            add_generation_prompt=False, return_tensors="pt",
        )
        if not isinstance(prefix_ids, torch.Tensor):
            prefix_ids = prefix_ids["input_ids"]
        prefix_ids = prefix_ids.to(self.model.device)
        self._prefix_cache = DynamicCache(config=self.model.config)
        with torch.no_grad():
            self.model(
                prefix_ids, past_key_values=self._prefix_cache,
                cache_position=torch.arange(prefix_ids.shape[1], device=prefix_ids.device),
                use_cache=True,
            )
        self._prefix_len = prefix_ids.shape[1]

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
        self._ensure_prefix_cache()
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
            # Reuse prefix KV cache — only new tokens get processed
            past_kv = copy.deepcopy(self._prefix_cache)
            new_ids = ids[:, self._prefix_len:]
            cache_pos = torch.arange(
                self._prefix_len, ids.shape[1],
                dtype=torch.long, device=ids.device,
            )
            attn_mask = torch.ones(1, ids.shape[1], dtype=torch.long, device=ids.device)
            with torch.no_grad():
                out = self.model.generate(
                    new_ids,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    cache_position=cache_pos,
                    max_new_tokens=200, use_cache=True,
                )
            resp = self.tokenizer.decode(out[0][new_ids.shape[-1]:], skip_special_tokens=False)
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
