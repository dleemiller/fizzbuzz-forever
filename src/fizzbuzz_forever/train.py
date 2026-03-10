import os
import sys

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import yaml
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import qwen3_schema

from fizzbuzz_forever.dataset import create_tool_sft_dataset


def patch_no_thinking(tokenizer):
    """Patch Qwen3 chat template to remove all thinking logic.

    The default Qwen3 template adds <think>...</think> to the last assistant
    message and to the generation prompt when enable_thinking=False. This
    creates an asymmetry between tool-calling turns (no think tags) and the
    final answer turn (has think tags), which confuses the model during
    agentic generation.

    This patched template never adds think tags, making all assistant turns
    consistent.
    """
    template = tokenizer.chat_template
    template = template.replace(
        """{%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}""",
        """{{- '<|im_start|>' + message.role + '\\n' + content }}"""
    )
    template = template.replace(
        """{%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}""",
        ""
    )
    tokenizer.chat_template = template
    tokenizer.response_schema = qwen3_schema
    return tokenizer


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    patch_no_thinking(tokenizer)

    ds_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    sft_dataset = create_tool_sft_dataset(
        num_samples=ds_cfg["num_samples"],
        max_number=ds_cfg["max_number"],
        system_prompt=ds_cfg["system_prompt"],
        verbalize_prob=ds_cfg.get("verbalize_prob", 0.0),
    )

    trainer = SFTTrainer(
        model=model_name,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            learning_rate=training_cfg["learning_rate"],
            max_steps=training_cfg["max_steps"],
            per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
            logging_steps=training_cfg["logging_steps"],
            output_dir=training_cfg["output_dir"],
            report_to=training_cfg["report_to"],
            bf16=True,
            model_init_kwargs={"torch_dtype": "bfloat16"},
        ),
    )

    trainer.train()

    from fizzbuzz_forever.agent import FizzBuzzAgent
    agent = FizzBuzzAgent(trainer.model, tokenizer, ds_cfg["system_prompt"])
    agent.save(f"{training_cfg['output_dir']}/final_model")


if __name__ == "__main__":
    main()
