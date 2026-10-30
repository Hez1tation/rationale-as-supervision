"""Generator wrappers for Stage 4 refinement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


DEFAULT_BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_DPO_ADAPTER = Path(
    "/project2/ruishanl_1185/rationale_as_supervision/DPO/model/checkpoints/"
    "dpo_overall_only_aggregate_4gpu_100e"
)
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Generator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class MockGenerator:
    """Deterministic dry-run generator."""

    def generate(self, prompt: str) -> str:
        if "Previous response:" in prompt:
            return (
                "I hear how difficult this feels, and it makes sense to want support. "
                "A safer next step is to talk with a licensed mental health professional "
                "or trusted support person, while using grounding, journaling, or a brief "
                "breathing exercise to get through the immediate moment. If there is any "
                "risk of harm, contact local emergency services or a crisis line right away."
            )
        return (
            "I am sorry you are dealing with this. What you are describing sounds painful, "
            "and you deserve support. It may help to talk with a mental health professional "
            "who can understand your situation and help you build coping strategies."
        )


class OpenAIGenerator:
    """OpenAI Responses API generator, used for GPT-4o refinement."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        api_key_env: str = "OPENAI_API_KEY",
        env_file: str | Path | None = DEFAULT_ENV_FILE,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        from openai import OpenAI

        load_env_file(env_file)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key. Set {api_key_env} before running, "
                f"for example: export {api_key_env}=..., or add it to {env_file}."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()
        return _extract_response_text(response).strip()


class HFGenerator:
    """HuggingFace causal-LM generator, optionally with a PEFT adapter."""

    def __init__(
        self,
        model_name: str = DEFAULT_BASE_MODEL,
        adapter_dir: str | Path | None = None,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))

        if not load_in_4bit:
            model = model.to(self.device)
        model.eval()
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def _encode(self, prompt: str):
        """Wrap the prompt as a chat turn so an *instruct* model answers in
        assistant mode instead of free-continuing the text as a document.

        Without the chat template, Llama-3.1-Instruct treats the prompt as a
        document to continue and tends to emit document-style scaffolding
        ('Response:', 'Here is the response:', trailing self-commentary). Routing
        it through apply_chat_template starts a clean assistant turn and removes
        most of that. Falls back to raw encoding if the tokenizer has no template.
        """
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return self.tokenizer(prompt, return_tensors="pt")

    def generate(self, prompt: str) -> str:
        inputs = self._encode(prompt).to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_generator(
    mode: str,
    model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: str | Path | None = None,
    device: str | None = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_env_file: str | Path | None = DEFAULT_ENV_FILE,
) -> Generator:
    if mode == "mock":
        return MockGenerator()
    if mode == "openai":
        return OpenAIGenerator(
            model_name=model_name,
            api_key_env=openai_api_key_env,
            env_file=openai_env_file,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    if mode == "hf":
        return HFGenerator(
            model_name=model_name,
            adapter_dir=adapter_dir,
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    raise ValueError(f"Unsupported generator mode: {mode}")


def _extract_response_text(response) -> str:
    """Best-effort text extraction for Responses API SDK objects."""
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        content_list = getattr(item, "content", None)
        if content_list is None and isinstance(item, dict):
            content_list = item.get("content")
        for content in content_list or []:
            text = getattr(content, "text", None)
            if text is None and isinstance(content, dict):
                text = content.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def load_env_file(path: str | Path | None) -> None:
    """Load KEY=VALUE lines from a .env file without overriding existing env vars."""
    if path is None:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
