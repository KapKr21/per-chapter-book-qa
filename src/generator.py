# src/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

class LongContextGenerator:
    """
    Pascal GPU generator.
    - Tries FlashAttention2 if available
    - Falls back automatically if not
    - Uses 4-bit quant by default (common for 7B+ on 1 GPU)
    """

    def __init__(
        self,
        model_id="Qwen/Qwen2.5-7B-Instruct-1M",
        load_in_4bit=True,
        try_flash_attention_2=False,
    ):
        self.model_id = model_id

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        print(f"Loading generator: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Some causal LMs need pad token defined for batching
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(
            device_map="auto",
            dtype=torch.float16,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        # Try flash_attention_2; if it fails, fall back.
        if try_flash_attention_2:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    attn_implementation="flash_attention_2",
                    **model_kwargs,
                )
                print("✅ Using FlashAttention2.")
            except Exception as e:
                print(f"⚠️ FlashAttention2 unavailable, falling back. Reason: {type(e).__name__}: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        self.model.eval()

    def generate_answer(self, question, context_chunks, max_new_tokens=200):
        """
        context_chunks: list[str] of *safe* text pieces (retrieved)
        """
        context = "\n\n".join(context_chunks)

        prompt = (
            "You are a spoiler-free assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say: \"I don't know based on the given text.\".\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            # Keep this conservative unless you *know* you can handle more
            max_length=8192,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Best-effort strip
        if "ANSWER:" in text:
            text = text.split("ANSWER:")[-1].strip()
        return text
