# src/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


class LongContextGenerator:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=False,                 # for 3B, keep False unless needed
        try_flash_attention_2=False,
        attn_implementation="eager",        # IMPORTANT: lowers peak memory vs sdpa on some GPUs
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

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(
            device_map="auto",
            dtype=torch.float16,      # ✅ FIXED (was dtype=...)
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        # Try flash_attention_2 if requested, else eager (safer on memory)
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    attn_implementation=attn_implementation,
                    **model_kwargs,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                **model_kwargs,
            )

        self.model.eval()

    def generate_answer(self, question, context_chunks, max_new_tokens=80, max_input_tokens=1024):
        context = "\n\n".join(context_chunks)

        prompt = (
            "You are a spoiler-free assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say: \"I don't know based on the given text.\".\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(self.model.device)
        attn = inputs["attention_mask"].to(self.model.device)

        # ✅ HARD CAP to avoid KV-cache OOM
        if input_ids.shape[1] > max_input_tokens:
            input_ids = input_ids[:, -max_input_tokens:]
            attn = attn[:, -max_input_tokens:]

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=6,
            )

        # ✅ Decode only new tokens (prevents prompt echo)
        new_tokens = outputs[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()