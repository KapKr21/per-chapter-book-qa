import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LongContextGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_id = model_id
        print(f"Loading generator: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

        # Some tokenizers/models don't set pad_token_id; make it safe for generate()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_answer(
        self,
        question: str,
        context_chunks: list[str],
        max_new_tokens: int = 64,
        max_input_tokens: int = 2048,
    ) -> str:
        context = "\n\n".join([c for c in context_chunks if c and str(c).strip()])

        system_msg = (
            "You are a spoiler-free assistant.\n"
            "Answer ONLY using the provided context.\n"
            f'If the answer is not in the context, say exactly: "{IDK_FALLBACK}"\n'
            "Keep the answer short."
        )
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Qwen instruct models prefer chat template
        enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # apply_chat_template may return a Tensor OR a BatchEncoding/dict
        if isinstance(enc, torch.Tensor):
            input_ids = enc
            attention_mask = None
        else:
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)

        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # Truncate from the LEFT to keep the most recent content (context/question)
        if input_ids.shape[1] > max_input_tokens:
            input_ids = input_ids[:, -max_input_tokens:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -max_input_tokens:]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=6,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Decode only NEW tokens (prevents prompt-echo looking like hallucination)
        new_tokens = out[0, input_ids.shape[1]:]
        ans = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Hard clamp: if model includes the fallback plus other text, keep only fallback
        if IDK_FALLBACK in ans:
            return IDK_FALLBACK

        return ans

# Keep this in generator.py so generator can reference it
IDK_FALLBACK = "I don't know based on the given text."