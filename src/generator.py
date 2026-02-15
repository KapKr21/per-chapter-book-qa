import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LongContextGenerator:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct-1M"):
        #Quantization is necessary for 1M context windows on typical GPUs
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        print(f"Loading {model_id} with 1M context support...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            #Enabling Flash Attention 2 for efficient long-sequence handling
            attn_implementation="flash_attention_2" 
        )

    def generate_answer(self, question, chapters_up_to_k):
        """
        Feeds the entire book context (up to the spoiler boundary) into the LLM.
        """
        context = "\n\n".join(chapters_up_to_k)
        
        prompt = f"""<|im_start|>system
You are a spoiler-free assistant. Answer the question based ONLY on the provided text.
<|im_end|>
<|im_start|>user
BOOK CONTEXT (Chapters 1 to {len(chapters_up_to_k)}):
{context}

QUESTION: {question}
<|im_end|>
<|im_start|>assistant"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        #1M token check
        if inputs.input_ids.shape[1] > 1000000:
            print("Warning: Input exceeds 1M tokens. Truncating.")
            
        outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)