"""
LLM Handler for the Qwen "Student" Judge Model using the Transformers library.

This module loads and interacts with the Qwen model directly from the
Hugging Face Hub. It uses the `transformers` pipeline for robust and
efficient text generation, and automatically handles device placement (GPU/CPU).
"""
import os
from dotenv import load_dotenv
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLMHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMHandler, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        model_name = "Qwen/Qwen3-0.6B"
        logger.info(f"Initializing Transformers pipeline for model: {model_name}")
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                token=hf_token,
                device_map="auto",
                torch_dtype=torch.bfloat16 # Recommended for performance
            )
            logger.success(f"Qwen model loaded on device: {self.pipe.device}")
        except Exception as e:
            logger.critical(f"Failed to load model from Hugging Face: {e}")
            raise

    def query(self, prompt: str, think_mode: bool = False) -> str:
        prompt_with_mode = f"{prompt}{' /think' if think_mode else ''}"
        messages = [{"role": "user", "content": prompt_with_mode}]
        prompt_formatted = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        params = {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.5}
        if think_mode:
            params.update({"temperature": 0.6, "top_p": 0.95})
            
        try:
            outputs = self.pipe(
                prompt_formatted,
                max_new_tokens=256,
                do_sample=True,
                **params
            )
            response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Error during LLM query: {e}")
            return '{"error": "LLM query failed"}' 