import re
import torch
import time
import copy
import math
import os
import sys
import logging
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import pandas as pd
import random
import numpy as np

def set_seed(seed=42):
    """Set a fixed random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(query_id, room_pil, query, candidate1_pil, candidate2_pil, cand1_id, cand2_id, processor, model, device):
        set_seed()
        images = [
            #Image.open(room_pil).convert("RGB"),
                   Image.open(candidate1_pil).convert("RGB"), Image.open(candidate2_pil).convert("RGB")]

        content = [
            # {"type": "text", "text": "Reference Room Scene (with a blue masked object): {}"},
            # {"type": "image"}, 
            {"type": "text", "text": f"\nQuery/You are given two images of furniture items. Your task is to carefully examine both images and determine which one matches the following description most accurately:: {query}\n"},
            {"type": "text", "text": f"Object 1 ({cand1_id}):"},
            {"type": "image"}, 
            {"type": "text", "text": f"Object 2 ({cand2_id}):"},
            {"type": "image"},
            {"type": "text", "text": (
                "\nTask: Compare object 1 and object 2 with the description.\n"
                """You must strictly focus on:
- **Color**: Compare the exact colors described with those visible in the images.
- **Material and texture**: Identify the material (e.g., wood, leather, fabric) and texture (e.g., smooth, rough) if mentioned.
- **Type and style**: Recognize the specific type of furniture (e.g., chair, sofa, table) and its style (e.g., modern, vintage, minimalist).
- **Functionality**: Consider the intended use of the item (e.g., relaxing, working, dining).\n"""
                "Respond in the following format ONLY:\n"
                "<think> [Explain step-by-step your reasoning for choosing one object that aligns better with the most critical elements of the description] </think>\n"
                "<answer> [Write ONLY '1' if object 1 is the better, or ONLY '2' if object 2 is the better] </answer>"
            )}
        ]

        messages = [{"role": "user", "content": content}]

        try:
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[chat_text],
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left", # Usually recommended for generation
                add_special_tokens=False,
            ).to(device)

            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                     eos_token_id=[processor.tokenizer.eos_token_id,
                                  processor.tokenizer.pad_token_id,
                                  processor.tokenizer.eos_token_id]
                )

            torch.cuda.empty_cache()

            prompt_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            # Be careful with skip_special_tokens=True if < > symbols are eos tokens
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

            # Extract thinking and answer
            think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL | re.IGNORECASE)
            thinking_process = think_match.group(1).strip() if think_match else "No <think> tag found."
            answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL | re.IGNORECASE)
            preferred_candidate_index = None # Default to None (failure)

            if answer_match:
                answer_content = answer_match.group(1).strip()
                print(answer_content)
                if answer_content == '1':
                    object_ids = []
                    object_ids.append(cand1_id)
                    object_ids.append(cand2_id)
                    
                    rerank_results ={ 
                    "query_id": query_id,
                    "objects": object_ids
                    }
                    return rerank_results
                elif answer_content == '2':
                    object_ids = []
                    object_ids.append(cand2_id)
                    object_ids.append(cand1_id)

                    rerank_results ={ 
                    "query_id": query_id,
                    "objects": object_ids
                    }
                    return rerank_results
                else:
                    error_msg = f"Invalid answer '{answer_content}'. Expected '1' or '2'."

                    thinking_process += f"\n[Error: {error_msg}]"

        except Exception as e:
            error_msg = f"Exception during VLM comparison ({cand1_id} vs {cand2_id}): {e}"
            return None, error_msg


def load_model():
    model_path = "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Consider adding trust_remote_code=True if needed by the specific model revision
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, # Use float32 on CPU
        attn_implementation="flash_attention_2"
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor, device