import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, set_seed
set_seed(42) 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model():
    path = 'omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    return model, tokenizer, generation_config


def main(model, tokenizer, generation_config, images_list, object_ids, query, query_id):
    # 1) Load và preprocess ảnh room, cand1, cand2
    pv_room = load_image(images_list[0], max_num=1).to(torch.bfloat16).cuda()
    pv_c1   = load_image(images_list[1], max_num=1).to(torch.bfloat16).cuda()
    pv_c2   = load_image(images_list[2], max_num=1).to(torch.bfloat16).cuda()
    pixel_values = torch.cat([pv_room, pv_c1, pv_c2], dim=0)

    # 2) Xây prompt từ content list
    content = [
        {"type": "text",  "text": "Reference Room Scene (with a missing object area):"},
        {"type": "image"},   # room
        {"type": "text",  "text": f"\nQuery/Description: {query}\n"},
        {"type": "text",  "text": f"Candidate Object 1 ({object_ids[0]}):"},
        {"type": "image"},   # cand1
        {"type": "text",  "text": f"Candidate Object 2 ({object_ids[1]}):"},
        {"type": "image"},   # cand2
        {"type": "text",  "text": (
            "\nTask: Compare Candidate 1 and Candidate 2.\n"
                "Which SINGLE candidate object fits better into the missing area of the Reference Room Scene, "
                "considering both the visual context/style of the room AND the details in the Query/Description? "
                "Analyze how well each candidate matches the requirements.\n"
                "Respond in the following format ONLY:\n"
                "<think> [Explain step-by-step your reasoning for choosing one candidate over the other, referencing the room scene and the query.] </think>\n"
                "<answer> [Write ONLY '1' if Candidate 1 is the better fit, or ONLY '2' if Candidate 2 is the better fit.] </answer>"
        )}
    ]
    prompt_lines = []
    for item in content:
        if item["type"] == "text":
            prompt_lines.append(item["text"])
        else:  # image
            prompt_lines.append("<image>")
    prompt = "\n".join(prompt_lines)

    # 3) Gọi model.chat với prompt và pixel_values đúng thứ tự
    promptt = tokenizer.apply_chat_template(content, add_generation_prompt=True, return_tensors="pt").to(model.device)
    output = model.generate(promptt, max_new_tokens=1024, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)




    # 4) Parse kết quả
    import re
    m = re.search(r'<answer>\s*([12])\s*</answer>', response)
    if not m:
        print(f"[ERROR] No valid <answer> in response: {response}")
        return None
    choice = int(m.group(1))
    winner_id = object_ids[choice]
    print(f"[RESULT] query_id={query_id} → WINNER: {winner_id}")
    print(f"[REASON]\n{response}")
    return {"query_id": query_id, "winner": winner_id}



        # response, history = model.chat(
    #     tokenizer,
    #     pixel_values,
    #     prompt,
    #     generation_config,
    #     history=None,
    #     return_history=True
    # )
# from PIL import Image
# import torch

# # Build content list
# content = [
#     {"type": "text", "text": "Reference Room Scene (with a missing object area):"},
#     {"type": "image", "image": images_list[0]},
#     {"type": "text", "text": f"\nQuery/Description: {query}\n"},
#     {"type": "text", "text": f"Candidate Object 1 ({object_ids[0]}):"},
#     {"type": "image", "image": images_list[1]},
#     {"type": "text", "text": f"Candidate Object 2 ({object_ids[1]}):"},
#     {"type": "image", "image": images_list[2]},
#     {"type": "text", "text": (
#         "\nTask: Compare Candidate 1 and Candidate 2. "
#         "Which SINGLE candidate fits best? "
#         "Respond ONLY with '<think>...'</think>' then '<answer>1 or 2</answer>'."
#     )}
# ]

# # Load images as PIL
# def load_pil(image_path):
#     return Image.open(image_path).convert('RGB')

# pil_images = [load_pil(img_path) for img_path in images_list]

# # Build prompt with images
# messages = []
# for item in content:
#     if item["type"] == "text":
#         messages.append({"type": "text", "text": item["text"]})
#     elif item["type"] == "image":
#         img_idx = len([m for m in messages if m['type'] == 'image'])
#         messages.append({"type": "image", "image": pil_images[img_idx]})

# # Apply chat template
# prompt_inputs = tokenizer.apply_chat_template(
#     messages, 
#     add_generation_prompt=True, 
#     return_tensors="pt"
# ).to(model.device)

# # Generate
# outputs = model.generate(
#     inputs=prompt_inputs,
#     max_new_tokens=1024,
#     do_sample=True
# )

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Parse response
# import re
# m = re.search(r'<answer>\s*([12])\s*</answer>', response)
# if not m:
#     print(f"[ERROR] No valid <answer> in response: {response}")
#     return None
# choice = int(m.group(1))
# winner_id = object_ids[choice - 1]
# print(f"[RESULT] query_id={query_id} → WINNER: {winner_id}")
# print(f"[REASON]\n{response}")
# return {"query_id": query_id, "winner": winner_id}
# import re
# from PIL import Image
# import torch

# def load_pil(image_path):
#     return Image.open(image_path).convert('RGB')

# def main(model, tokenizer, generation_config, images_list, object_ids, query, query_id):
#     # 1) Load ảnh PIL
#     pil_images = [load_pil(p) for p in images_list]

#     # 2) Xây content
#     content = [
#         {"type": "text", "text": "Reference Room Scene (with a missing object area):"},
#         {"type": "image", "image": pil_images[0]},
#         {"type": "text", "text": f"\nQuery/Description: {query}\n"},
#         {"type": "text", "text": f"Candidate Object 1 ({object_ids[0]}):"},
#         {"type": "image", "image": pil_images[1]},
#         {"type": "text", "text": f"Candidate Object 2 ({object_ids[1]}):"},
#         {"type": "image", "image": pil_images[2]},
#         {"type": "text", "text": (
#             "\nTask: Compare Candidate 1 and Candidate 2.\n"
#                 "Which SINGLE candidate object fits better into the missing area of the Reference Room Scene, "
#                 "considering both the visual context/style of the room AND the details in the Query/Description? "
#                 "Analyze how well each candidate matches the requirements.\n"
#                 "Respond in the following format ONLY:\n"
#                 "<think> [Explain step-by-step your reasoning for choosing one candidate over the other, referencing the room scene and the query.] </think>\n"
#                 "<answer> [Write ONLY '1' if Candidate 1 is the better fit, or ONLY '2' if Candidate 2 is the better fit.] </answer>"
#         )}
#     ]
    
#     # 3) Build prompt
#     prompt_inputs = tokenizer.apply_chat_template(
#         content, 
#         add_generation_prompt=True,
#         return_tensors="pt"
#     ).to(model.device)

#     # 4) Generate
#     outputs = model.generate(
#         inputs=prompt_inputs,
#         max_new_tokens=generation_config.get('max_new_tokens', 512),
#         do_sample=generation_config.get('do_sample', True),
#     )

#     # 5) Decode response
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"[RESPONSE]\n{response}")

#     # 6) Parse answer
#     m = re.search(r'<answer>\s*([12])\s*</answer>', response)
#     if not m:
#         raise ValueError(f"[ERROR] No valid <answer> in response: {response}")

#     choice = int(m.group(1))
#     winner_id = object_ids[choice - 1]
#     print(f"[RESULT] query_id={query_id} → WINNER: {winner_id}")
#     print(f"[REASON]\n{response}")

#     return {"query_id": query_id, "winner": winner_id}
# import torch
# from PIL import Image
# import torchvision.transforms as T
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# def load_model(model_path, device_map="auto"):
#     """Load Qwen2.5-VL model with 4-bit quantization."""
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         device_map=device_map,
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#         )
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True
#     )
#     return model, tokenizer


# def load_image(image_path):
#     """Load image and convert to tensor."""
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = T.ToTensor()(image)
#     return image_tensor.unsqueeze(0)

# def build_transform():
#     """Build transform to resize and normalize image."""
#     transform = T.Compose([
#         T.Resize((512, 512)),
#         T.ToTensor(),
#         T.Normalize(
#             mean=[0.5, 0.5, 0.5],
#             std=[0.5, 0.5, 0.5],
#         ),
#     ])
#     return transform

# def dynamic_preprocess(image_list):
#     """Apply preprocessing to list of images."""
#     transform = build_transform()
#     processed_images = [transform(img) for img in image_list]
#     pixel_values = torch.stack(processed_images, dim=0)
#     return pixel_values

# def load_pil(image_path):
#     """Load image and keep as PIL Image."""
#     return Image.open(image_path).convert('RGB')

# def main(model, tokenizer, generation_config, images_list, object_ids, query, query_id):
#     """Main function to perform inference."""

#     # Load PIL images (for tokenizer)
#     pil_images = [load_pil(p) for p in images_list]

#     # Build content for prompt
#     content = [
#         {"type": "text", "text": "Reference Room Scene (with a missing object area):"},
#         {"type": "image", "image": pil_images[0]},
#         {"type": "text", "text": f"\nQuery/Description: {query}\n"},
#         {"type": "text", "text": f"Candidate Object 1 ({object_ids[0]}):"},
#         {"type": "image", "image": pil_images[1]},
#         {"type": "text", "text": f"Candidate Object 2 ({object_ids[1]}):"},
#         {"type": "image", "image": pil_images[2]},
#         {"type": "text", "text": (
#             "\nTask: Compare Candidate 1 and Candidate 2. "
#             "Which SINGLE candidate fits best? "
#             "Respond ONLY with '<think>...'</think>' then '<answer>1 or 2</answer>'."
#         )}
#     ]

#     # Tokenize
#     prompt_inputs = tokenizer.apply_chat_template(
#         content, 
#         add_generation_prompt=True,
#         return_tensors="pt"
#     ).to(model.device)

#     # Generate output
#     outputs = model.generate(
#         inputs=prompt_inputs,
#         max_new_tokens=generation_config.get('max_new_tokens', 512),
#         do_sample=generation_config.get('do_sample', True),
#     )

#     # Decode response
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"[RESPONSE]\n{response}")

#     # Parse choice
#     m = re.search(r'<answer>\s*([12])\s*</answer>', response)
#     if not m:
#         raise ValueError(f"[ERROR] No valid <answer> in response: {response}")

#     choice = int(m.group(1))
#     winner_id = object_ids[choice - 1]
#     print(f"[RESULT] query_id={query_id} → WINNER: {winner_id}")

#     return {"query_id": query_id, "winner": winner_id}
# import numpy as np
# import torch
# import torchvision.transforms as T
# from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
# import re

# set_seed(42)

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# def build_transform(input_size):
#     MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#     transform = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=MEAN, std=STD),
#     ])
#     return transform

# def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
#     best_ratio_diff = float('inf')
#     best_ratio = (1, 1)
#     area = width * height
#     for ratio in target_ratios:
#         target_aspect_ratio = ratio[0] / ratio[1]
#         ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#         if ratio_diff < best_ratio_diff:
#             best_ratio_diff = ratio_diff
#             best_ratio = ratio
#         elif ratio_diff == best_ratio_diff:
#             if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                 best_ratio = ratio
#     return best_ratio

# def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
#     orig_width, orig_height = image.size
#     aspect_ratio = orig_width / orig_height

#     target_ratios = set(
#         (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
#         if i * j <= max_num and i * j >= min_num
#     )
#     target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

#     target_aspect_ratio = find_closest_aspect_ratio(
#         aspect_ratio, target_ratios, orig_width, orig_height, image_size
#     )

#     target_width = image_size * target_aspect_ratio[0]
#     target_height = image_size * target_aspect_ratio[1]
#     blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

#     resized_img = image.resize((target_width, target_height))
#     processed_images = []
#     for i in range(blocks):
#         box = (
#             (i % (target_width // image_size)) * image_size,
#             (i // (target_width // image_size)) * image_size,
#             ((i % (target_width // image_size)) + 1) * image_size,
#             ((i // (target_width // image_size)) + 1) * image_size,
#         )
#         split_img = resized_img.crop(box)
#         processed_images.append(split_img)
#     assert len(processed_images) == blocks

#     if use_thumbnail and len(processed_images) != 1:
#         thumbnail_img = image.resize((image_size, image_size))
#         processed_images.append(thumbnail_img)

#     return processed_images

# def load_image(image_file, input_size=448, max_num=1):
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
#     pixel_values = [transform(img) for img in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

# def load_model():
#     path = 'omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps'
#     model = AutoModelForCausalLM.from_pretrained(
#         path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#         )
#     ).eval()
#     tokenizer = AutoTokenizer.from_pretrained(
#         path,
#         trust_remote_code=True,
#         use_fast=False,
#     )
#     generation_config = dict(max_new_tokens=1024, do_sample=True)
#     return model, tokenizer, generation_config

# def main(model, tokenizer, generation_config, images_list, object_ids, query, query_id):
#     pv_room = load_image(images_list[0], max_num=1).to(torch.bfloat16).cuda()
#     pv_c1   = load_image(images_list[1], max_num=1).to(torch.bfloat16).cuda()
#     pv_c2   = load_image(images_list[2], max_num=1).to(torch.bfloat16).cuda()

#     pixel_values = torch.cat([pv_room, pv_c1, pv_c2], dim=0)

#     content = [
#         {"type": "text",  "text": "Reference Room Scene (with a missing object area):"},
#         {"type": "image"},
#         {"type": "text",  "text": f"\nQuery/Description: {query}\n"},
#         {"type": "text",  "text": f"Candidate Object 1 ({object_ids[0]}):"},
#         {"type": "image"},
#         {"type": "text",  "text": f"Candidate Object 2 ({object_ids[1]}):"},
#         {"type": "image"},
#         {"type": "text",  "text": (
#             "\nTask: Compare Candidate 1 and Candidate 2.\n"
#                 "Which SINGLE candidate object fits better into the missing area of the Reference Room Scene, "
#                 "considering both the visual context/style of the room AND the details in the Query/Description? "
#                 "Analyze how well each candidate matches the requirements.\n"
#                 "Respond in the following format ONLY:\n"
#                 "<think> [Explain step-by-step your reasoning for choosing one candidate over the other, referencing the room scene and the query.] </think>\n"
#                 "<answer> [Write ONLY '1' if Candidate 1 is the better fit, or ONLY '2' if Candidate 2 is the better fit.] </answer>"
#         )}
#     ]

#     prompt_inputs = tokenizer.apply_chat_template(
#         content, add_generation_prompt=True, return_tensors="pt"
#     ).to(model.device)

#     outputs = model.generate(
#         inputs=prompt_inputs,
#         max_new_tokens=generation_config.get('max_new_tokens', 1024),
#         do_sample=generation_config.get('do_sample', True),
#     )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"[RESPONSE]\n{response}")

#     m = re.search(r'<answer>\s*([12])\s*</answer>', response)
#     if not m:
#         print(f"[ERROR] No valid <answer> in response: {response}")
#         return None
#     choice = int(m.group(1))
#     winner_id = object_ids[choice - 1]
#     print(f"[RESULT] query_id={query_id} → WINNER: {winner_id}")
#     return {"query_id": query_id, "winner": winner_id}
