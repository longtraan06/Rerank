# vlmr1.py
import re
import torch
import time
import logging
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Configure library-level logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

class VlmRoomCompletion:
    """
    A simplified class to handle room completion tasks using a Vision Language Model (VLM).
    Specifically designed to compare two objects and determine the best fit.
    """

    def __init__(self, model_path: str = "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps", device: str | None = None):
        logger.info(f"Initializing VlmRoomCompletion with model: {model_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2"
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path)
            logger.info("Model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model or processor: {e}", exc_info=True)
            raise

    def compare_two_objects(self,
                            room: Image.Image,
                            query: str,
                            obj1: Image.Image,
                            obj2: Image.Image,
                            id1: str,
                            id2: str) -> tuple[str, Image.Image, str, bool]:
        """
        Core method: sends both object images and the room+query to the VLM and parses the result.
        Returns:
            - winner_id: str
            - winner_image: PIL.Image
            - reasoning: str
            - success: bool
        """
        start_time = time.time()
        logger.info(f"Comparing '{id1}' vs '{id2}' for query '{query}'")

        images = [room.convert("RGB"), obj1.convert("RGB"), obj2.convert("RGB")]
        content = [
            {"type":"text","text":"Room scene:"},
            {"type":"image"},
            {"type":"text","text":f"Query: {query}"},
            {"type":"text","text":f"Object 1 ({id1}):"},
            {"type":"image"},
            {"type":"text","text":f"Object 2 ({id2}):"},
            {"type":"image"},
            {"type":"text","text":(
                "Task: Choose the single object that fits best, then respond with tags:\n"
                "<think>...</think> and <answer>1 or 2</answer>."
            )}
        ]

        messages = [{"role":"user","content":content}]
        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_text],
                images=images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            ).to(self.device)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            torch.cuda.empty_cache()

            # decode and trim prompt
            prompt_len = inputs['input_ids'].shape[1]
            out_ids = gen_ids[:,prompt_len:]
            text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            think = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            reasoning = think.group(1).strip() if think else ''
            ans = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            choice = ans.group(1).strip() if ans else None

            if choice == '1':
                return id1, obj1, reasoning, True
            elif choice == '2':
                return id2, obj2, reasoning, True
            else:
                logger.error(f"Invalid choice '{choice}'")
                return '', None, reasoning + f" [Invalid answer: {choice}]", False

        except Exception as e:
            logger.error(f"Error during compare: {e}", exc_info=True)
            return '', None, str(e), False
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Comparison took {elapsed:.2f}s")

    def select_best_object(self,
                           room: Image.Image,
                           query: str,
                           obj1_data: tuple[str, Image.Image],
                           obj2_data: tuple[str, Image.Image]) -> tuple[tuple[str, Image.Image] | None, str, bool]:
        """
        Wrapper that takes id/image tuples, calls compare_two_objects, and returns winner tuple.
        """
        id1, img1 = obj1_data
        id2, img2 = obj2_data
        winner_id, winner_img, reasoning, success = self.compare_two_objects(
            room, query, img1, img2, id1, id2
        )
        if success:
            return (winner_id, winner_img), reasoning, True
        else:
            return None, reasoning, False


def main(objects_path: list[str],
         object_ids: list[str],
         query: str,
         query_id: str,
         query_image_path: str) -> dict:
    """
    Load model internally, process selection, return summary.
    """

    # Load model
    model = VlmRoomCompletion()

    # Load images
    room_img = Image.open(query_image_path).convert("RGB")
    obj1_img = Image.open(objects_path[0]).convert("RGB")
    obj2_img = Image.open(objects_path[1]).convert("RGB")

    id1, id2 = object_ids

    (winner_data, reasoning, success) = model.select_best_object(
        room_img, query, (id1, obj1_img), (id2, obj2_img)
    )

    best_id = winner_data[0] if success and winner_data else None
    ranks = {id1: None, id2: None}
    if best_id:
        ranks[best_id] = 1
        other = id2 if best_id == id1 else id1
        ranks[other] = 2

    return {
        "query_id": query_id,
        "best_object_id": best_id,
        "ranks": ranks,
        "reasoning": reasoning
    }
if __name__ == "__main__":
    # test nhanh khi run file ecec.py trực tiếp
    pass
