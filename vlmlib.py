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

# Configure library-level logging
logger = logging.getLogger(__name__)
# Set default logging level if the user hasn't configured it
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class VlmRoomCompletion:
    """
    A class to handle room completion tasks using a Vision Language Model (VLM).

    Provides methods for:
    - Loading the VLM model and processor.
    - Comparing two candidate objects for fit in a room scene based on a query.
    - Running an elimination tournament to rank multiple candidates.
    - (Optional: The original single-prompt selection method is kept for reference/completeness)
    """

    def __init__(self, model_path: str = "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps", device: str | None = None):
        """
        Initializes the VlmRoomCompletion handler.

        Args:
            model_path (str): Path or Hugging Face identifier for the VLM model.
            device (str | None): The device to run the model on ('cuda', 'cpu').
                                 If None, attempts to use CUDA if available, otherwise CPU.
        """
        logger.info(f"Initializing VlmRoomCompletion with model: {model_path}")

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            # Consider adding trust_remote_code=True if needed by the specific model revision
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, # Use float32 on CPU
                attn_implementation="flash_attention_2"
            )
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path)
            logger.info("Model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model or processor from {model_path}: {e}", exc_info=True)
            raise

    def compare_two_candidates_for_fit(self, room_pil: Image.Image, query: str,
                                     candidate1_pil: Image.Image, candidate2_pil: Image.Image,
                                     cand1_id: str, cand2_id: str) -> tuple[int | None, str]:
        """
        Asks the VLM to compare two candidate objects for fit in a room scene.

        Args:
            room_pil (PIL.Image): The room scene image.
            query (str): The textual query/clue describing the desired object.
            candidate1_pil (PIL.Image): The first candidate object image.
            candidate2_pil (PIL.Image): The second candidate object image.
            cand1_id (str): Identifier for candidate 1 (e.g., filename or index).
            cand2_id (str): Identifier for candidate 2 (e.g., filename or index).

        Returns:
            tuple: (preferred_candidate_index, reasoning_text)
                - preferred_candidate_index (int | None): 1 if candidate 1 is better,
                                                          2 if candidate 2 is better.
                                                          None on failure.
                - reasoning_text (str): The reasoning text from the VLM.
        """
        start_time = time.time()
        logger.info(f"Comparing Candidate '{cand1_id}' vs Candidate '{cand2_id}' for fit...")

        images = [room_pil.convert("RGB"), candidate1_pil.convert("RGB"), candidate2_pil.convert("RGB")]
        content = [
            {"type": "text", "text": "Reference Room Scene (with a blue masked object):"},
            {"type": "image"}, # Placeholder for room_pil
            {"type": "text", "text": f"\nQuery/Description for the masked object: {query}\n"},
            {"type": "text", "text": f"Object 1 ({cand1_id}):"},
            {"type": "image"}, # Placeholder for candidate1_pil
            {"type": "text", "text": f"Candidate Object 2 ({cand2_id}):"},
            {"type": "image"}, # Placeholder for candidate2_pil
            {"type": "text", "text": (
                "\nTask: Compare Candidate 1 and Candidate 2.\n"
                "Which SINGLE candidate object fits better into the missing area of the Reference Room Scene, "
                "considering both the visual context/style of the room AND the details in the Query/Description? "
                "Analyze how well each candidate matches the requirements.\n"
                "Respond in the following format ONLY:\n"
                "<think> [Explain step-by-step your reasoning for choosing one candidate over the other, referencing the room scene and the query.] </think>\n"
                "<answer> [Write ONLY '1' if Candidate 1 is the better fit, or ONLY '2' if Candidate 2 is the better fit.] </answer>"
            )}
        ]

        messages = [{"role": "user", "content": content}]

        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_text],
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left", # Usually recommended for generation
                add_special_tokens=False,
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                     eos_token_id=[self.processor.tokenizer.eos_token_id,
                                  self.processor.tokenizer.pad_token_id,
                                  self.processor.tokenizer.eos_token_id]
                )

            torch.cuda.empty_cache()

            prompt_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            # Be careful with skip_special_tokens=True if < > symbols are eos tokens
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            logger.debug(f"Raw VLM Output for ({cand1_id} vs {cand2_id}):\n{output_text}")

            # Extract thinking and answer
            think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL | re.IGNORECASE)
            thinking_process = think_match.group(1).strip() if think_match else "No <think> tag found."

            answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL | re.IGNORECASE)
            preferred_candidate_index = None # Default to None (failure)

            if answer_match:
                answer_content = answer_match.group(1).strip()
                if answer_content == '1':
                    preferred_candidate_index = 1
                elif answer_content == '2':
                    preferred_candidate_index = 2
                else:
                    error_msg = f"Invalid answer '{answer_content}'. Expected '1' or '2'."
                    logger.error(f"VLM comparison failed ({cand1_id} vs {cand2_id}): {error_msg}")
                    thinking_process += f"\n[Error: {error_msg}]"
            else:
                error_msg = "No <answer> tag found."
                logger.error(f"VLM comparison failed ({cand1_id} vs {cand2_id}): {error_msg}")
                thinking_process += f"\n[Error: {error_msg}]"

            end_time = time.time()
            if preferred_candidate_index is not None:
                logger.info(f"Comparison ({cand1_id} vs {cand2_id}) -> Winner: Candidate {preferred_candidate_index}. Time: {end_time - start_time:.2f}s")
            else:
                 logger.warning(f"Comparison ({cand1_id} vs {cand2_id}) -> Failed. Time: {end_time - start_time:.2f}s")

            return preferred_candidate_index, thinking_process

        except Exception as e:
            error_msg = f"Exception during VLM comparison ({cand1_id} vs {cand2_id}): {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def run_elimination_tournament(self, room_pil: Image.Image, query: str,
                                  candidates_data: list[tuple[str, Image.Image]]) -> tuple[tuple[str, Image.Image] | None, str, bool]:
        """
        Ranks candidates using a single-elimination tournament bracket.

        Args:
            room_pil (PIL.Image): The room scene image.
            query (str): The textual query/clue.
            candidates_data (list): A list of tuples [(filename, pil_image), ...].

        Returns:
            tuple: (winner_data, aggregated_thinking_log, success)
                - winner_data (tuple | None): (filename, pil_image) of the winning candidate, or None if failed.
                - aggregated_thinking_log (str): Log of all comparisons made.
                - success (bool): True if the tournament completed without errors, False otherwise.
        """
        n = len(candidates_data)
        if n == 0:
            return None, "No candidates provided for tournament.", False
        if n == 1:
            return candidates_data[0], "Only one candidate, automatically the winner.", True

        aggregated_thinking = f"--- Starting Elimination Tournament with {n} candidates ---\n"
        # Ensure candidate list is shuffled or sorted consistently if needed, otherwise uses input order
        current_round_candidates = copy.deepcopy(candidates_data) # Work on a copy
        round_num = 1
        error_occurred = False

        while len(current_round_candidates) > 1:
            aggregated_thinking += f"\n--- Round {round_num} --- ({len(current_round_candidates)} candidates)\n"
            next_round_candidates = []
            winners_this_round = 0
            losers_this_round = 0

            has_bye = len(current_round_candidates) % 2 != 0
            num_matches = len(current_round_candidates) // 2

            for i in range(num_matches):
                cand1_data = current_round_candidates[2 * i]
                cand2_data = current_round_candidates[2 * i + 1]
                cand1_fname, cand1_pil = cand1_data
                cand2_fname, cand2_pil = cand2_data

                aggregated_thinking += f"\nMatch: {cand1_fname} vs {cand2_fname}\n"

                # Call the pairwise comparison method of this class
                preferred_index, thinking = self.compare_two_candidates_for_fit(
                    room_pil, query, cand1_pil, cand2_pil, cand1_fname, cand2_fname
                )
                aggregated_thinking += thinking

                if preferred_index == 1:
                    next_round_candidates.append(cand1_data)
                    aggregated_thinking += f"\nWinner: {cand1_fname}\n"
                    winners_this_round += 1
                    losers_this_round += 1
                elif preferred_index == 2:
                    next_round_candidates.append(cand2_data)
                    aggregated_thinking += f"\nWinner: {cand2_fname}\n"
                    winners_this_round += 1
                    losers_this_round += 1
                else:
                    # Comparison failed!
                    aggregated_thinking += "\n[Comparison Error - Match outcome undetermined]\n"
                    error_occurred = True
                    losers_this_round += 2 # Both effectively eliminated due to error

            # Handle the candidate with a bye
            if has_bye:
                bye_candidate_data = current_round_candidates[-1]
                next_round_candidates.append(bye_candidate_data)
                aggregated_thinking += f"\nMatch: {bye_candidate_data[0]} gets a bye.\nWinner: {bye_candidate_data[0]}\n"
                winners_this_round += 1

            current_round_candidates = next_round_candidates
            aggregated_thinking += f"End of Round {round_num}: {winners_this_round} winners, {losers_this_round} losers/eliminated.\n"
            round_num += 1

            # Safety break
            # Using log2(n) + buffer might be more accurate, but n+2 is simpler and safe.
            if round_num > n + 2 :
                 aggregated_thinking += "\n[Error: Tournament exceeded expected rounds - halting]\n"
                 error_occurred = True
                 break

        # Final result determination
        if len(current_round_candidates) == 1 and not error_occurred:
            winner_data = current_round_candidates[0]
            aggregated_thinking += f"\n--- Tournament Winner: {winner_data[0]} ---\n"
            return winner_data, aggregated_thinking, True
        elif len(current_round_candidates) == 1 and error_occurred:
             winner_data = current_round_candidates[0]
             aggregated_thinking += f"\n--- Tournament Completed with Errors - Finalist: {winner_data[0]} (Result may be unreliable) ---\n"
             return winner_data, aggregated_thinking, False # Completed, but with errors
        elif len(current_round_candidates) > 1 :
            aggregated_thinking += "\n--- Tournament Failed: Multiple candidates remaining due to errors ---\n"
            return None, aggregated_thinking, False
        else: # len == 0
            aggregated_thinking += "\n--- Tournament Failed: No winner determined due to comparison errors ---\n"
            return None, aggregated_thinking, False

    def find_best_candidate_tournament(self, room_pil: Image.Image,
                                     candidates_data: list[tuple[str, Image.Image]],
                                     description: str) -> tuple[str, str]:
        """
        Finds the best candidate using the elimination tournament method.

        Args:
            room_pil (PIL.Image): The room scene image.
            candidates_data (list): List of (filename, PIL.Image) tuples for candidates.
            description (str): Text description/clue for the missing object.

        Returns:
            tuple: (thinking_log, winner_filename_or_message)
        """
        start_total_time = time.time()
        thinking_log = "--- Starting Elimination Tournament Workflow ---\n"

        if not candidates_data:
             return thinking_log + "Error: No candidate data provided.", "Failed: No candidates."

        thinking_log += f"Processing {len(candidates_data)} candidates with description: '{description[:50]}...'\n"

        winner_data, tournament_log, success = self.run_elimination_tournament(
            room_pil, description, candidates_data
        )

        thinking_log += tournament_log # Append detailed tournament log

        end_total_time = time.time()
        total_time = end_total_time - start_total_time
        thinking_log += f"\nTotal Tournament Time: {total_time:.2f}s\n"

        if success and winner_data:
            winner_filename = winner_data[0]
            thinking_log += f"--- Tournament Winner Found: {winner_filename} ---"
            return thinking_log, f"Winner: {winner_filename}"
        elif winner_data: # Tournament finished with errors, but identified a finalist
            winner_filename = winner_data[0]
            thinking_log += f"--- Tournament Finished with Errors. Finalist: {winner_filename} (Result may be unreliable) ---"
            return thinking_log, f"Finalist (Errors Occurred): {winner_filename}"
        else: # Tournament failed to determine a winner
            thinking_log += f"--- Tournament Failed to determine a winner ---"
            return thinking_log, "Tournament Failed: No winner found (check logs for errors)."

    # --- Original single-prompt method (kept for reference) ---
    def process_room_completion_single_prompt(self, room_pil: Image.Image,
                                             candidates_data: list[tuple[str, Image.Image]],
                                             description: str) -> tuple[str, str]:
        """
        Processes the room scene, candidates, and description in a single prompt.
        (This was the original method before the tournament implementation).

        Args:
            room_pil (PIL.Image): The room scene image.
            candidates_data (list): List of (filename, PIL.Image) tuples for candidates.
            description (str): Text description/clue for the missing object.

        Returns:
            tuple: (thinking_process, chosen_object_filename_or_error)
        """
        logger.info("Processing room completion using single prompt method.")
        images = []
        candidate_filenames = []
        content = []

        try:
            room_filename = "room_scene.png" # Placeholder name if not available
            room_pil = room_pil.convert("RGB")
            images.append(room_pil)

            content.append({"type": "text", "text": f"Main Scene Image (Filename: {room_filename}): This image shows a room with a missing object."})
            content.append({"type": "image"}) # Placeholder for the room image

            content.append({"type": "text", "text": f"\nClue for the missing object: {description}\n"})

            content.append({"type": "text", "text": "Candidate Objects to choose from:"})
            if not candidates_data:
                 return "Error: No candidate data provided.", "Failed: No candidates."

            for i, (cand_filename, cand_pil) in enumerate(candidates_data):
                logger.debug(f" - Processing Candidate {i+1}: '{cand_filename}'")
                images.append(cand_pil.convert("RGB"))
                candidate_filenames.append(cand_filename)

                content.append({"type": "text", "text": f"- Candidate {i+1} (index: {i+1}, filename: {cand_filename}):"})
                content.append({"type": "image"}) # Placeholder for this candidate image

            logger.info(f"Prepared 1 room scene and {len(candidates_data)} candidates for single prompt.")

        except Exception as e:
            logger.error(f"Error preparing images/content for single prompt: {e}", exc_info=True)
            return f"Error preparing inputs: {e}", "Failed to process inputs."

        task_instruction = (
            "\nTask Instructions:\n"
            "1. Analyze the 'Main Scene Image' to understand the room's context and where an object is missing.\n"
            "2. Consider the 'Clue for the missing object', paying close attention to specific visual and semantic details (e.g., number of pillows, color, texture, pattern like floral print, material, etc.).\n"
            "3. Examine each 'Candidate Object' image provided.\n"
            "4. Perform detailed visual and semantic reasoning to determine which *single* Candidate Object best fits into the missing area of the Main Scene Image, considering both the overall context of the room and all specific characteristics described in the Clue.\n"
            "   - Compare each object carefully: Does it have the described features? How well does it match the clue?\n"
            "   - If the clue describes a specific pattern, color, or object count (e.g., 'two pillows and floral simple headboard'), verify these against each candidate image.\n"
            "5. Respond in the following format ONLY:\n"
            "<think> [Provide your detailed step-by-step reasoning for choosing the best candidate, explaining why it fits better than the others based on both the room context and every detail in the clue.] </think>\n"
            "<answer> [Write the exact object index (1-indexed) (e.g., '1') of the chosen Candidate Object here.] </answer>"
        )
        content.append({"type": "text", "text": task_instruction})
        messages = [{"role": "user", "content": content}]

        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.debug(f"Formatted Prompt Preview (first 500 chars):\n{chat_text[:500]}...")

            inputs = self.processor(
                text=[chat_text],
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            ).to(self.device)
            logger.debug("Processor inputs prepared for single prompt generation.")

            with torch.no_grad():
                logger.info("Generating response (single prompt)...")
                generated_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                logger.info("Generation complete (single prompt).")

            torch.cuda.empty_cache()

            prompt_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            logger.debug(f"Raw Decoded Output (single prompt):\n{output_text}")

            think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL | re.IGNORECASE)
            thinking_process = think_match.group(1).strip() if think_match else "No <think> tag found or content empty."

            answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL | re.IGNORECASE)
            chosen_object_filename = "No <answer> tag found or content empty."
            if answer_match:
                try:
                    chosen_index = int(answer_match.group(1).strip())
                    if 1 <= chosen_index <= len(candidate_filenames):
                        chosen_object_filename = candidate_filenames[chosen_index - 1]
                        logger.info(f"Extracted Answer Index: {chosen_index}, Filename: {chosen_object_filename}")
                    else:
                         error_msg = f"Answer index '{chosen_index}' out of range (1-{len(candidate_filenames)})."
                         logger.warning(error_msg)
                         chosen_object_filename = f"Error: {error_msg}"
                except ValueError:
                    error_msg = f"Invalid non-integer answer '{answer_match.group(1).strip()}'."
                    logger.warning(error_msg)
                    chosen_object_filename = f"Error: {error_msg}"

            return thinking_process, chosen_object_filename

        except Exception as e:
            logger.error(f"Error during single prompt processing or generation: {e}", exc_info=True)
            return f"Error during processing: {e}", "Failed to generate response."

def load_image(image_path: str) -> Image.Image | None:
    """Safely loads an image using PIL."""
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: {image_path}")
        return None
    try:
        img = Image.open(image_path)
        # It's good practice to load the image data immediately
        img.load()
        # Convert to RGB if needed, but do it later just before passing to model
        # return img.convert("RGB")
        return img
    except UnidentifiedImageError:
        logger.error(f"Cannot identify image file (corrupted or wrong format): {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def get_query_for_scene(scene_id: str, scene_base_path: str) -> str:
    """
    Retrieves the query/description for a given scene_id by reading
    a text file located at '{scene_base_path}/{scene_id}/query.txt'.

    Args:
        scene_id (str): The ID of the scene.
        scene_base_path (str): The base directory where scene folders are located.

    Returns:
        str: The query text read from the file, or a default query if the
             file is not found or cannot be read.
    """
    default_query = "Describe the missing object based on the room context." # Fallback
    query_file_path = os.path.join(scene_base_path, scene_id, "query.txt")

    if not os.path.exists(query_file_path):
        logger.warning(f"Query file not found: {query_file_path}. Using default query for scene {scene_id}.")
        return default_query

    try:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_text = f.read().strip()
        if not query_text:
            logger.warning(f"Query file is empty: {query_file_path}. Using default query for scene {scene_id}.")
            return default_query
        return query_text
    except Exception as e:
        logger.error(f"Error reading query file {query_file_path}: {e}. Using default query for scene {scene_id}.")
        return default_query



def rerank_predictions_with_vlm(
    input_csv_path: str,
    output_csv_path: str,
    n_candidates_to_rerank: int,
    scene_base_path: str,
    object_base_path: str,
    vlm_handler: VlmRoomCompletion,
    # query_source: dict | str | None, # Dict {scene_id: query}, fixed string, or None
    rerank_log_file: str | None = None # Optional file to save detailed logs
):
    """
    Reads a CSV of ranked predictions, re-ranks the top N using VLM, and saves the updated CSV.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the updated CSV file.
        n_candidates_to_rerank (int): Number of top candidates (N) to re-rank for each scene.
        scene_base_path (str): Base directory path for scene images.
        object_base_path (str): Base directory path for object images.
        vlm_handler (VlmRoomCompletion): An initialized instance of the VLM handler class.
        query_source (dict | str | None): Source for text queries. Can be:
            - A dictionary mapping scene_id (str) to query text (str).
            - A single string to be used as the query for all scenes.
            - None, in which case a default query will be used.
        rerank_log_file (str | None): If provided, detailed tournament logs are saved here.
    """
    logger.info(f"Starting VLM re-ranking process for {input_csv_path}")
    logger.info(f"Re-ranking top {n_candidates_to_rerank} candidates per scene.")
    logger.info(f"Output will be saved to: {output_csv_path}")
    if rerank_log_file:
        logger.info(f"Detailed logs will be saved to: {rerank_log_file}")

    try:
        # Read the CSV, assuming the first row is the header
        df = pd.read_csv(input_csv_path, dtype=str) # Read all as string initially
        logger.info(f"Loaded {len(df)} rows from {input_csv_path}")

        # Ensure N is not larger than available object columns (assuming columns '1' to '10')
        num_object_cols = len([col for col in df.columns if col.isdigit() and int(col) > 0])
        if n_candidates_to_rerank > num_object_cols:
            logger.warning(f"Requested N={n_candidates_to_rerank} is greater than available object columns ({num_object_cols}). Clamping N to {num_object_cols}.")
            n_candidates_to_rerank = num_object_cols
        if n_candidates_to_rerank < 1:
             logger.error("n_candidates_to_rerank must be at least 1. Aborting.")
             return

        # Column names for objects (assuming '1' through '10' or however many exist)
        object_cols = [str(i) for i in range(1, num_object_cols + 1)]
        scene_col = '0' # Assuming scene ID is in column '0'

        all_logs = [] # Store logs for saving later

        # Iterate through each scene (row) in the DataFrame
        for index, row in df.iterrows():
            scene_id = row[scene_col]
            scene_log_prefix = f"Scene {index+1}/{len(df)} (ID: {scene_id}):"
            logger.info(f"{scene_log_prefix} Processing...")
            current_log = [f"--- Log for Scene ID: {scene_id} ---"]

            # 1. Get Scene Image Path and Load
            scene_image_path = os.path.join(scene_base_path, scene_id, "masked.png")
            room_pil = load_image(scene_image_path)
            if room_pil is None:
                logger.error(f"{scene_log_prefix} Failed to load scene image. Skipping re-ranking for this scene.")
                current_log.append("ERROR: Failed to load scene image. Original ranking kept.")
                all_logs.append("\n".join(current_log))
                continue # Skip to the next row

            # 2. Get the query for this scene
            query = get_query_for_scene(scene_id, scene_base_path)
            current_log.append(f"Query: {query}")

            # 3. Get Original Ranked Object IDs
            original_ranked_ids = row[object_cols].tolist()

            # 4. Select Top N Candidates for Re-ranking
            ids_to_rerank = original_ranked_ids[:n_candidates_to_rerank]
            current_log.append(f"Original Top {n_candidates_to_rerank} IDs: {ids_to_rerank}")

            if len(ids_to_rerank) < 2:
                logger.info(f"{scene_log_prefix} Fewer than 2 candidates ({len(ids_to_rerank)}) selected for re-ranking. No tournament needed.")
                current_log.append("INFO: Fewer than 2 candidates. Keeping original order.")
                all_logs.append("\n".join(current_log))
                continue # No re-ranking needed if 0 or 1 candidate

            # 5. Load Top N Candidate Images
            candidates_data = []
            for obj_id in ids_to_rerank:
                if pd.isna(obj_id): # Handle potential empty cells
                    logger.warning(f"{scene_log_prefix} Found NaN object ID in top {n_candidates_to_rerank}. Skipping it.")
                    continue
                obj_image_path = os.path.join(object_base_path, str(obj_id), "image.jpg") # Ensure obj_id is string
                obj_pil = load_image(obj_image_path)
                if obj_pil:
                    candidates_data.append((str(obj_id), obj_pil)) # Store as (id_string, image)
                else:
                    logger.warning(f"{scene_log_prefix} Failed to load image for candidate object {obj_id}.")
                    # Decide whether to proceed without this candidate or skip the scene
                    # Let's proceed with fewer candidates if some fail to load
                    current_log.append(f"WARNING: Failed to load image for candidate {obj_id}.")

            if len(candidates_data) < 2:
                logger.warning(f"{scene_log_prefix} Fewer than 2 candidate images loaded successfully ({len(candidates_data)}). Cannot run tournament.")
                current_log.append("ERROR: Fewer than 2 candidate images loaded. Keeping original order.")
                all_logs.append("\n".join(current_log))
                # Close images if loaded
                for _, img in candidates_data: img.close()
                room_pil.close()
                continue

            # 6. Run the VLM Tournament
            logger.info(f"{scene_log_prefix} Running VLM tournament for {len(candidates_data)} candidates...")
            try:
                # Pass only the successfully loaded candidates
                winner_data, tournament_log, success = vlm_handler.run_elimination_tournament(
                    room_pil, query, candidates_data
                )
                current_log.append("\n--- Tournament Log ---")
                current_log.append(tournament_log)
                current_log.append("--- End Tournament Log ---\n")

                new_top_n_order = []
                if winner_data:
                    winner_id = winner_data[0]
                    logger.info(f"{scene_log_prefix} Tournament winner/finalist: {winner_id} (Success: {success})")
                    current_log.append(f"Tournament winner/finalist ID: {winner_id} (Success Flag: {success})")

                    # Construct the new top N order: winner first, then others
                    new_top_n_order.append(winner_id)
                    original_top_n_ids_loaded = [data[0] for data in candidates_data]
                    for oid in original_top_n_ids_loaded:
                        if oid != winner_id:
                            new_top_n_order.append(oid)

                    # If some candidates failed to load, their IDs are missing here.
                    # Check if the length matches the number of *loaded* candidates.
                    if len(new_top_n_order) != len(candidates_data):
                         logger.error(f"{scene_log_prefix} Internal error: Length mismatch in reordered list. {len(new_top_n_order)} vs {len(candidates_data)}")
                         # Fallback to original order for safety
                         new_top_n_order = [data[0] for data in candidates_data]
                         current_log.append("ERROR: Internal error during reordering. Keeping original order of loaded candidates.")

                else:
                    # Tournament failed completely
                    logger.warning(f"{scene_log_prefix} VLM tournament failed to determine a winner.")
                    current_log.append("ERROR: Tournament failed. Keeping original order of loaded candidates.")
                    # Keep the original order of the candidates that were *loaded*
                    new_top_n_order = [data[0] for data in candidates_data]

                # 7. Combine Re-ranked Top N with Remaining Candidates
                # We need to fill the top N slots.
                # If fewer than N candidates were loaded/reranked, we need to pull from the original N list.
                final_top_n = []
                added_ids = set()

                # Add the VLM-ranked ones first
                for oid in new_top_n_order:
                    if oid not in added_ids:
                        final_top_n.append(oid)
                        added_ids.add(oid)

                # Fill remaining slots up to N from the original top N list (excluding already added ones)
                num_needed = n_candidates_to_rerank - len(final_top_n)
                if num_needed > 0:
                     original_top_n_ids_all = original_ranked_ids[:n_candidates_to_rerank]
                     for oid in original_top_n_ids_all:
                         if len(final_top_n) >= n_candidates_to_rerank:
                             break
                         if oid not in added_ids and not pd.isna(oid):
                             final_top_n.append(str(oid)) # Ensure string
                             added_ids.add(oid)


                # Get the remaining candidates (N+1 onwards) from the original list
                remaining_ids = original_ranked_ids[n_candidates_to_rerank:]
                # Filter out NaNs just in case
                remaining_ids_clean = [str(oid) for oid in remaining_ids if not pd.isna(oid)]

                # Combine the final top N and the rest
                final_ranking = final_top_n + remaining_ids_clean
                current_log.append(f"Final Re-ranked IDs ({len(final_ranking)} total): {final_ranking}")

                # 8. Update the DataFrame Row
                # Ensure the final list has the correct number of columns for the DataFrame
                if len(final_ranking) < len(object_cols):
                    # Pad with None or empty string if needed (depends on desired output format)
                    final_ranking.extend([None] * (len(object_cols) - len(final_ranking)))
                elif len(final_ranking) > len(object_cols):
                    # Truncate if too long (shouldn't happen with correct logic)
                    final_ranking = final_ranking[:len(object_cols)]

                df.loc[index, object_cols] = final_ranking
                logger.info(f"{scene_log_prefix} Updated ranking in DataFrame.")

            except Exception as e:
                logger.error(f"{scene_log_prefix} Unhandled exception during tournament/update: {e}", exc_info=True)
                current_log.append(f"CRITICAL ERROR: {e}. Original ranking kept.")
            finally:
                # Ensure PIL images are closed to free memory
                if room_pil:
                    room_pil.close()
                for _, img in candidates_data:
                    if img:
                        img.close()
                # Clear CUDA cache frequently if using GPU
                if vlm_handler.device == 'cuda':
                    torch.cuda.empty_cache()

            all_logs.append("\n".join(current_log))
            logger.info(f"{scene_log_prefix} Processing finished.")
            # Optional: Add a small delay if hitting API limits or for stability
            # time.sleep(1)

        # 9. Save the Updated DataFrame
        logger.info(f"Saving updated DataFrame to {output_csv_path}...")
        df.to_csv(output_csv_path, index=False)
        logger.info("Re-ranking process completed.")

        # 10. Save the detailed log file
        if rerank_log_file:
            try:
                with open(rerank_log_file, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(all_logs))
                logger.info(f"Detailed logs saved to {rerank_log_file}")
            except Exception as e:
                logger.error(f"Failed to write log file {rerank_log_file}: {e}")

    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {input_csv_path}")
    except Exception as e:
        logger.error(f"An error occurred during the re-ranking process: {e}", exc_info=True)


# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration ---
    INPUT_CSV = "MealsRetrieval0.9283.csv"        # Your input CSV file path
    OUTPUT_CSV = "predictions_reranked.csv" # Output file path
    LOG_FILE = "reranking_details.log"      # Detailed log file path
    N_RERANK = 5                           # How many top candidates to re-rank (e.g., 5)
    SCENE_DIR = "../private_data/scenes"   # Base path to scene folders
    OBJECT_DIR = "../private_data/objects" # Base path to object folders
    MODEL_PATH = "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps" # Or your chosen model
    DEVICE = None                          # Auto-detect ('cuda' or 'cpu')

    # --- Run the Re-ranking ---
    try:
        # Initialize the VLM Handler
        # Make sure you have the model downloaded or accessible
        # This might take time and memory
        print("Initializing VLM Handler...")
        vlm = VlmRoomCompletion(model_path=MODEL_PATH, device=DEVICE)
        print("VLM Handler initialized.")

        # Run the re-ranking function
        rerank_predictions_with_vlm(
            input_csv_path=INPUT_CSV,
            output_csv_path=OUTPUT_CSV,
            n_candidates_to_rerank=N_RERANK,
            scene_base_path=SCENE_DIR,
            object_base_path=OBJECT_DIR,
            vlm_handler=vlm,
            rerank_log_file=LOG_FILE
        )

        print(f"\nRe-ranking finished. Updated predictions saved to {OUTPUT_CSV}")
        print(f"Detailed logs saved to {LOG_FILE}")

    except ImportError as e:
        logger.error(f"Import error: {e}. Please ensure all required libraries (transformers, torch, Pillow, pandas) are installed.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)

# Example of how to use the library (optional, can be removed or kept for testing)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True) # Force reconfig for testing
#     logger.info("Running VLM Room Completion Library Self-Test...")

#     # Create dummy images for testing
#     dummy_room = Image.open('../test-imgs/masked.png')
#     dummy_cand1 = Image.open('../test-imgs/giuongdung.jpg')
#     dummy_cand2 = Image.open('../test-imgs/giuongsai1.png')
#     dummy_cand3 = Image.open('../test-imgs/giuonhsai1.jpg')

#     test_candidates = [
#         ("giuongdung.jpg", dummy_cand1),
#         ("giuongsai1.png", dummy_cand2),
#         ("giuonhsai1.jpg", dummy_cand3),
#     ]
#     test_query = "a bunk bed consists of two beds stacked on top of each other, typically with a ladder or stairs connecting them."

#     try:
#         # Use a smaller model for faster local testing if available, e.g. "llava-hf/llava-1.5-7b-hf"
#         # Or stick to the original: "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps"
#         # NOTE: Requires significant compute resources. Test might fail without appropriate hardware.
#         vlm_handler = VlmRoomCompletion(model_path="omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps")

#         # --- Test Tournament ---
#         logger.info("\n--- Testing Tournament ---")
#         think_log, winner_msg = vlm_handler.find_best_candidate_tournament(
#             dummy_room, test_candidates, test_query
#         )
#         logger.info(f"Tournament Result: {winner_msg}")
#         logger.info(f"Tournament Log:\n{think_log}")

#         # --- Test Single Prompt (Optional) ---
#         # logger.info("\n--- Testing Single Prompt ---")
#         # think_log_sp, winner_msg_sp = vlm_handler.process_room_completion_single_prompt(
#         #     dummy_room, test_candidates, test_query
#         # )
#         # logger.info(f"Single Prompt Result: {winner_msg_sp}")
#         # logger.info(f"Single Prompt Thinking:\n{think_log_sp}")

#     except Exception as e:
#         logger.error(f"Self-test failed: {e}", exc_info=True)

#     logger.info("VLM Room Completion Library Self-Test Finished.")