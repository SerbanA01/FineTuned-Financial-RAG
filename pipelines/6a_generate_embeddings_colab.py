# pipelines/6a_generate_embeddings_colab.py

"""
GENERATE EMBEDDINGS (COLAB GPU)

This script is designed to be run in a Google Colab environment with a GPU.
Its purpose is to:
1. Mount the user's Google Drive to access project files.
2. Load the filtered, chunked .jsonl files for a specific market (e.g., 'sp500').
3. Use a pre-trained Sentence Transformer model ('BAAI/bge-base-en-v1.5') to
   generate vector embeddings for each text chunk.
4. Save the embeddings (as .npy files) and their corresponding metadata payloads
   (as .jsonl files) back to Google Drive. The output is partitioned into
   versioned parts to handle large datasets gracefully.

Instructions for use in Google Colab:
1. Upload this script to your Colab environment.
2. Ensure your Google Drive has a folder structure like:
   /MyDrive/YourProject/data/processed/chunks/
   /MyDrive/YourProject/data/processed/embeddings/
3. Place the .jsonl files from pipeline step 5 into the 'chunks' directory.
4. Install necessary libraries by running this command in a Colab cell:
   !pip install sentence-transformers torch numpy
5. Run the script from a Colab cell, specifying the target market:
   !python 6a_generate_embeddings_colab.py --market sp500
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# --- Configuration ---
# The name of the Sentence Transformer model to use for generating embeddings.
MODEL_NAME_ST = 'BAAI/bge-base-en-v1.5'
# The standard mount point for Google Drive in Colab environments.
DRIVE_MOUNT_POINT = '/content/drive'

# IMPORTANT: This must be the path to your project folder within Google Drive.
# The script will read from and write to subdirectories within this path.
PROJECT_DIR_ON_DRIVE = 'MyDrive/FINANCIAL_RAG_WEBAPP' # <-- ADJUST IF NEEDED

# The number of text chunks to process in a single batch on the GPU.
# This should be tuned based on the available VRAM of the Colab GPU instance.
ST_ENCODE_BATCH_SIZE = 512

# The number of chunks to include in each saved output file part.
# This breaks down a potentially huge dataset into more manageable file sizes.
SAVE_PART_SIZE_CHUNKS = 100000

# A global variable to hold the initialized model.
# This acts as a singleton to prevent reloading the large model into memory.
_embedding_model_global: SentenceTransformer | None = None

def mount_gdrive_if_needed() -> bool:
    """
    Mounts the user's Google Drive to the Colab filesystem if not already mounted.

    Returns:
        bool: True if the drive is successfully mounted or already mounted, False otherwise.
    """
    if not os.path.isdir(os.path.join(DRIVE_MOUNT_POINT, "MyDrive")):
        print("Mounting Google Drive...")
        try:
            from google.colab import drive
            # Force remount to ensure a fresh connection.
            drive.mount(DRIVE_MOUNT_POINT, force_remount=True)
            # A short pause to allow the filesystem to stabilize after mounting.
            time.sleep(5)
            if not os.path.isdir(os.path.join(DRIVE_MOUNT_POINT, "MyDrive")):
                print("ERROR: Failed to mount Google Drive.")
                return False
            print("Google Drive mounted successfully.")
        except Exception as e:
            print(f"ERROR: Exception during Google Drive mount: {e}")
            return False
    else:
        print("Google Drive appears to be already mounted.")
    return True

def initialize_embedding_model_once() -> SentenceTransformer | None:
    """
    Loads and initializes the Sentence Transformer model.

    This function uses a global variable to ensure the model is loaded only once
    (a singleton pattern), which saves significant time and memory on subsequent calls.
    It automatically selects a CUDA device if available, otherwise falls back to CPU.

    Returns:
        SentenceTransformer | None: The initialized model instance, or None on failure.
    """
    global _embedding_model_global
    if _embedding_model_global is None:
        # Prioritize CUDA GPU for performance; fall back to CPU if not available.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            print(f"Loading BGE model: {MODEL_NAME_ST} on device: {device}...")
            start_load_time = time.time()
            _embedding_model_global = SentenceTransformer(MODEL_NAME_ST, device=device)
            end_load_time = time.time()
            print(f"Model loaded in {end_load_time - start_load_time:.2f} seconds.")
            if device == 'cpu':
                print("WARNING: Model loaded on CPU. Embedding generation will be very slow.")
        except Exception as e:
            print(f"FATAL: Error loading Sentence Transformer model: {e}")
            raise
    return _embedding_model_global


def main():
    """
    Main function to orchestrate the embedding generation and saving process.
    """
    # Set up command-line arguments to make the script reusable for different markets.
    parser = argparse.ArgumentParser(description="Generate embeddings for chunked data using a Colab GPU.")
    parser.add_argument(
        "--market",
        type=str,
        required=True,
        help="The market index suffix for the input files (e.g., 'sp500')."
    )
    args = parser.parse_args()

    # The entire process is contingent on accessing Google Drive.
    if not mount_gdrive_if_needed():
        print("Aborting: Google Drive issue.")
        sys.exit(1)

    # Initialize the embedding model. This is a critical step that can fail.
    try:
        embedding_model = initialize_embedding_model_once()
        if not embedding_model:
            sys.exit(1)
    except Exception:
        print("Aborting: Model initialization failed.")
        sys.exit(1)

    # Construct the necessary input and output paths relative to the project root in Drive.
    full_project_path = os.path.join(DRIVE_MOUNT_POINT, PROJECT_DIR_ON_DRIVE)
    input_chunks_dir = os.path.join(full_project_path, 'data/processed/chunks')
    save_output_dir = os.path.join(full_project_path, 'data/processed/embeddings')
    os.makedirs(save_output_dir, exist_ok=True)

    # Define the specific input files to be processed based on the market argument.
    input_jsonl_files = [
        os.path.join(input_chunks_dir, f"processed_10k_{args.market}.jsonl"),
        os.path.join(input_chunks_dir, f"processed_10q_{args.market}.jsonl"),
        os.path.join(input_chunks_dir, f"processed_earnings_{args.market}.jsonl"),
    ]
    
    # Create a base name for the output files for consistency.
    save_file_basename = f"embeddings_{args.market}"

    all_texts_to_process = []
    all_payloads_to_process = []

    print(f"\n--- Loading Chunks for market '{args.market}' from Google Drive ---")
    # This design loads all data into memory first. This is feasible in Colab for
    # moderately sized datasets and simplifies the subsequent batching logic.
    for jsonl_file_path in input_jsonl_files:
        if not os.path.exists(jsonl_file_path):
            print(f"Warning: File not found - {jsonl_file_path}. Skipping.")
            continue

        print(f"Reading from: {os.path.basename(jsonl_file_path)}...")
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunk_data = json.loads(line)
                        text = chunk_data['text']
                        metadata = chunk_data['metadata']
                        
                        # The payload is the metadata that will be stored in the vector DB
                        # alongside the embedding. We include the original text for context.
                        payload_for_saving = metadata.copy()
                        payload_for_saving["chunk_text"] = text
                        
                        all_texts_to_process.append(text)
                        all_payloads_to_process.append(payload_for_saving)
                    except (json.JSONDecodeError, KeyError) as e_line:
                        print(f"  Warning: Error processing a line in {os.path.basename(jsonl_file_path)}: {e_line}. Skipping line.")

    if not all_texts_to_process:
        print("No text chunks were loaded to process. Aborting."); return
    print(f"\nTotal chunks loaded for embedding: {len(all_texts_to_process)}")

    overall_start_time = time.time()
    total_chunks_embedded_and_saved = 0

    print(f"\nStarting embedding (GPU batch: {ST_ENCODE_BATCH_SIZE}) and saving to Drive in parts of ~{SAVE_PART_SIZE_CHUNKS} chunks...")

    # Process and save the data in large, numbered parts.
    for part_start_idx in range(0, len(all_texts_to_process), SAVE_PART_SIZE_CHUNKS):
        # Slice the in-memory lists to get the data for the current part.
        part_texts_slice = all_texts_to_process[part_start_idx : part_start_idx + SAVE_PART_SIZE_CHUNKS]
        part_payloads_slice = all_payloads_to_process[part_start_idx : part_start_idx + SAVE_PART_SIZE_CHUNKS]

        if not part_texts_slice: continue

        part_num_str = str(part_start_idx // SAVE_PART_SIZE_CHUNKS).zfill(4)
        print(f"\nProcessing and saving part {part_num_str} ({len(part_texts_slice)} chunks)...")

        # This is the core computation step where the model generates embeddings.
        # It's highly optimized to use the GPU for parallel processing.
        # `normalize_embeddings=True` is crucial for many vector similarity metrics (like cosine similarity).
        batch_embeddings_np = embedding_model.encode(
            part_texts_slice,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=ST_ENCODE_BATCH_SIZE
        )

        # Save the part's data to Google Drive.
        embeddings_save_path = os.path.join(save_output_dir, f"{save_file_basename}_embeddings_part_{part_num_str}.npy")
        payloads_save_path = os.path.join(save_output_dir, f"{save_file_basename}_payloads_part_{part_num_str}.jsonl")

        try:
            print(f"  Saving embeddings part to {os.path.basename(embeddings_save_path)}")
            # Saving as a NumPy binary file is highly efficient for numerical data.
            embeddings_array = np.array(batch_embeddings_np, dtype=np.float32)
            np.save(embeddings_save_path, embeddings_array)

            print(f"  Saving payloads part to {os.path.basename(payloads_save_path)}")
            # Saving payloads as JSONL is flexible and easy to parse later.
            with open(payloads_save_path, 'w', encoding='utf-8') as f_payloads:
                for payload in part_payloads_slice:
                    f_payloads.write(json.dumps(payload) + "\n")

            total_chunks_embedded_and_saved += len(part_texts_slice)
            print(f"  Part {part_num_str} saved successfully. Total chunks saved so far: {total_chunks_embedded_and_saved}")
        except Exception as e_save:
            print(f"    FATAL: Error saving part {part_num_str} files: {e_save}")
            continue

    processing_duration = time.time() - overall_start_time
    print(f"\n--- Embedding and Saving to Drive Complete ---")
    print(f"Total time for {total_chunks_embedded_and_saved} chunks: {processing_duration:.2f}s.")
    if total_chunks_embedded_and_saved > 0:
        print(f"Avg time/chunk: {processing_duration / total_chunks_embedded_and_saved:.4f}s.")
    
    # Report GPU memory usage and clear the cache as a good practice in CUDA programming.
    if torch.cuda.is_available():
        print(f"Peak GPU Memory Used: {torch.cuda.max_memory_allocated(device='cuda') / (1024**2):.2f} MB")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()