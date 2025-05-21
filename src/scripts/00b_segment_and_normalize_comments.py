# src/scripts/00b_segment_and_normalize_comments.py

import json
import os
import argparse
import logging
import re
from dotenv import load_dotenv
from kexp_processing.normalization import normalize_text

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def segment_raw_comment(raw_comment_text: str):
    """
    Splits a raw comment text by various potential separators in a staged manner.
    1. Explicit '---' or '--' lines.
    2. Paragraph breaks (two or more consecutive newlines).
    Segments are then normalized.
    """
    if not raw_comment_text:
        return []

    # Stage 1: Split by explicit '---' or '--' lines.
    # These are strong indicators of separate sections.
    # Pattern: A line that consists only of '---' or '--', possibly with whitespace around them on that line.
    # Matches separator on its own line, surrounded by newlines
    strong_delimiter_pattern = r'\n\s*(?:---|--)\s*\n'

    # Temporarily replace these strong delimiters to split by them first.
    # Using a unique placeholder that re.split can use.
    # We add newlines around the placeholder to preserve them for the next stage if needed,
    # or they'll be handled by normalize_text if they become leading/trailing.
    placeholder_strong = "%%STRONG_SEP%%"

    # First, handle cases where the comment might start or end with these delimiters
    # to avoid empty strings at the beginning/end of initial_chunks.
    # Initial strip to handle leading/trailing newlines before processing
    text_to_process = raw_comment_text.strip()

    # Replace the strong delimiters. Adding newlines around placeholder for clarity.
    # We use re.sub and then split.
    text_with_placeholders = re.sub(
        strong_delimiter_pattern, f"\n{placeholder_strong}\n", text_to_process)

    initial_chunks = text_with_placeholders.split(placeholder_strong)

    final_segments = []
    for chunk in initial_chunks:
        if not chunk.strip():  # Skip empty chunks that might arise from splitting
            continue

        # Stage 2: For each chunk from Stage 1, split by paragraph breaks (2+ newlines).
        # A single newline is just a line break within a paragraph.
        # Pattern: Two or more newlines.
        # Matches 2 or more newlines, with optional spaces on the empty lines
        paragraph_delimiter_pattern = r'\n\s*\n+'

        # Split the chunk by paragraph breaks.
        paragraphs = re.split(paragraph_delimiter_pattern, chunk)

        for paragraph_text in paragraphs:
            # Stage 3: Normalize each resulting paragraph.
            # normalize_text will handle remaining single newlines (converting to space),
            # excess internal spaces, and leading/trailing whitespace for the segment.
            normalized_segment = normalize_text(paragraph_text)
            if normalized_segment:  # Only add non-empty segments
                final_segments.append(normalized_segment)

    return final_segments

# ... (rest of the main() function remains the same, as it iterates over what segment_raw_comment returns)
# Ensure main() correctly calls this updated segment_raw_comment


def main():
    parser = argparse.ArgumentParser(
        description="Segments raw KEXP comments based on '--', '---', or multiple newline delimiters and normalizes each segment."
    )
    parser.add_argument(
        "input_jsonl_file",
        help="Path to input JSONL file (output of 00_extract_kexp_comments.py, with raw, unnormalized text)."
    )
    parser.add_argument(
        "output_jsonl_file",
        help="Path to output JSONL file with segmented and normalized comments."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit the number of original comments to process (for testing)."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    if not os.path.exists(args.input_jsonl_file):
        logging.error(f"Input JSONL file not found: {args.input_jsonl_file}")
        return

    processed_original_comments = 0
    total_segments_written = 0

    try:
        with open(args.input_jsonl_file, 'r', encoding='utf-8') as infile, \
                open(args.output_jsonl_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                if args.limit and i >= args.limit:
                    logging.info(
                        f"Reached processing limit of {args.limit} original comments.")
                    break

                processed_original_comments += 1
                original_comment_text_for_error_logging = "N/A"

                try:
                    record = json.loads(line)
                    raw_comment_text = record.get("text")
                    original_meta = record.get("meta", {})

                    original_comment_text_for_error_logging = raw_comment_text if raw_comment_text else "N/A"

                    if not raw_comment_text:
                        logging.warning(
                            f"Skipping line {i+1} from input due to missing or empty 'text' field: {line.strip()}")
                        continue

                    segments = segment_raw_comment(raw_comment_text)

                    if not segments:
                        normalized_full_comment = normalize_text(
                            raw_comment_text)
                        if normalized_full_comment:
                            # Add a play_id to the meta if it exists in original_meta
                            meta_for_unsegmented = {
                                "segment_index": 0, "segmentation_status": "unsegmented"}
                            if "play_id" in original_meta:
                                meta_for_unsegmented["original_play_id"] = original_meta["play_id"]
                            # Copy other relevant original_meta fields
                            for key, value in original_meta.items():
                                if key not in meta_for_unsegmented:  # Avoid overwriting original_play_id if play_id was the same
                                    meta_for_unsegmented[key] = value

                            output_record = {
                                "text": normalized_full_comment,
                                "meta": meta_for_unsegmented
                            }
                            outfile.write(json.dumps(
                                output_record, ensure_ascii=False) + '\n')
                            total_segments_written += 1
                            logging.info(
                                f"Original comment at line {i+1} written as unsegmented. Original: '{raw_comment_text[:100]}...'")
                        else:
                            logging.info(
                                f"Original comment at line {i+1} resulted in no valid segments and was empty after normalization. Original: '{raw_comment_text[:100]}...'")
                        continue

                    for seg_idx, segment_text in enumerate(segments):
                        new_meta = original_meta.copy()
                        if "play_id" in original_meta:  # Ensure original_play_id is set if play_id was present
                            # Use original_play_id, remove play_id to avoid confusion
                            new_meta["original_play_id"] = original_meta.pop(
                                "play_id", None)

                        new_meta["segment_index"] = seg_idx
                        new_meta["total_segments_from_original"] = len(
                            segments)

                        output_record = {
                            "text": segment_text,
                            "meta": new_meta
                        }
                        outfile.write(json.dumps(
                            output_record, ensure_ascii=False) + '\n')
                        total_segments_written += 1

                    if processed_original_comments % 200 == 0:
                        logging.info(
                            f"Processed {processed_original_comments} original comments, yielding {total_segments_written} segments...")

                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON line {i+1} in input: {line.strip()}")
                except Exception as e:
                    logging.error(
                        f"Error processing line {i+1} from input (original text: '{original_comment_text_for_error_logging[:100]}...'): {e}", exc_info=True)

        logging.info(
            f"Successfully processed {processed_original_comments} original comments, "
            f"writing {total_segments_written} segments to: {args.output_jsonl_file}"
        )

    except Exception as e:
        logging.error(
            f"A critical error occurred during segmentation: {e}", exc_info=True)


if __name__ == "__main__":
    main()
