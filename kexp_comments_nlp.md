This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

Additional Info:
----------------

================================================================
Directory Structure
================================================================
src/
  kexp_data.egg-info/
    dependency_links.txt
    PKG-INFO
    SOURCES.txt
    top_level.txt
  kexp_processing/
    normalization.py
  scripts/
    00_extract_kexp_comments.py
    00b_segment_and_normalize_comments.py
    04_prelabel_for_prodigy.py
.gitignore
.repomixignore
data_to_spacy.sh
music_prodigy_annotate.md
prodigy_docs.md
requirements.txt
setup.py
setup.sh

================================================================
Files
================================================================

================
File: src/kexp_data.egg-info/dependency_links.txt
================


================
File: src/kexp_data.egg-info/PKG-INFO
================
Metadata-Version: 2.4
Name: kexp_data
Version: 0.1

================
File: src/kexp_data.egg-info/SOURCES.txt
================
README.md
setup.py
src/kexp_data.egg-info/PKG-INFO
src/kexp_data.egg-info/SOURCES.txt
src/kexp_data.egg-info/dependency_links.txt
src/kexp_data.egg-info/top_level.txt
src/kexp_processing/__init__.py
src/kexp_processing/normalization.py
src/scripts/00_extract_kexp_comments.py
src/scripts/04_prelabel_for_prodigy.py
src/scripts/__init__.py

================
File: src/kexp_data.egg-info/top_level.txt
================
kexp_processing
scripts

================
File: src/kexp_processing/normalization.py
================
import re


def normalize_text(text_content):
    """
    Normalizes text for consistent processing, matching, and output.
    - Converts to string.
    - Replaces various dash types (--, ---, em-dash, en-dash) with a single space.
    - Normalizes all other whitespace (spaces, tabs, newlines) to a single space.
    - Strips leading/trailing whitespace.
    """
    if text_content is None:
        return ""  # Return empty string for None input

    text_str = str(text_content)

    # Replace specific dash patterns first
    text_str = text_str.replace("---", " ")
    text_str = text_str.replace("--", " ")
    text_str = text_str.replace("—", " ")  # em-dash U+2014
    text_str = text_str.replace("–", " ")  # en-dash U+2013

    # Normalize all whitespace (including spaces from replaced dashes) to a single space
    text_str = re.sub(r'\s+', ' ', text_str)

    # Strip leading/trailing whitespace that might result from replacements or original text
    text_str = text_str.strip()

    return text_str


def normalize_text_for_gazetteer(text):
    """
    Normalizes text for gazetteer entries.
    - Converts to string.
    - Replaces various dash types and multiple hyphens with a single space.
    - Normalizes all whitespace to a single space.
    - Strips leading/trailing whitespace.
    """
    if text is None:
        return None
    text_str = str(text)

    # Replace various dash/hyphen constructs with a single space
    text_str = text_str.replace("---", " ")
    text_str = text_str.replace("--", " ")
    text_str = text_str.replace("—", " ")  # em-dash
    text_str = text_str.replace("–", " ")  # en-dash

    # Normalize internal whitespace (e.g., multiple spaces to one)
    text_str = re.sub(r'\s+', ' ', text_str)
    text_str = text_str.strip()  # Strip leading/trailing whitespace after normalization

    return text_str if text_str else None


def clean_term(term):
    """
    Cleans a term for gazetteer creation.
    - Normalizes text (dashes, whitespace).
    - Removes common "Various Artists" prefixes.
    - Strips defined leading/trailing punctuation.
    - Ensures term does not start or end with hyphen or apostrophe after initial cleaning.
    """
    # Step 1: Core normalization (handles dashes, multiple spaces, leading/trailing spaces)
    normalized_term = normalize_text_for_gazetteer(term)
    if not normalized_term:  # if term was None or became empty after normalization
        return None

    # Step 2: Remove common "Various Artists" prefixes
    if normalized_term.lower().startswith('(v/a)') or normalized_term.lower().startswith('[v/a]'):
        return None  # Filter out

    # Step 3: Strip leading/trailing non-alphanumeric, non-space, non-dash, non-apostrophe characters.
    cleaned_term = re.sub(r"^[^\w\s'-]+|[^\w\s'-]+$",
                          "", normalized_term).strip()

    # Step 4: After the above, ensure it doesn't start or end with a single hyphen or apostrophe
    if cleaned_term.startswith("'") or cleaned_term.endswith("'"):
        cleaned_term = cleaned_term.strip("'")
    if cleaned_term.startswith("-") or cleaned_term.endswith("-"):
        cleaned_term = cleaned_term.strip("-")

    # Final strip to catch any new leading/trailing spaces
    cleaned_term = cleaned_term.strip()

    return cleaned_term if cleaned_term else None


def is_well_formed(term, min_len=3, min_alpha_ratio=0.5, min_alphanum_ratio=0.7):
    """
    Checks if a cleaned term is well-formed for a gazetteer with stricter rules.
    - Minimum length.
    - Minimum ratio of alphabetic characters.
    - Minimum ratio of alphanumeric characters.
    - Not purely digits.
    - Not purely punctuation/symbols (even if it passes length).
    """
    if not term:
        return False

    term_len = len(term)
    if term_len < min_len:
        return False

    alpha_chars = sum(1 for char in term if char.isalpha())
    alphanum_chars = sum(1 for char in term if char.isalnum())

    if term_len > 0:  # Avoid division by zero
        if (alpha_chars / term_len) < min_alpha_ratio:
            return False
        if (alphanum_chars / term_len) < min_alphanum_ratio:
            return False
    else:
        return False

    if term.isdigit():
        return False

    if alphanum_chars == 0 and term_len > 0:  # Purely non-alphanumeric
        return False

    # Filter out terms that are just one or two quote characters or similar noise
    if term in ['"', "''", "'", " ", "-", "--", "---"]:
        return False

    return True

================
File: src/scripts/00_extract_kexp_comments.py
================
#!/usr/bin/env python3
import sqlite3
import json
import os
import argparse
import logging
from dotenv import load_dotenv
from kexp_processing.normalization import normalize_text
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_play_comments_with_details(db_path, limit=None, normalize=False, random=False):
    """
    Fetches plays with comments from the KEXP database and joins them with song
    (for song title, album, artist, release_date) and show (for host names) information.
    If normalize is True, normalizes the comment text using normalize_text.
    If random is True, selects comments in random order.

    Args:
        db_path (str): Path to the SQLite database file.
        limit (int, optional): Maximum number of records to fetch. Defaults to None (all).
        normalize (bool, optional): If True, normalize the comment text. Defaults to False.
        random (bool, optional): If True, select comments in random order. Defaults to False.

    Yields:
        dict: A dictionary representing a play's comment with its associated metadata.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # Schema based on user input:
        # plays.song_id -> songs.song_id
        # songs table contains: song (title), artist (name), album (title), release_date
        # plays.show -> shows.id (to get host_names)
        # Comment is directly on the plays table.
        query = """
        SELECT
            p.id AS play_id,
            p.comment AS comment_text,
            p.airdate AS play_airdate,
            p.song_id,
            s.song AS song_title,          -- From songs.song
            s.album AS album_title,        -- From songs.album
            s.artist AS artist_name,       -- From songs.artist
            s.release_date AS song_release_date, -- From songs.release_date
            sh.host_names AS dj_host_names -- Get host names from the shows table
        FROM
            plays p
        LEFT JOIN
            songs s ON p.song_id = s.song_id -- Corrected join field for songs
        LEFT JOIN
            shows sh ON p.show = sh.id      -- Join with shows table
        WHERE
            p.comment IS NOT NULL AND TRIM(p.comment) != ''
            AND LENGTH(TRIM(p.comment)) >= 20
        """

        if random:
            query += " ORDER BY RANDOM()"
        else:
            query += " ORDER BY p.airdate DESC"

        if limit:
            query += f" LIMIT {limit}"

        logging.info(f"Executing query (first 500 chars): {query[:500]}...")
        cursor.execute(query)

        count = 0
        for row in cursor:
            # Ensure comment_text (which is p.comment) is a string and not None
            comment_text = row['comment_text'] if row['comment_text'] is not None else ""
            output_text = normalize_text(
                comment_text) if normalize else comment_text

            record = {
                # This is the DJ's comment (normalized if requested)
                "text": output_text,
                "meta": {
                    "play_id": row['play_id'],
                    "dj_host_names": row['dj_host_names'],
                    "play_airdate": row['play_airdate'],
                    # Keep the original song_id from DB
                    "db_song_id": row['song_id'],
                    "db_song_title": row['song_title'],
                    "db_album_title": row['album_title'],
                    "db_artist_name": row['artist_name'],
                    "db_song_release_date": row['song_release_date']
                }
            }
            yield record
            count += 1
            if count % 1000 == 0:
                logging.info(f"Processed {count} play comments...")

        logging.info(
            f"Finished processing. Total play comments fetched: {count}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KEXP play comments with song/artist/show details to JSONL for Prodigy.")
    parser.add_argument(
        "output_file",
        help="Path to the output JSONL file (e.g., data/raw_kexp_data/kexp_play_comments_with_meta.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit the number of comments to process (for testing)."
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="If set, normalize the comment text using the normalization utility."
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, select comments in random order (up to the limit)."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting data extraction from: {db_path}")
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in fetch_play_comments_with_details(db_path, args.limit, normalize=args.normalize_text, random=args.random):
                outfile.write(json.dumps(record) + '\n')
        logging.info(
            f"Successfully wrote play comments to: {args.output_file}")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()

================
File: src/scripts/00b_segment_and_normalize_comments.py
================
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

================
File: src/scripts/04_prelabel_for_prodigy.py
================
#!/usr/bin/env python3
import spacy
from spacy.tokens import Doc
from spacy.util import filter_spans
import os
import argparse
import json
import logging
from dotenv import load_dotenv
from collections import defaultdict
import re
from src.kexp_processing.normalization import normalize_text

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
nlp_global = None

# Define label priorities
LABEL_PRIORITY = {
    "METADATA_ARTIST_NAME": 0,
    "METADATA_ALBUM_TITLE": 1,
    "METADATA_SONG_TITLE": 2
}


def get_label_priority(label):
    return LABEL_PRIORITY.get(label, 99)


def load_spacy_model(model_name="en_core_web_trf"):
    global nlp_global
    if nlp_global is None:
        try:
            nlp_global = spacy.load(model_name, disable=[
                "parser", "tagger", "ner", "lemmatizer"])
            logging.info(
                f"spaCy model '{model_name}' loaded (most components disabled for pre-labeling).")
        except OSError:
            logging.error(
                f"Model '{model_name}' not found. `python -m spacy download {model_name}`")
            raise
    return nlp_global


def find_exact_string_matches(normalized_doc_text, normalized_search_term, label_prefix, source_suffix):
    """
    Finds all exact, case-insensitive matches of a (normalized) search_term
    within the (normalized) doc_text.
    Args:
        normalized_doc_text (str): The pre-normalized text of the document/comment.
        normalized_search_term (str): The pre-normalized metadata term to search for.
        label_prefix (str): The base label (e.g., "ARTIST_NAME").
        source_suffix (str): Suffix for the source (e.g., "artist").
    Returns:
        list: List of match dictionaries.
    """
    matches = []
    pattern_to_search = None  # Initialize to avoid unbound reference
    # Check if search term or text is empty after normalization
    if not normalized_search_term or not normalized_doc_text:
        return matches

    try:
        # The normalized_search_term already has its internal spaces normalized.
        # We need to escape it for regex and then replace its internal single spaces
        # with \s+ to allow for flexible whitespace matching in the doc_text.
        # Example: "Artist Name" -> "Artist\s+Name"

        # Escape the whole term first to handle any special regex characters within words
        escaped_term = re.escape(normalized_search_term)

        # If normalized_search_term was "word1 word2", escaped_term is "word1\ word2".
        # We want "word1\s+word2". So, split the *original* normalized_search_term by space,
        # escape each part, then join with \s+.
        # Split by single space (it's normalized)
        words = normalized_search_term.split(' ')
        # Filter out empty strings if any
        escaped_words = [re.escape(word) for word in words if word]

        if not escaped_words:  # Should not happen if normalized_search_term is not empty
            return matches

        # This pattern will match the sequence of words from normalized_search_term,
        # allowing one or more whitespace characters between them in normalized_doc_text.
        flexible_space_pattern = r'\s+'.join(escaped_words)

        # Add word boundaries to ensure we match whole words/phrases.
        # (?<!\w) - negative lookbehind for a word character
        # (?!\w) - negative lookahead for a word character
        # This helps prevent matching "art" in "artist" if search term is "art".
        # However, if your search term can be part of a larger compound, this might be too strict.
        # For names and titles, it's often safer.
        # Consider if your terms can have leading/trailing non-word chars, e.g. "A.B.C."
        # re.escape handles the '.', so word boundaries should still work.
        # For now, let's use word boundaries.
        # If a term is "A-Go-Go", pattern becomes "A\-Go\-Go". Word boundaries work.
        # If a term is "!Action", pattern is "\!Action". (?<!\w)\!Action(?!\w) - might be fine.

        # More robust word boundary handling for terms that might start/end with non-alphanum:
        # If first/last char of term is non-alphanum, don't require word boundary there.

        prefix_boundary = r'(?<!\w)' if normalized_search_term[0].isalnum(
        ) else ''
        suffix_boundary = r'(?!\w)' if normalized_search_term[-1].isalnum(
        ) else ''

        pattern_to_search = prefix_boundary + flexible_space_pattern + suffix_boundary

        for match in re.finditer(pattern_to_search, normalized_doc_text, re.IGNORECASE):
            matches.append({
                "start_char": match.start(),
                "end_char": match.end(),
                # The actual matched text from the comment
                "text": match.group(0),
                "label": f"METADATA_{label_prefix}",
                "source": f"metadata_{source_suffix}"
            })
    except re.error as e:
        if pattern_to_search is not None:
            logging.warning(
                f"Regex error matching '{normalized_search_term}' with pattern '{pattern_to_search}' in '{normalized_doc_text[:50]}...': {e}")
        else:
            logging.warning(
                f"Regex error matching '{normalized_search_term}' (pattern was not constructed) in '{normalized_doc_text[:50]}...': {e}")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Pre-labels KEXP comments with ARTIST_NAME, SONG_TITLE, ALBUM_TITLE using ONLY comment metadata, with text normalization."
    )
    parser.add_argument("input_jsonl_file",
                        help="Path to input JSONL (from script 00)")
    parser.add_argument("output_jsonl_file",
                        help="Path to output pre-labeled JSONL for Prodigy")
    parser.add_argument(
        "--spacy_model", default="en_core_web_trf", help="spaCy model.")
    parser.add_argument("--limit", type=int, help="Limit number of comments.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input_jsonl_file):
        logging.error(f"Input JSONL file not found: {args.input_jsonl_file}")
        return

    try:
        nlp = load_spacy_model(args.spacy_model)
        processed_count = 0

        with open(args.input_jsonl_file, 'r', encoding='utf-8') as infile, \
                open(args.output_jsonl_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                if args.limit and i >= args.limit:
                    logging.info(
                        f"Reached processing limit of {args.limit} records.")
                    break

                original_comment_text_for_error_logging = "N/A"
                try:
                    record = json.loads(line)
                    original_comment_text = record.get("text")
                    meta = record.get("meta", {})
                    original_comment_text_for_error_logging = original_comment_text if original_comment_text else "N/A"

                    if not original_comment_text:
                        logging.warning(
                            f"Skipping line {i+1} due to missing or empty 'text' field: {line.strip()}")
                        continue

                    # --- NORMALIZATION STEP 1: Normalize the input comment text ---
                    normalized_comment_text = normalize_text(
                        original_comment_text)

                    if not normalized_comment_text:  # If text became empty after normalization
                        logging.info(
                            f"Skipping line {i+1} as text became empty after normalization. Original: '{original_comment_text[:100]}...'")
                        continue

                    # --- Create spaCy Doc from the NORMALIZED text ---
                    doc = nlp(normalized_comment_text)
                    candidate_matches_info = []

                    # --- Metadata-based matches using normalized text and normalized metadata terms ---
                    metadata_fields_to_check = {
                        "db_artist_name": ("ARTIST_NAME", "artist"),
                        "db_song_title": ("SONG_TITLE", "song"),
                        "db_album_title": ("ALBUM_TITLE", "album")
                    }

                    for meta_key, (label_prefix, source_suffix) in metadata_fields_to_check.items():
                        raw_meta_value = meta.get(meta_key)
                        if raw_meta_value:
                            # --- NORMALIZATION STEP 2: Normalize the metadata term ---
                            normalized_meta_term = normalize_text(
                                raw_meta_value)
                            if normalized_meta_term:  # Ensure not empty after normalization
                                candidate_matches_info.extend(find_exact_string_matches(
                                    doc.text,  # This is normalized_comment_text
                                    normalized_meta_term,
                                    label_prefix,
                                    source_suffix
                                ))

                    # --- Resolve conflicts for identical spans by priority ---
                    spans_by_offset = defaultdict(list)
                    for match_info in candidate_matches_info:
                        spans_by_offset[(match_info["start_char"], match_info["end_char"])].append(
                            match_info)

                    unique_spans_for_filtering = []
                    for (start_char, end_char), match_group in spans_by_offset.items():
                        if not match_group:
                            continue
                        match_group.sort(key=lambda m: (
                            get_label_priority(m["label"]), m["label"]))
                        best_match = match_group[0]

                        # Create span object using the *normalized* doc
                        span_obj = doc.char_span(
                            best_match["start_char"], best_match["end_char"], label=best_match["label"])
                        if span_obj:
                            unique_spans_for_filtering.append(span_obj)
                        else:
                            logging.debug(
                                f"Could not create span for: {best_match} in doc (len {len(doc.text)}): '{doc.text[:70]}...' "
                                f"(start: {best_match['start_char']}, end: {best_match['end_char']}). "
                                f"Original text for this span was: '{normalized_comment_text[best_match['start_char']:best_match['end_char']]}'"
                            )

                    # --- Filter overlapping span boundaries using filter_spans ---
                    final_spans_obj = filter_spans(unique_spans_for_filtering)

                    # --- Convert final spaCy Span objects to Prodigy format ---
                    spans_for_prodigy = []
                    for span in final_spans_obj:
                        final_label = span.label_
                        if final_label.startswith("METADATA_"):
                            final_label = final_label.replace("METADATA_", "")

                        spans_for_prodigy.append({
                            "start": span.start_char,
                            "end": span.end_char,
                            "label": final_label,
                            "text": span.text,  # Text of the span from the normalized doc
                            "source": "metadata_match_v4_normalized_word_boundaries"  # Updated source
                        })
                    spans_for_prodigy.sort(key=lambda x: x["start"])

                    # --- NORMALIZATION STEP 3: Output the NORMALIZED text to Prodigy ---
                    output_record = {
                        "text": normalized_comment_text,  # Use the normalized text for Prodigy
                        # Keep original text in meta if needed
                        "meta": {**meta, "original_comment_text": original_comment_text},
                        "spans": spans_for_prodigy
                    }
                    outfile.write(json.dumps(
                        output_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logging.info(
                            f"Pre-labeled {processed_count} comments...")

                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON line {i+1}: {line.strip()}")
                except Exception as e:
                    logging.error(
                        f"Error processing line {i+1} (original text: '{original_comment_text_for_error_logging[:100]}...'): {e}", exc_info=True)

        logging.info(
            f"Successfully pre-labeled {processed_count} comments to: {args.output_jsonl_file}")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: .gitignore
================
.vscode
.env
.venv
__pycache__
*.pyc
*.pyo
*.pyd
*.pyw
*.pyz
*.pywz
/data/raw_kexp_data

================
File: .repomixignore
================
data/
models/

================
File: data_to_spacy.sh
================
prodigy data-to-spacy ./data/spacy_training_data_music_only/ \
    --ner kexp_ner_annotations_v2 \
    --eval-split 0.2 \
    -l ARTIST_NAME,SONG_TITLE,ALBUM_TITLE,VENUE_NAME,LOCATION,DATE_EXPR

================
File: music_prodigy_annotate.md
================
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

Additional Info:
----------------

================================================================
Directory Structure
================================================================
scripts/
  00_extract_kexp_comments.py
  01_extract_db_terms_for_gazetteers.py
  02_create_and_test_matchers.py
  04_prelabel_for_prodigy.py
.gitignore
.repomixignore
requirements.txt
setup.sh

================================================================
Files
================================================================

================
File: scripts/00_extract_kexp_comments.py
================
#!/usr/bin/env python3
import sqlite3
import json
import os
import argparse
import logging
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_play_comments_with_details(db_path, limit=None):
    """
    Fetches plays with comments from the KEXP database and joins them with song
    (for song title, album, artist, release_date) and show (for host names) information.

    Args:
        db_path (str): Path to the SQLite database file.
        limit (int, optional): Maximum number of records to fetch. Defaults to None (all).

    Yields:
        dict: A dictionary representing a play's comment with its associated metadata.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # Schema based on user input:
        # plays.song_id -> songs.song_id
        # songs table contains: song (title), artist (name), album (title), release_date
        # plays.show -> shows.id (to get host_names)
        # Comment is directly on the plays table.
        query = """
        SELECT
            p.id AS play_id,
            p.comment AS comment_text,
            p.airdate AS play_airdate,
            p.song_id,
            s.song AS song_title,          -- From songs.song
            s.album AS album_title,        -- From songs.album
            s.artist AS artist_name,       -- From songs.artist
            s.release_date AS song_release_date, -- From songs.release_date
            sh.host_names AS dj_host_names -- Get host names from the shows table
        FROM
            plays p
        LEFT JOIN
            songs s ON p.song_id = s.song_id -- Corrected join field for songs
        LEFT JOIN
            shows sh ON p.show = sh.id      -- Join with shows table
        WHERE
            p.comment IS NOT NULL AND TRIM(p.comment) != ''
        ORDER BY
            p.airdate DESC -- Or p.id, or any other preferred order
        """

        if limit:
            query += f" LIMIT {limit}"

        logging.info(f"Executing query (first 500 chars): {query[:500]}...")
        cursor.execute(query)

        count = 0
        for row in cursor:
            # Ensure comment_text (which is p.comment) is a string and not None
            comment_text = row['comment_text'] if row['comment_text'] is not None else ""

            record = {
                "text": comment_text,  # This is the DJ's comment
                "meta": {
                    "play_id": row['play_id'],
                    "dj_host_names": row['dj_host_names'],
                    "play_airdate": row['play_airdate'],
                    # Keep the original song_id from DB
                    "db_song_id": row['song_id'],
                    "db_song_title": row['song_title'],
                    "db_album_title": row['album_title'],
                    "db_artist_name": row['artist_name'],
                    "db_song_release_date": row['song_release_date']
                }
            }
            yield record
            count += 1
            if count % 1000 == 0:
                logging.info(f"Processed {count} play comments...")

        logging.info(
            f"Finished processing. Total play comments fetched: {count}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KEXP play comments with song/artist/show details to JSONL for Prodigy.")
    parser.add_argument(
        "output_file",
        help="Path to the output JSONL file (e.g., data/raw_kexp_data/kexp_play_comments_with_meta.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit the number of comments to process (for testing)."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting data extraction from: {db_path}")
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in fetch_play_comments_with_details(db_path, args.limit):
                outfile.write(json.dumps(record) + '\n')
        logging.info(
            f"Successfully wrote play comments to: {args.output_file}")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()

================
File: scripts/01_extract_db_terms_for_gazetteers.py
================
#!/usr/bin/env python3
import sqlite3
import os
import argparse
import logging
import pandas as pd
from dotenv import load_dotenv
import re  # For more advanced cleaning

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def clean_term(term):
    """
    Cleans a term for gazetteer creation.
    - Converts to string and strips leading/trailing whitespace.
    - Removes common "Various Artists" prefixes.
    - Normalizes internal whitespace.
    - Strips defined leading/trailing punctuation.
    - Ensures term does not start or end with hyphen or apostrophe after initial cleaning.
    """
    if term is None:
        return None

    term_str = str(term).strip()

    if not term_str:
        return None

    # Remove common "Various Artists" prefixes
    if term_str.lower().startswith('(v/a)') or term_str.lower().startswith('[v/a]'):
        return None  # Filter out

    # Normalize internal whitespace (e.g., multiple spaces to one)
    term_str = re.sub(r'\s+', ' ', term_str)

    # Strip leading/trailing non-alphanumeric, non-space, non-dash, non-apostrophe characters
    # This preserves internal hyphens and apostrophes.
    term_str = re.sub(r"^[^\w\s'-]+|[^\w\s'-]+$", "", term_str).strip()

    # After the above, ensure it doesn't start or end with a hyphen or apostrophe
    if term_str.startswith("'") or term_str.endswith("'"):
        term_str = term_str.strip("'")
    if term_str.startswith("-") or term_str.endswith("-"):
        term_str = term_str.strip("-")

    term_str = term_str.strip()  # Final strip

    if not term_str:
        return None

    return term_str


def is_well_formed(term, min_len=3, min_alpha_ratio=0.5, min_alphanum_ratio=0.7):
    """
    Checks if a cleaned term is well-formed for a gazetteer with stricter rules.
    - Minimum length.
    - Minimum ratio of alphabetic characters.
    - Minimum ratio of alphanumeric characters.
    - Not purely digits.
    - Not purely punctuation/symbols (even if it passes length).
    """
    if not term:
        return False

    term_len = len(term)
    if term_len < min_len:
        return False

    alpha_chars = sum(1 for char in term if char.isalpha())
    alphanum_chars = sum(1 for char in term if char.isalnum())

    if term_len > 0:  # Avoid division by zero
        if (alpha_chars / term_len) < min_alpha_ratio:
            return False
        if (alphanum_chars / term_len) < min_alphanum_ratio:
            return False
    else:  # Should have been caught by min_len or previous checks
        return False

    # Check if it's purely digits
    if term.isdigit():
        return False

    # Check if (after cleaning) it's now purely non-alphanumeric (e.g. "!!", "--")
    # This is a bit redundant given the ratio checks but can catch edge cases.
    if alphanum_chars == 0 and term_len > 0:
        return False

    # Filter out terms that are just one or two quote characters or similar noise that might have slipped through
    if term in ['"', "''", "'", " ", "-", "--", "---"]:  # Added hyphens here
        return False

    return True


def extract_terms(db_path, table_name, column_name, output_file, term_type="generic"):
    """
    Extracts unique, non-empty, cleaned, and well-formed terms.
    Applies more conservative filtering.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT DISTINCT \"{column_name}\" FROM \"{table_name}\" WHERE \"{column_name}\" IS NOT NULL AND TRIM(\"{column_name}\") != '';"
        logging.info(
            f"Executing query for {table_name}.{column_name}: {query}")
        df = pd.read_sql_query(query, conn)

        if df.empty or column_name not in df.columns:
            logging.warning(
                f"No data found or column '{column_name}' not in table '{table_name}'. Skipping {output_file}.")
            return

        cleaned_terms = []
        original_count = len(df)
        cleaned_count = 0
        well_formed_count = 0

        for term_val in df[column_name].astype(str):
            cleaned = clean_term(term_val)
            if cleaned:
                cleaned_count += 1
                # Adjust min_len for specific types if necessary, e.g., allow shorter artist names
                current_min_len = 2
                if term_type == "artist":
                    # Potentially allow very short artist names if they are common (e.g., "M", "X")
                    # For now, let's stick to a slightly more general rule for artists.
                    # If a name is just one letter, it must be an alpha char.
                    if len(cleaned) == 1 and cleaned.isalpha():
                        current_min_len = 1
                    else:
                        current_min_len = 2  # Default for multi-char artists
                elif term_type in ["song_title", "album_title"]:
                    current_min_len = 2  # Titles should generally be longer than 1 char

                if is_well_formed(cleaned, min_len=current_min_len):
                    cleaned_terms.append(cleaned)
                    well_formed_count += 1

        unique_terms = sorted(list(set(cleaned_terms)))

        logging.info(f"For {table_name}.{column_name}:")
        logging.info(
            f"  Original distinct terms from DB (non-null, non-empty): {original_count}")
        logging.info(
            f"  Terms remaining after basic cleaning (strip, (V/A), normalize space): {cleaned_count}")
        logging.info(
            f"  Terms remaining after well-formedness checks: {well_formed_count}")
        logging.info(
            f"  Unique well-formed terms to be written: {len(unique_terms)}")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for term_item in unique_terms:
                f.write(term_item + '\n')
        logging.info(
            f"Successfully wrote {len(unique_terms)} unique, cleaned terms from {table_name}.{column_name} to {output_file}")

    except sqlite3.Error as e:
        logging.error(f"Database error for {table_name}.{column_name}: {e}")
    except Exception as e:
        logging.error(
            f"An unexpected error occurred for {table_name}.{column_name}: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clean terms from KEXP database for gazetteer files.")
    parser.add_argument(
        "--output_dir",
        default="data/gazetteers",
        help="Directory to save the gazetteer files (default: data/gazetteers)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting gazetteer extraction from: {db_path}")

        # Apply stricter rules for artists, songs, and albums
        extract_terms(db_path, "songs", "artist", os.path.join(
            args.output_dir, "artists.txt"), term_type="artist")
        extract_terms(db_path, "songs", "song", os.path.join(
            args.output_dir, "songs.txt"), term_type="song_title")
        extract_terms(db_path, "songs", "album", os.path.join(
            args.output_dir, "albums.txt"), term_type="album_title")

        logging.info("Gazetteer extraction process complete.")

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during gazetteer generation: {e}")


if __name__ == "__main__":
    main()

================
File: scripts/02_create_and_test_matchers.py
================
#!/usr/bin/env python3
import spacy
from spacy.matcher import PhraseMatcher
# Matcher removed as we are not doing rule-based date patterns here for pre-labeling
import os
import argparse
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
NLP = None


def load_spacy_model(model_name="en_core_web_trf"):
    """Loads a spaCy model."""
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load(model_name)
            logging.info(f"spaCy model '{model_name}' loaded successfully.")
        except OSError:
            logging.error(
                f"spaCy model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
            raise
    return NLP


def load_gazetteer(filepath):
    """Loads terms from a text file, one term per line."""
    if not os.path.exists(filepath):
        logging.warning(
            f"Gazetteer file not found: {filepath}. Returning empty list.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(terms)} terms from {filepath}")
    return terms


def create_phrase_matcher(nlp_model, gazetteers_map):
    """
    Creates a spaCy PhraseMatcher from a dictionary of gazetteers.

    Args:
        nlp_model: The loaded spaCy model.
        gazetteers_map (dict): A dictionary where keys are entity labels (e.g., "ARTIST_NAME")
                               and values are lists of terms for that entity.

    Returns:
        spacy.matcher.PhraseMatcher: The configured PhraseMatcher.
    """
    matcher = PhraseMatcher(
        nlp_model.vocab, attr="LOWER")  # Match on lowercased text
    for label, terms in gazetteers_map.items():
        if terms:
            patterns = [nlp_model.make_doc(term) for term in terms]
            matcher.add(label, patterns)
            logging.info(
                f"Added {len(patterns)} patterns for label '{label}' to PhraseMatcher.")
        else:
            logging.warning(
                f"No terms provided for label '{label}'. Skipping PhraseMatcher addition.")
    return matcher

# create_rule_matcher function has been removed as per user request


def main():
    parser = argparse.ArgumentParser(
        description="Create and test PhraseMatcher patterns from core gazetteers (artist, song, album).")
    parser.add_argument(
        "--gazetteer_dir",
        default="data/gazetteers",
        help="Directory containing the gazetteer text files."
    )
    parser.add_argument(
        "--sample_text_file",
        help="Optional: Path to a JSONL file with texts to test the matcher on (expects 'text' key from script 00)."
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default='Death Cab For Cutie will play the Showbox on Friday, November 22nd. Their new album is "Plans", released last month.',
        help="A sample text string to test the matcher if no file is provided."
    )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_trf",
        help="Name of the spaCy model to use."
    )

    args = parser.parse_args()

    try:
        nlp = load_spacy_model(args.spacy_model)

        # Define which gazetteers to load for PhraseMatcher
        # Focused on Artist, Song, Album as per user request for pre-labeling
        gazetteer_files_for_phrase_matcher = {
            "ARTIST_NAME": os.path.join(args.gazetteer_dir, "artists.txt"),
            "SONG_TITLE": os.path.join(args.gazetteer_dir, "songs.txt"),
            "ALBUM_TITLE": os.path.join(args.gazetteer_dir, "albums.txt"),
            # VENUE_NAME and EVENT_NAME (formerly FESTIVAL_NAME) gazetteers will be manually annotated for NER training,
            # not used for pre-labeling by this script.
        }

        gazetteers_data = {}
        for label, filepath in gazetteer_files_for_phrase_matcher.items():
            gazetteers_data[label] = load_gazetteer(filepath)

        # Create the PhraseMatcher
        phrase_matcher = create_phrase_matcher(nlp, gazetteers_data)

        # Rule-based Matcher for dates is removed from this script's pre-labeling focus.
        # Date expressions will be manually annotated or handled by spaCy's default DATE NER.

        # Test the PhraseMatcher
        texts_to_process = []
        if args.sample_text_file:
            if not os.path.exists(args.sample_text_file):
                logging.error(
                    f"Sample text file not found: {args.sample_text_file}")
                return
            with open(args.sample_text_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        texts_to_process.append(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(
                            f"Skipping malformed line {i+1} in sample text file: {line.strip()} - Error: {e}")
            logging.info(
                f"Loaded {len(texts_to_process)} texts from {args.sample_text_file} for testing.")
        else:
            texts_to_process.append(args.sample_text)
            logging.info(f"Using provided sample text for testing.")

        # Test on first 5 samples if from file, or the single sample_text
        for text_to_test in texts_to_process[:5]:
            logging.info(f"\n--- Testing on text: --- \n'{text_to_test}'")
            doc = nlp(text_to_test)

            phrase_matches_output = []

            # PhraseMatcher results
            phrase_matches = phrase_matcher(doc)
            logging.info(f"PhraseMatcher found {len(phrase_matches)} matches:")
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                label = nlp.vocab.strings[match_id]
                logging.info(
                    f"  Label: {label}, Span: '{span.text}', Start: {span.start_char}, End: {span.end_char}")
                phrase_matches_output.append(
                    {"start": span.start_char, "end": span.end_char, "label": label})

            if phrase_matches_output:
                # Sort by start character to make it easier to read / process
                phrase_matches_sorted = sorted(
                    phrase_matches_output, key=lambda x: x["start"])
                prodigy_like_output = {
                    "text": text_to_test, "spans": phrase_matches_sorted}
                logging.info(
                    f"Prodigy-like output for this text (from PhraseMatcher only): {json.dumps(prodigy_like_output, indent=2)}")
            elif not phrase_matches:
                logging.info("  No matches found by PhraseMatcher.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: scripts/04_prelabel_for_prodigy.py
================
#!/usr/bin/env python3
import spacy
from spacy.tokens import Doc  # Import Doc for creating spaCy documents
from spacy.util import filter_spans
import os
import argparse
import json
import logging
from dotenv import load_dotenv
# Still useful if a metadata term matches multiple times
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
NLP = None

# Define label priorities - simplified as we are only dealing with metadata sources now
# but keeping structure in case of future expansion or if a term is in multiple meta fields.
LABEL_PRIORITY = {
    "METADATA_ARTIST_NAME": 0,  # Highest priority
    "METADATA_ALBUM_TITLE": 1,
    "METADATA_SONG_TITLE": 2
    # No general gazetteer labels needed here for priority
}


def get_label_priority(label):
    return LABEL_PRIORITY.get(label, 99)


def load_spacy_model(model_name="en_core_web_trf"):
    global NLP
    if NLP is None:
        try:
            # We only need tokenization for character offsets and creating Doc objects.
            # Disabling other components for speed.
            NLP = spacy.load(model_name, disable=[
                             "parser", "tagger", "ner", "lemmatizer"])
            logging.info(
                f"spaCy model '{model_name}' loaded (most components disabled for pre-labeling).")
        except OSError:
            logging.error(
                f"Model '{model_name}' not found. `python -m spacy download {model_name}`")
            raise
    return NLP


def find_exact_string_matches(text_content, search_string, label_prefix, source_suffix):
    """
    Finds all exact, case-insensitive matches of search_string in text_content,
    being robust to internal whitespace variations in text_content.
    Args:
        text_content (str): The text to search within.
        search_string (str): The string to search for (from metadata).
        label_prefix (str): The base label (e.g., "ARTIST_NAME").
        source_suffix (str): Suffix for the source (e.g., "artist").
    Returns:
        list: List of match dictionaries.
    """
    matches = []
    if not search_string or not text_content:  # Ensure search_string is not None or empty
        return matches

    # Ensure it's a string and strip it
    search_string_str = str(search_string).strip()
    if not search_string_str:  # If stripping makes it empty
        return matches

    try:
        # Create a regex pattern that allows for flexible whitespace
        # between words in the search string.
        # Split metadata term by any whitespace
        words = re.split(r'\s+', search_string_str)
        # Escape each part, ignore empty parts
        escaped_words = [re.escape(word) for word in words if word]

        if not escaped_words:  # If search string was only whitespace
            return matches

        # Join with \s+ to match one or more whitespace characters between words in the target text
        flexible_space_pattern = r'\s+'.join(escaped_words)

        # Use word boundaries (\b) if you want to match whole words only,
        # but this can be tricky if search_string itself starts/ends with non-word chars.
        # For titles/names, sometimes it's better without \b for partial but correct phrase matches.
        # Example: If search_string is "Live", \bLive\b won't match "Live at KEXP".
        # For now, let's try without strict word boundaries on the overall pattern for flexibility.
        # If you find it's too greedy, you can add them:
        # flexible_space_pattern = r'\b' + flexible_space_pattern + r'\b'
        # However, re.escape already handles individual words, so this should be fairly specific.

        for match in re.finditer(flexible_space_pattern, text_content, re.IGNORECASE):
            matches.append({
                "start_char": match.start(),
                "end_char": match.end(),
                # The actual matched text from the comment (which might have varied whitespace)
                "text": match.group(0),
                # Temporary label for prioritization
                "label": f"METADATA_{label_prefix}",
                "source": f"metadata_{source_suffix}"
            })
    except re.error as e:
        logging.warning(
            f"Regex error matching '{search_string_str}' in '{text_content[:50]}...': {e}")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Pre-labels KEXP comments with ARTIST_NAME, SONG_TITLE, ALBUM_TITLE using ONLY comment metadata."
    )
    parser.add_argument("input_jsonl_file",
                        help="Path to input JSONL (from script 00)")
    parser.add_argument("output_jsonl_file",
                        help="Path to output pre-labeled JSONL for Prodigy")
    parser.add_argument(
        "--spacy_model", default="en_core_web_trf", help="spaCy model.")
    parser.add_argument("--limit", type=int, help="Limit number of comments.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input_jsonl_file):
        logging.error(f"Input JSONL file not found: {args.input_jsonl_file}")
        return

    try:
        nlp = load_spacy_model(args.spacy_model)

        processed_count = 0
        with open(args.input_jsonl_file, 'r', encoding='utf-8') as infile, \
                open(args.output_jsonl_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                if args.limit and i >= args.limit:
                    logging.info(
                        f"Reached processing limit of {args.limit} records.")
                    break
                try:
                    record = json.loads(line)
                    text_to_annotate = record.get("text")
                    meta = record.get("meta", {})
                    if not text_to_annotate:
                        logging.warning(
                            f"Skipping line {i+1} due to missing or empty 'text' field: {line.strip()}")
                        continue

                    doc = nlp(text_to_annotate)

                    candidate_matches_info = []

                    # 1. Metadata-based matches ONLY
                    if meta.get("db_artist_name"):
                        candidate_matches_info.extend(find_exact_string_matches(
                            doc.text, meta["db_artist_name"], "ARTIST_NAME", "artist"
                        ))
                    if meta.get("db_song_title"):
                        candidate_matches_info.extend(find_exact_string_matches(
                            doc.text, meta["db_song_title"], "SONG_TITLE", "song"
                        ))
                    if meta.get("db_album_title"):
                        candidate_matches_info.extend(find_exact_string_matches(
                            doc.text, meta["db_album_title"], "ALBUM_TITLE", "album"
                        ))

                    # 2. Resolve conflicts for *identical spans* first by priority
                    spans_by_offset = defaultdict(list)
                    for match_info in candidate_matches_info:
                        spans_by_offset[(match_info["start_char"], match_info["end_char"])].append(
                            match_info)

                    unique_spans_for_filtering = []
                    for (start_char, end_char), match_group in spans_by_offset.items():
                        if not match_group:
                            continue
                        match_group.sort(key=lambda m: (
                            get_label_priority(m["label"]), m["label"]))
                        best_match = match_group[0]

                        span_obj = doc.char_span(
                            best_match["start_char"], best_match["end_char"], label=best_match["label"])
                        if span_obj:
                            unique_spans_for_filtering.append(span_obj)
                        else:
                            logging.debug(
                                f"Could not create span for: {best_match} in doc '{doc.text[:50]}...' (start: {best_match['start_char']}, end: {best_match['end_char']})")

                    # 3. Filter overlapping span boundaries using filter_spans
                    final_spans_obj = filter_spans(unique_spans_for_filtering)

                    # 4. Convert final spaCy Span objects to Prodigy format
                    spans_for_prodigy = []
                    for span in final_spans_obj:
                        final_label = span.label_
                        if final_label.startswith("METADATA_"):
                            final_label = final_label.replace("METADATA_", "")

                        spans_for_prodigy.append({
                            "start": span.start_char,
                            "end": span.end_char,
                            "label": final_label,
                            "text": span.text,
                            "source": "metadata_match_v2"  # Updated source
                        })

                    spans_for_prodigy.sort(key=lambda x: x["start"])

                    output_record = {
                        "text": text_to_annotate,
                        "meta": meta,
                        "spans": spans_for_prodigy
                    }
                    outfile.write(json.dumps(
                        output_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logging.info(
                            f"Pre-labeled {processed_count} comments...")

                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON line {i+1}: {line.strip()}")
                except Exception as e:
                    logging.error(
                        f"Error processing line {i+1}: '{text_to_annotate[:100] if text_to_annotate else 'N/A'}...' - {e}", exc_info=True)

        logging.info(
            f"Successfully pre-labeled {processed_count} comments to: {args.output_jsonl_file}")

    except Exception as e:
        logging.error(f"An critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: .gitignore
================
.vscode
.env
.venv
__pycache__
*.pyc
*.pyo
*.pyd
*.pyw
*.pyz
*.pywz
/data/raw_kexp_data

================
File: .repomixignore
================
data/

================
File: requirements.txt
================
aiofiles==24.1.0
annotated-types==0.7.0
anyio==4.9.0
blis==0.7.11
cachetools==5.5.2
catalogue==2.0.10
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.1.8
cloudpathlib==0.21.1
confection==0.1.5
curated-tokenizers==0.0.9
curated-transformers==0.1.1
cymem==2.0.11
en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl#sha256=272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5
fastapi==0.110.3
filelock==3.18.0
fsspec==2025.3.2
h11==0.16.0
idna==3.10
Jinja2==3.1.6
langcodes==3.5.0
language_data==1.3.0
marisa-trie==1.2.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
murmurhash==1.0.12
networkx==3.4.2
numpy==1.26.4
packaging==25.0
peewee==3.16.3
preshed==3.0.9
prodigy @ file:///Users/pooks/Dev/kexp_data/prodigy-1.17.5-py3-none-any.whl#sha256=a4c4bf09295ac418956c4f28f2d22c47c4d1708b99874bb9af8d917b0b56ebc2
pydantic==2.11.4
pydantic_core==2.33.2
Pygments==2.19.1
PyJWT==2.10.1
python-dotenv==1.1.0
radicli==0.0.25
regex==2024.11.6
requests==2.32.3
rich==14.0.0
setuptools==80.7.1
shellingham==1.5.4
smart-open==7.1.0
sniffio==1.3.1
spacy==3.7.5
spacy-curated-transformers==0.3.0
spacy-legacy==3.0.12
spacy-llm==0.7.3
spacy-loggers==1.0.5
srsly==2.5.1
starlette==0.37.2
sympy==1.14.0
thinc==8.2.5
thinc_apple_ops==1.0.0
toolz==0.12.1
torch==2.7.0
tqdm==4.67.1
typeguard==3.0.2
typer==0.15.4
typing-inspection==0.4.0
typing_extensions==4.13.2
urllib3==2.4.0
uvicorn==0.34.2
wasabi==1.1.3
weasel==0.4.1
wrapt==1.17.2

================
File: setup.sh
================
# Create the main project directory
mkdir kexp_knowledge_base
cd kexp_knowledge_base

# Create subdirectories
mkdir -p data/raw_kexp_data
mkdir -p data/gazetteers
mkdir -p data/prodigy_exports
mkdir -p data/spacy_training_data
mkdir models
mkdir scripts
mkdir notebooks
mkdir -p src/kexp_processing
mkdir configs

# Create some initial files (you'll populate these later)
touch data/raw_kexp_data/.gitkeep  # .gitkeep makes empty dirs trackable by git
touch data/gazetteers/artists.txt
touch data/gazetteers/songs.txt
touch data/gazetteers/albums.txt
touch data/gazetteers/venues.txt
touch data/gazetteers/festivals.txt
touch data/prodigy_exports/.gitkeep
touch data/spacy_training_data/.gitkeep
touch models/.gitkeep
touch scripts/01_extract_db_terms.py
touch scripts/02_compile_gazetteers.py
touch scripts/03_create_matchers.py
touch scripts/04_prelabel_for_prodigy.py
touch scripts/05_train_spacy_model.py
touch scripts/06_extract_relations.py
touch notebooks/data_exploration.ipynb
touch src/__init__.py
touch src/kexp_processing/__init__.py
touch src/kexp_processing/entity_extractors.py
touch configs/ner_config.cfg
touch .gitignore
touch README.md
touch requirements.txt

echo "Project directory structure for 'kexp_knowledge_base' created."
echo "Next steps:"
echo "1. cd kexp_knowledge_base"
echo "2. python3 -m venv venv  # Create a virtual environment"
echo "3. source venv/bin/activate # Activate it (on Linux/macOS)"
echo "4. pip install -r requirements.txt # (After you add dependencies to requirements.txt)"
echo "5. git init && git add . && git commit -m 'Initial project structure' # Initialize Git"



================================================================
End of Codebase
================================================================

================
File: prodigy_docs.md
================
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

Additional Info:
----------------

================================================================
Directory Structure
================================================================
scripts/
  00_extract_kexp_comments.py
  01_extract_db_terms_for_gazetteers.py
  02_create_and_test_matchers.py
  04_prelabel_for_prodigy.py
.gitignore
.repomixignore
data_to_spacy.sh
music_prodigy_annotate.md
requirements.txt
setup.sh

================================================================
Files
================================================================

================
File: scripts/00_extract_kexp_comments.py
================
#!/usr/bin/env python3
import sqlite3
import json
import os
import argparse
import logging
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_play_comments_with_details(db_path, limit=None):
    """
    Fetches plays with comments from the KEXP database and joins them with song
    (for song title, album, artist, release_date) and show (for host names) information.

    Args:
        db_path (str): Path to the SQLite database file.
        limit (int, optional): Maximum number of records to fetch. Defaults to None (all).

    Yields:
        dict: A dictionary representing a play's comment with its associated metadata.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # Schema based on user input:
        # plays.song_id -> songs.song_id
        # songs table contains: song (title), artist (name), album (title), release_date
        # plays.show -> shows.id (to get host_names)
        # Comment is directly on the plays table.
        query = """
        SELECT
            p.id AS play_id,
            p.comment AS comment_text,
            p.airdate AS play_airdate,
            p.song_id,
            s.song AS song_title,          -- From songs.song
            s.album AS album_title,        -- From songs.album
            s.artist AS artist_name,       -- From songs.artist
            s.release_date AS song_release_date, -- From songs.release_date
            sh.host_names AS dj_host_names -- Get host names from the shows table
        FROM
            plays p
        LEFT JOIN
            songs s ON p.song_id = s.song_id -- Corrected join field for songs
        LEFT JOIN
            shows sh ON p.show = sh.id      -- Join with shows table
        WHERE
            p.comment IS NOT NULL AND TRIM(p.comment) != ''
        ORDER BY
            p.airdate DESC -- Or p.id, or any other preferred order
        """

        if limit:
            query += f" LIMIT {limit}"

        logging.info(f"Executing query (first 500 chars): {query[:500]}...")
        cursor.execute(query)

        count = 0
        for row in cursor:
            # Ensure comment_text (which is p.comment) is a string and not None
            comment_text = row['comment_text'] if row['comment_text'] is not None else ""

            record = {
                "text": comment_text,  # This is the DJ's comment
                "meta": {
                    "play_id": row['play_id'],
                    "dj_host_names": row['dj_host_names'],
                    "play_airdate": row['play_airdate'],
                    # Keep the original song_id from DB
                    "db_song_id": row['song_id'],
                    "db_song_title": row['song_title'],
                    "db_album_title": row['album_title'],
                    "db_artist_name": row['artist_name'],
                    "db_song_release_date": row['song_release_date']
                }
            }
            yield record
            count += 1
            if count % 1000 == 0:
                logging.info(f"Processed {count} play comments...")

        logging.info(
            f"Finished processing. Total play comments fetched: {count}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KEXP play comments with song/artist/show details to JSONL for Prodigy.")
    parser.add_argument(
        "output_file",
        help="Path to the output JSONL file (e.g., data/raw_kexp_data/kexp_play_comments_with_meta.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit the number of comments to process (for testing)."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting data extraction from: {db_path}")
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in fetch_play_comments_with_details(db_path, args.limit):
                outfile.write(json.dumps(record) + '\n')
        logging.info(
            f"Successfully wrote play comments to: {args.output_file}")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()

================
File: scripts/01_extract_db_terms_for_gazetteers.py
================
#!/usr/bin/env python3
import sqlite3
import os
import argparse
import logging
import pandas as pd
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def normalize_text_for_gazetteer(text):
    """
    Normalizes text for gazetteer entries.
    - Converts to string.
    - Replaces various dash types and multiple hyphens with a single space.
    - Normalizes all whitespace to a single space.
    - Strips leading/trailing whitespace.
    """
    if text is None:
        return None
    text_str = str(text)

    # Replace various dash/hyphen constructs with a single space
    text_str = text_str.replace("---", " ")
    text_str = text_str.replace("--", " ")
    text_str = text_str.replace("—", " ")  # em-dash
    text_str = text_str.replace("–", " ")  # en-dash

    # Normalize internal whitespace (e.g., multiple spaces to one)
    text_str = re.sub(r'\s+', ' ', text_str)
    text_str = text_str.strip()  # Strip leading/trailing whitespace after normalization

    return text_str if text_str else None


def clean_term(term):
    """
    Cleans a term for gazetteer creation.
    - Normalizes text (dashes, whitespace).
    - Removes common "Various Artists" prefixes.
    - Strips defined leading/trailing punctuation.
    - Ensures term does not start or end with hyphen or apostrophe after initial cleaning.
    """
    # Step 1: Core normalization (handles dashes, multiple spaces, leading/trailing spaces)
    normalized_term = normalize_text_for_gazetteer(term)
    if not normalized_term:  # if term was None or became empty after normalization
        return None

    # Step 2: Remove common "Various Artists" prefixes
    if normalized_term.lower().startswith('(v/a)') or normalized_term.lower().startswith('[v/a]'):
        return None  # Filter out

    # Step 3: Strip leading/trailing non-alphanumeric, non-space, non-dash, non-apostrophe characters.
    # This preserves internal hyphens and apostrophes that are part of the word.
    # Note: normalize_text_for_gazetteer already converted multi-hyphens to spaces.
    # This step now primarily handles other punctuation at the ends.
    cleaned_term = re.sub(r"^[^\w\s'-]+|[^\w\s'-]+$",
                          "", normalized_term).strip()

    # Step 4: After the above, ensure it doesn't start or end with a single hyphen or apostrophe
    # that might have been left if they were single and internal before becoming terminal by step 3.
    if cleaned_term.startswith("'") or cleaned_term.endswith("'"):
        cleaned_term = cleaned_term.strip("'")
    if cleaned_term.startswith("-") or cleaned_term.endswith("-"):
        # This check is for single hyphens, as multi-hyphens are now spaces
        cleaned_term = cleaned_term.strip("-")

    # Final strip to catch any new leading/trailing spaces
    cleaned_term = cleaned_term.strip()

    return cleaned_term if cleaned_term else None


def is_well_formed(term, min_len=3, min_alpha_ratio=0.5, min_alphanum_ratio=0.7):
    """
    Checks if a cleaned term is well-formed for a gazetteer with stricter rules.
    - Minimum length.
    - Minimum ratio of alphabetic characters.
    - Minimum ratio of alphanumeric characters.
    - Not purely digits.
    - Not purely punctuation/symbols (even if it passes length).
    """
    if not term:
        return False

    term_len = len(term)
    if term_len < min_len:
        return False

    alpha_chars = sum(1 for char in term if char.isalpha())
    alphanum_chars = sum(1 for char in term if char.isalnum())

    if term_len > 0:  # Avoid division by zero
        if (alpha_chars / term_len) < min_alpha_ratio:
            return False
        if (alphanum_chars / term_len) < min_alphanum_ratio:
            return False
    else:
        return False

    if term.isdigit():
        return False

    if alphanum_chars == 0 and term_len > 0:  # Purely non-alphanumeric
        return False

    # Filter out terms that are just one or two quote characters or similar noise
    # This check is after all cleaning and normalization.
    # "--", "---" should be gone due to normalization
    if term in ['"', "''", "'", " ", "-", "--", "---"]:
        return False

    return True


def extract_terms(db_path, table_name, column_name, output_file, term_type="generic"):
    """
    Extracts unique, non-empty, cleaned, and well-formed terms.
    Applies more conservative filtering.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT DISTINCT \"{column_name}\" FROM \"{table_name}\" WHERE \"{column_name}\" IS NOT NULL AND TRIM(\"{column_name}\") != '';"
        logging.info(
            f"Executing query for {table_name}.{column_name}: {query}")
        df = pd.read_sql_query(query, conn)

        if df.empty or column_name not in df.columns:
            logging.warning(
                f"No data found or column '{column_name}' not in table '{table_name}'. Skipping {output_file}.")
            return

        cleaned_terms = []
        original_count = len(df)
        terms_after_cleaning = 0
        well_formed_count = 0

        # Ensure we iterate over strings
        for term_val in df[column_name].astype(str):
            # This now includes advanced normalization
            cleaned = clean_term(term_val)
            if cleaned:
                terms_after_cleaning += 1
                current_min_len = 2
                if term_type == "artist":
                    if len(cleaned) == 1 and cleaned.isalpha():
                        current_min_len = 1
                    else:
                        current_min_len = 2
                elif term_type in ["song_title", "album_title"]:
                    current_min_len = 2

                if is_well_formed(cleaned, min_len=current_min_len):
                    cleaned_terms.append(cleaned)
                    well_formed_count += 1
                # else:
                #     logging.debug(f"Term '{cleaned}' (original: '{term_val[:50]}...') did not pass well-formedness check for type '{term_type}'.")
            # else:
            #     logging.debug(f"Term '{term_val[:50]}...' became None/empty after cleaning.")

        unique_terms = sorted(list(set(cleaned_terms)))

        logging.info(f"For {table_name}.{column_name}:")
        logging.info(
            f"  Original distinct terms from DB (non-null, non-empty): {original_count}")
        logging.info(
            f"  Terms remaining after full cleaning & normalization: {terms_after_cleaning}")
        logging.info(
            f"  Terms remaining after well-formedness checks: {well_formed_count}")
        logging.info(
            f"  Unique well-formed terms to be written: {len(unique_terms)}")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for term_item in unique_terms:
                f.write(term_item + '\n')
        logging.info(
            f"Successfully wrote {len(unique_terms)} unique, cleaned terms from {table_name}.{column_name} to {output_file}")

    except sqlite3.Error as e:
        logging.error(f"Database error for {table_name}.{column_name}: {e}")
    except Exception as e:
        logging.error(
            f"An unexpected error occurred for {table_name}.{column_name}: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clean terms from KEXP database for gazetteer files.")
    parser.add_argument(
        "--output_dir",
        default="data/gazetteers",
        help="Directory to save the gazetteer files (default: data/gazetteers)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting gazetteer extraction from: {db_path}")

        extract_terms(db_path, "songs", "artist", os.path.join(
            args.output_dir, "artists.txt"), term_type="artist")
        extract_terms(db_path, "songs", "song", os.path.join(
            args.output_dir, "songs.txt"), term_type="song_title")
        extract_terms(db_path, "songs", "album", os.path.join(
            args.output_dir, "albums.txt"), term_type="album_title")

        logging.info("Gazetteer extraction process complete.")

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during gazetteer generation: {e}")


if __name__ == "__main__":
    main()

================
File: scripts/02_create_and_test_matchers.py
================
#!/usr/bin/env python3
import spacy
from spacy.matcher import PhraseMatcher
# Matcher removed as we are not doing rule-based date patterns here for pre-labeling
import os
import argparse
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
NLP = None


def load_spacy_model(model_name="en_core_web_trf"):
    """Loads a spaCy model."""
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load(model_name)
            logging.info(f"spaCy model '{model_name}' loaded successfully.")
        except OSError:
            logging.error(
                f"spaCy model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
            raise
    return NLP


def load_gazetteer(filepath):
    """Loads terms from a text file, one term per line."""
    if not os.path.exists(filepath):
        logging.warning(
            f"Gazetteer file not found: {filepath}. Returning empty list.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(terms)} terms from {filepath}")
    return terms


def create_phrase_matcher(nlp_model, gazetteers_map):
    """
    Creates a spaCy PhraseMatcher from a dictionary of gazetteers.

    Args:
        nlp_model: The loaded spaCy model.
        gazetteers_map (dict): A dictionary where keys are entity labels (e.g., "ARTIST_NAME")
                               and values are lists of terms for that entity.

    Returns:
        spacy.matcher.PhraseMatcher: The configured PhraseMatcher.
    """
    matcher = PhraseMatcher(
        nlp_model.vocab, attr="LOWER")  # Match on lowercased text
    for label, terms in gazetteers_map.items():
        if terms:
            patterns = [nlp_model.make_doc(term) for term in terms]
            matcher.add(label, patterns)
            logging.info(
                f"Added {len(patterns)} patterns for label '{label}' to PhraseMatcher.")
        else:
            logging.warning(
                f"No terms provided for label '{label}'. Skipping PhraseMatcher addition.")
    return matcher

# create_rule_matcher function has been removed as per user request


def main():
    parser = argparse.ArgumentParser(
        description="Create and test PhraseMatcher patterns from core gazetteers (artist, song, album).")
    parser.add_argument(
        "--gazetteer_dir",
        default="data/gazetteers",
        help="Directory containing the gazetteer text files."
    )
    parser.add_argument(
        "--sample_text_file",
        help="Optional: Path to a JSONL file with texts to test the matcher on (expects 'text' key from script 00)."
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default='Death Cab For Cutie will play the Showbox on Friday, November 22nd. Their new album is "Plans", released last month.',
        help="A sample text string to test the matcher if no file is provided."
    )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_trf",
        help="Name of the spaCy model to use."
    )

    args = parser.parse_args()

    try:
        nlp = load_spacy_model(args.spacy_model)

        # Define which gazetteers to load for PhraseMatcher
        # Focused on Artist, Song, Album as per user request for pre-labeling
        gazetteer_files_for_phrase_matcher = {
            "ARTIST_NAME": os.path.join(args.gazetteer_dir, "artists.txt"),
            "SONG_TITLE": os.path.join(args.gazetteer_dir, "songs.txt"),
            "ALBUM_TITLE": os.path.join(args.gazetteer_dir, "albums.txt"),
            # VENUE_NAME and EVENT_NAME (formerly FESTIVAL_NAME) gazetteers will be manually annotated for NER training,
            # not used for pre-labeling by this script.
        }

        gazetteers_data = {}
        for label, filepath in gazetteer_files_for_phrase_matcher.items():
            gazetteers_data[label] = load_gazetteer(filepath)

        # Create the PhraseMatcher
        phrase_matcher = create_phrase_matcher(nlp, gazetteers_data)

        # Rule-based Matcher for dates is removed from this script's pre-labeling focus.
        # Date expressions will be manually annotated or handled by spaCy's default DATE NER.

        # Test the PhraseMatcher
        texts_to_process = []
        if args.sample_text_file:
            if not os.path.exists(args.sample_text_file):
                logging.error(
                    f"Sample text file not found: {args.sample_text_file}")
                return
            with open(args.sample_text_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        texts_to_process.append(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(
                            f"Skipping malformed line {i+1} in sample text file: {line.strip()} - Error: {e}")
            logging.info(
                f"Loaded {len(texts_to_process)} texts from {args.sample_text_file} for testing.")
        else:
            texts_to_process.append(args.sample_text)
            logging.info(f"Using provided sample text for testing.")

        # Test on first 5 samples if from file, or the single sample_text
        for text_to_test in texts_to_process[:5]:
            logging.info(f"\n--- Testing on text: --- \n'{text_to_test}'")
            doc = nlp(text_to_test)

            phrase_matches_output = []

            # PhraseMatcher results
            phrase_matches = phrase_matcher(doc)
            logging.info(f"PhraseMatcher found {len(phrase_matches)} matches:")
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                label = nlp.vocab.strings[match_id]
                logging.info(
                    f"  Label: {label}, Span: '{span.text}', Start: {span.start_char}, End: {span.end_char}")
                phrase_matches_output.append(
                    {"start": span.start_char, "end": span.end_char, "label": label})

            if phrase_matches_output:
                # Sort by start character to make it easier to read / process
                phrase_matches_sorted = sorted(
                    phrase_matches_output, key=lambda x: x["start"])
                prodigy_like_output = {
                    "text": text_to_test, "spans": phrase_matches_sorted}
                logging.info(
                    f"Prodigy-like output for this text (from PhraseMatcher only): {json.dumps(prodigy_like_output, indent=2)}")
            elif not phrase_matches:
                logging.info("  No matches found by PhraseMatcher.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: scripts/04_prelabel_for_prodigy.py
================
#!/usr/bin/env python3
import spacy
from spacy.tokens import Doc
from spacy.util import filter_spans
import os
import argparse
import json
import logging
from dotenv import load_dotenv
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
NLP = None

# Define label priorities
LABEL_PRIORITY = {
    "METADATA_ARTIST_NAME": 0,
    "METADATA_ALBUM_TITLE": 1,
    "METADATA_SONG_TITLE": 2
}


def get_label_priority(label):
    return LABEL_PRIORITY.get(label, 99)


def load_spacy_model(model_name="en_core_web_trf"):
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load(model_name, disable=[
                             "parser", "tagger", "ner", "lemmatizer"])
            logging.info(
                f"spaCy model '{model_name}' loaded (most components disabled for pre-labeling).")
        except OSError:
            logging.error(
                f"Model '{model_name}' not found. `python -m spacy download {model_name}`")
            raise
    return NLP


def normalize_text(text_content):
    """
    Normalizes text for consistent processing, matching, and output.
    - Converts to string.
    - Replaces various dash types (--, ---, em-dash, en-dash) with a single space.
    - Normalizes all other whitespace (spaces, tabs, newlines) to a single space.
    - Strips leading/trailing whitespace.
    """
    if text_content is None:
        return ""  # Return empty string for None input

    text_str = str(text_content)

    # Replace specific dash patterns first
    text_str = text_str.replace("---", " ")
    text_str = text_str.replace("--", " ")
    text_str = text_str.replace("—", " ")  # em-dash U+2014
    text_str = text_str.replace("–", " ")  # en-dash U+2013

    # Normalize all whitespace (including spaces from replaced dashes) to a single space
    text_str = re.sub(r'\s+', ' ', text_str)

    # Strip leading/trailing whitespace that might result from replacements or original text
    text_str = text_str.strip()

    return text_str


def find_exact_string_matches(normalized_doc_text, normalized_search_term, label_prefix, source_suffix):
    """
    Finds all exact, case-insensitive matches of a (normalized) search_term
    within the (normalized) doc_text.
    Args:
        normalized_doc_text (str): The pre-normalized text of the document/comment.
        normalized_search_term (str): The pre-normalized metadata term to search for.
        label_prefix (str): The base label (e.g., "ARTIST_NAME").
        source_suffix (str): Suffix for the source (e.g., "artist").
    Returns:
        list: List of match dictionaries.
    """
    matches = []
    # Check if search term or text is empty after normalization
    if not normalized_search_term or not normalized_doc_text:
        return matches

    try:
        # The normalized_search_term already has its internal spaces normalized.
        # We need to escape it for regex and then replace its internal single spaces
        # with \s+ to allow for flexible whitespace matching in the doc_text.
        # Example: "Artist Name" -> "Artist\s+Name"

        # Escape the whole term first to handle any special regex characters within words
        escaped_term = re.escape(normalized_search_term)

        # If normalized_search_term was "word1 word2", escaped_term is "word1\ word2".
        # We want "word1\s+word2". So, split the *original* normalized_search_term by space,
        # escape each part, then join with \s+.
        # Split by single space (it's normalized)
        words = normalized_search_term.split(' ')
        # Filter out empty strings if any
        escaped_words = [re.escape(word) for word in words if word]

        if not escaped_words:  # Should not happen if normalized_search_term is not empty
            return matches

        # This pattern will match the sequence of words from normalized_search_term,
        # allowing one or more whitespace characters between them in normalized_doc_text.
        flexible_space_pattern = r'\s+'.join(escaped_words)

        # Add word boundaries to ensure we match whole words/phrases.
        # (?<!\w) - negative lookbehind for a word character
        # (?!\w) - negative lookahead for a word character
        # This helps prevent matching "art" in "artist" if search term is "art".
        # However, if your search term can be part of a larger compound, this might be too strict.
        # For names and titles, it's often safer.
        # Consider if your terms can have leading/trailing non-word chars, e.g. "A.B.C."
        # re.escape handles the '.', so word boundaries should still work.
        # For now, let's use word boundaries.
        # If a term is "A-Go-Go", pattern becomes "A\-Go\-Go". Word boundaries work.
        # If a term is "!Action", pattern is "\!Action". (?<!\w)\!Action(?!\w) - might be fine.

        # More robust word boundary handling for terms that might start/end with non-alphanum:
        # If first/last char of term is non-alphanum, don't require word boundary there.

        prefix_boundary = r'(?<!\w)' if normalized_search_term[0].isalnum(
        ) else ''
        suffix_boundary = r'(?!\w)' if normalized_search_term[-1].isalnum(
        ) else ''

        # If the search term is very short (e.g. "U2", "Go"), boundaries are important.
        # If search term is "The Cure", boundaries are also good.
        # If search term is "Song (Remix)", pattern "Song\s+\(Remix\)"
        # prefix_boundary would be (?<!\w) and suffix_boundary would be '' (since ')' is not alnum) - this seems correct.

        pattern_to_search = prefix_boundary + flexible_space_pattern + suffix_boundary

        for match in re.finditer(pattern_to_search, normalized_doc_text, re.IGNORECASE):
            matches.append({
                "start_char": match.start(),
                "end_char": match.end(),
                # The actual matched text from the comment
                "text": match.group(0),
                "label": f"METADATA_{label_prefix}",
                "source": f"metadata_{source_suffix}"
            })
    except re.error as e:
        logging.warning(
            f"Regex error matching '{normalized_search_term}' with pattern '{pattern_to_search}' in '{normalized_doc_text[:50]}...': {e}")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Pre-labels KEXP comments with ARTIST_NAME, SONG_TITLE, ALBUM_TITLE using ONLY comment metadata, with text normalization."
    )
    parser.add_argument("input_jsonl_file",
                        help="Path to input JSONL (from script 00)")
    parser.add_argument("output_jsonl_file",
                        help="Path to output pre-labeled JSONL for Prodigy")
    parser.add_argument(
        "--spacy_model", default="en_core_web_trf", help="spaCy model.")
    parser.add_argument("--limit", type=int, help="Limit number of comments.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input_jsonl_file):
        logging.error(f"Input JSONL file not found: {args.input_jsonl_file}")
        return

    try:
        nlp = load_spacy_model(args.spacy_model)
        processed_count = 0

        with open(args.input_jsonl_file, 'r', encoding='utf-8') as infile, \
                open(args.output_jsonl_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                if args.limit and i >= args.limit:
                    logging.info(
                        f"Reached processing limit of {args.limit} records.")
                    break

                original_comment_text_for_error_logging = "N/A"
                try:
                    record = json.loads(line)
                    original_comment_text = record.get("text")
                    meta = record.get("meta", {})
                    original_comment_text_for_error_logging = original_comment_text if original_comment_text else "N/A"

                    if not original_comment_text:
                        logging.warning(
                            f"Skipping line {i+1} due to missing or empty 'text' field: {line.strip()}")
                        continue

                    # --- NORMALIZATION STEP 1: Normalize the input comment text ---
                    normalized_comment_text = normalize_text(
                        original_comment_text)

                    if not normalized_comment_text:  # If text became empty after normalization
                        logging.info(
                            f"Skipping line {i+1} as text became empty after normalization. Original: '{original_comment_text[:100]}...'")
                        continue

                    # --- Create spaCy Doc from the NORMALIZED text ---
                    doc = nlp(normalized_comment_text)
                    candidate_matches_info = []

                    # --- Metadata-based matches using normalized text and normalized metadata terms ---
                    metadata_fields_to_check = {
                        "db_artist_name": ("ARTIST_NAME", "artist"),
                        "db_song_title": ("SONG_TITLE", "song"),
                        "db_album_title": ("ALBUM_TITLE", "album")
                    }

                    for meta_key, (label_prefix, source_suffix) in metadata_fields_to_check.items():
                        raw_meta_value = meta.get(meta_key)
                        if raw_meta_value:
                            # --- NORMALIZATION STEP 2: Normalize the metadata term ---
                            normalized_meta_term = normalize_text(
                                raw_meta_value)
                            if normalized_meta_term:  # Ensure not empty after normalization
                                candidate_matches_info.extend(find_exact_string_matches(
                                    doc.text,  # This is normalized_comment_text
                                    normalized_meta_term,
                                    label_prefix,
                                    source_suffix
                                ))

                    # --- Resolve conflicts for identical spans by priority ---
                    spans_by_offset = defaultdict(list)
                    for match_info in candidate_matches_info:
                        spans_by_offset[(match_info["start_char"], match_info["end_char"])].append(
                            match_info)

                    unique_spans_for_filtering = []
                    for (start_char, end_char), match_group in spans_by_offset.items():
                        if not match_group:
                            continue
                        match_group.sort(key=lambda m: (
                            get_label_priority(m["label"]), m["label"]))
                        best_match = match_group[0]

                        # Create span object using the *normalized* doc
                        span_obj = doc.char_span(
                            best_match["start_char"], best_match["end_char"], label=best_match["label"])
                        if span_obj:
                            unique_spans_for_filtering.append(span_obj)
                        else:
                            logging.debug(
                                f"Could not create span for: {best_match} in doc (len {len(doc.text)}): '{doc.text[:70]}...' "
                                f"(start: {best_match['start_char']}, end: {best_match['end_char']}). "
                                f"Original text for this span was: '{normalized_comment_text[best_match['start_char']:best_match['end_char']]}'"
                            )

                    # --- Filter overlapping span boundaries using filter_spans ---
                    final_spans_obj = filter_spans(unique_spans_for_filtering)

                    # --- Convert final spaCy Span objects to Prodigy format ---
                    spans_for_prodigy = []
                    for span in final_spans_obj:
                        final_label = span.label_
                        if final_label.startswith("METADATA_"):
                            final_label = final_label.replace("METADATA_", "")

                        spans_for_prodigy.append({
                            "start": span.start_char,
                            "end": span.end_char,
                            "label": final_label,
                            "text": span.text,  # Text of the span from the normalized doc
                            "source": "metadata_match_v4_normalized_word_boundaries"  # Updated source
                        })
                    spans_for_prodigy.sort(key=lambda x: x["start"])

                    # --- NORMALIZATION STEP 3: Output the NORMALIZED text to Prodigy ---
                    output_record = {
                        "text": normalized_comment_text,  # Use the normalized text for Prodigy
                        # Keep original text in meta if needed
                        "meta": {**meta, "original_comment_text": original_comment_text},
                        "spans": spans_for_prodigy
                    }
                    outfile.write(json.dumps(
                        output_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logging.info(
                            f"Pre-labeled {processed_count} comments...")

                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON line {i+1}: {line.strip()}")
                except Exception as e:
                    logging.error(
                        f"Error processing line {i+1} (original text: '{original_comment_text_for_error_logging[:100]}...'): {e}", exc_info=True)

        logging.info(
            f"Successfully pre-labeled {processed_count} comments to: {args.output_jsonl_file}")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: .gitignore
================
.vscode
.env
.venv
__pycache__
*.pyc
*.pyo
*.pyd
*.pyw
*.pyz
*.pywz
/data/raw_kexp_data

================
File: .repomixignore
================
data/

================
File: data_to_spacy.sh
================
prodigy data-to-spacy ./data/spacy_training_data_music_only/ \
    --ner kexp_ner_annotations_v2 \
    --eval-split 0.2 \
    -l ARTIST_NAME,SONG_TITLE,ALBUM_TITLE,VENUE_NAME,LOCATION,DATE_EXPR

================
File: music_prodigy_annotate.md
================
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

Additional Info:
----------------

================================================================
Directory Structure
================================================================
scripts/
  00_extract_kexp_comments.py
  01_extract_db_terms_for_gazetteers.py
  02_create_and_test_matchers.py
  04_prelabel_for_prodigy.py
.gitignore
.repomixignore
requirements.txt
setup.sh

================================================================
Files
================================================================

================
File: scripts/00_extract_kexp_comments.py
================
#!/usr/bin/env python3
import sqlite3
import json
import os
import argparse
import logging
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_play_comments_with_details(db_path, limit=None):
    """
    Fetches plays with comments from the KEXP database and joins them with song
    (for song title, album, artist, release_date) and show (for host names) information.

    Args:
        db_path (str): Path to the SQLite database file.
        limit (int, optional): Maximum number of records to fetch. Defaults to None (all).

    Yields:
        dict: A dictionary representing a play's comment with its associated metadata.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # Schema based on user input:
        # plays.song_id -> songs.song_id
        # songs table contains: song (title), artist (name), album (title), release_date
        # plays.show -> shows.id (to get host_names)
        # Comment is directly on the plays table.
        query = """
        SELECT
            p.id AS play_id,
            p.comment AS comment_text,
            p.airdate AS play_airdate,
            p.song_id,
            s.song AS song_title,          -- From songs.song
            s.album AS album_title,        -- From songs.album
            s.artist AS artist_name,       -- From songs.artist
            s.release_date AS song_release_date, -- From songs.release_date
            sh.host_names AS dj_host_names -- Get host names from the shows table
        FROM
            plays p
        LEFT JOIN
            songs s ON p.song_id = s.song_id -- Corrected join field for songs
        LEFT JOIN
            shows sh ON p.show = sh.id      -- Join with shows table
        WHERE
            p.comment IS NOT NULL AND TRIM(p.comment) != ''
        ORDER BY
            p.airdate DESC -- Or p.id, or any other preferred order
        """

        if limit:
            query += f" LIMIT {limit}"

        logging.info(f"Executing query (first 500 chars): {query[:500]}...")
        cursor.execute(query)

        count = 0
        for row in cursor:
            # Ensure comment_text (which is p.comment) is a string and not None
            comment_text = row['comment_text'] if row['comment_text'] is not None else ""

            record = {
                "text": comment_text,  # This is the DJ's comment
                "meta": {
                    "play_id": row['play_id'],
                    "dj_host_names": row['dj_host_names'],
                    "play_airdate": row['play_airdate'],
                    # Keep the original song_id from DB
                    "db_song_id": row['song_id'],
                    "db_song_title": row['song_title'],
                    "db_album_title": row['album_title'],
                    "db_artist_name": row['artist_name'],
                    "db_song_release_date": row['song_release_date']
                }
            }
            yield record
            count += 1
            if count % 1000 == 0:
                logging.info(f"Processed {count} play comments...")

        logging.info(
            f"Finished processing. Total play comments fetched: {count}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KEXP play comments with song/artist/show details to JSONL for Prodigy.")
    parser.add_argument(
        "output_file",
        help="Path to the output JSONL file (e.g., data/raw_kexp_data/kexp_play_comments_with_meta.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit the number of comments to process (for testing)."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting data extraction from: {db_path}")
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in fetch_play_comments_with_details(db_path, args.limit):
                outfile.write(json.dumps(record) + '\n')
        logging.info(
            f"Successfully wrote play comments to: {args.output_file}")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()

================
File: scripts/01_extract_db_terms_for_gazetteers.py
================
#!/usr/bin/env python3
import sqlite3
import os
import argparse
import logging
import pandas as pd
from dotenv import load_dotenv
import re  # For more advanced cleaning

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def clean_term(term):
    """
    Cleans a term for gazetteer creation.
    - Converts to string and strips leading/trailing whitespace.
    - Removes common "Various Artists" prefixes.
    - Normalizes internal whitespace.
    - Strips defined leading/trailing punctuation.
    - Ensures term does not start or end with hyphen or apostrophe after initial cleaning.
    """
    if term is None:
        return None

    term_str = str(term).strip()

    if not term_str:
        return None

    # Remove common "Various Artists" prefixes
    if term_str.lower().startswith('(v/a)') or term_str.lower().startswith('[v/a]'):
        return None  # Filter out

    # Normalize internal whitespace (e.g., multiple spaces to one)
    term_str = re.sub(r'\s+', ' ', term_str)

    # Strip leading/trailing non-alphanumeric, non-space, non-dash, non-apostrophe characters
    # This preserves internal hyphens and apostrophes.
    term_str = re.sub(r"^[^\w\s'-]+|[^\w\s'-]+$", "", term_str).strip()

    # After the above, ensure it doesn't start or end with a hyphen or apostrophe
    if term_str.startswith("'") or term_str.endswith("'"):
        term_str = term_str.strip("'")
    if term_str.startswith("-") or term_str.endswith("-"):
        term_str = term_str.strip("-")

    term_str = term_str.strip()  # Final strip

    if not term_str:
        return None

    return term_str


def is_well_formed(term, min_len=3, min_alpha_ratio=0.5, min_alphanum_ratio=0.7):
    """
    Checks if a cleaned term is well-formed for a gazetteer with stricter rules.
    - Minimum length.
    - Minimum ratio of alphabetic characters.
    - Minimum ratio of alphanumeric characters.
    - Not purely digits.
    - Not purely punctuation/symbols (even if it passes length).
    """
    if not term:
        return False

    term_len = len(term)
    if term_len < min_len:
        return False

    alpha_chars = sum(1 for char in term if char.isalpha())
    alphanum_chars = sum(1 for char in term if char.isalnum())

    if term_len > 0:  # Avoid division by zero
        if (alpha_chars / term_len) < min_alpha_ratio:
            return False
        if (alphanum_chars / term_len) < min_alphanum_ratio:
            return False
    else:  # Should have been caught by min_len or previous checks
        return False

    # Check if it's purely digits
    if term.isdigit():
        return False

    # Check if (after cleaning) it's now purely non-alphanumeric (e.g. "!!", "--")
    # This is a bit redundant given the ratio checks but can catch edge cases.
    if alphanum_chars == 0 and term_len > 0:
        return False

    # Filter out terms that are just one or two quote characters or similar noise that might have slipped through
    if term in ['"', "''", "'", " ", "-", "--", "---"]:  # Added hyphens here
        return False

    return True


def extract_terms(db_path, table_name, column_name, output_file, term_type="generic"):
    """
    Extracts unique, non-empty, cleaned, and well-formed terms.
    Applies more conservative filtering.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT DISTINCT \"{column_name}\" FROM \"{table_name}\" WHERE \"{column_name}\" IS NOT NULL AND TRIM(\"{column_name}\") != '';"
        logging.info(
            f"Executing query for {table_name}.{column_name}: {query}")
        df = pd.read_sql_query(query, conn)

        if df.empty or column_name not in df.columns:
            logging.warning(
                f"No data found or column '{column_name}' not in table '{table_name}'. Skipping {output_file}.")
            return

        cleaned_terms = []
        original_count = len(df)
        cleaned_count = 0
        well_formed_count = 0

        for term_val in df[column_name].astype(str):
            cleaned = clean_term(term_val)
            if cleaned:
                cleaned_count += 1
                # Adjust min_len for specific types if necessary, e.g., allow shorter artist names
                current_min_len = 2
                if term_type == "artist":
                    # Potentially allow very short artist names if they are common (e.g., "M", "X")
                    # For now, let's stick to a slightly more general rule for artists.
                    # If a name is just one letter, it must be an alpha char.
                    if len(cleaned) == 1 and cleaned.isalpha():
                        current_min_len = 1
                    else:
                        current_min_len = 2  # Default for multi-char artists
                elif term_type in ["song_title", "album_title"]:
                    current_min_len = 2  # Titles should generally be longer than 1 char

                if is_well_formed(cleaned, min_len=current_min_len):
                    cleaned_terms.append(cleaned)
                    well_formed_count += 1

        unique_terms = sorted(list(set(cleaned_terms)))

        logging.info(f"For {table_name}.{column_name}:")
        logging.info(
            f"  Original distinct terms from DB (non-null, non-empty): {original_count}")
        logging.info(
            f"  Terms remaining after basic cleaning (strip, (V/A), normalize space): {cleaned_count}")
        logging.info(
            f"  Terms remaining after well-formedness checks: {well_formed_count}")
        logging.info(
            f"  Unique well-formed terms to be written: {len(unique_terms)}")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for term_item in unique_terms:
                f.write(term_item + '\n')
        logging.info(
            f"Successfully wrote {len(unique_terms)} unique, cleaned terms from {table_name}.{column_name} to {output_file}")

    except sqlite3.Error as e:
        logging.error(f"Database error for {table_name}.{column_name}: {e}")
    except Exception as e:
        logging.error(
            f"An unexpected error occurred for {table_name}.{column_name}: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clean terms from KEXP database for gazetteer files.")
    parser.add_argument(
        "--output_dir",
        default="data/gazetteers",
        help="Directory to save the gazetteer files (default: data/gazetteers)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting gazetteer extraction from: {db_path}")

        # Apply stricter rules for artists, songs, and albums
        extract_terms(db_path, "songs", "artist", os.path.join(
            args.output_dir, "artists.txt"), term_type="artist")
        extract_terms(db_path, "songs", "song", os.path.join(
            args.output_dir, "songs.txt"), term_type="song_title")
        extract_terms(db_path, "songs", "album", os.path.join(
            args.output_dir, "albums.txt"), term_type="album_title")

        logging.info("Gazetteer extraction process complete.")

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during gazetteer generation: {e}")


if __name__ == "__main__":
    main()

================
File: scripts/02_create_and_test_matchers.py
================
#!/usr/bin/env python3
import spacy
from spacy.matcher import PhraseMatcher
# Matcher removed as we are not doing rule-based date patterns here for pre-labeling
import os
import argparse
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
NLP = None


def load_spacy_model(model_name="en_core_web_trf"):
    """Loads a spaCy model."""
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load(model_name)
            logging.info(f"spaCy model '{model_name}' loaded successfully.")
        except OSError:
            logging.error(
                f"spaCy model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
            raise
    return NLP


def load_gazetteer(filepath):
    """Loads terms from a text file, one term per line."""
    if not os.path.exists(filepath):
        logging.warning(
            f"Gazetteer file not found: {filepath}. Returning empty list.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(terms)} terms from {filepath}")
    return terms


def create_phrase_matcher(nlp_model, gazetteers_map):
    """
    Creates a spaCy PhraseMatcher from a dictionary of gazetteers.

    Args:
        nlp_model: The loaded spaCy model.
        gazetteers_map (dict): A dictionary where keys are entity labels (e.g., "ARTIST_NAME")
                               and values are lists of terms for that entity.

    Returns:
        spacy.matcher.PhraseMatcher: The configured PhraseMatcher.
    """
    matcher = PhraseMatcher(
        nlp_model.vocab, attr="LOWER")  # Match on lowercased text
    for label, terms in gazetteers_map.items():
        if terms:
            patterns = [nlp_model.make_doc(term) for term in terms]
            matcher.add(label, patterns)
            logging.info(
                f"Added {len(patterns)} patterns for label '{label}' to PhraseMatcher.")
        else:
            logging.warning(
                f"No terms provided for label '{label}'. Skipping PhraseMatcher addition.")
    return matcher

# create_rule_matcher function has been removed as per user request


def main():
    parser = argparse.ArgumentParser(
        description="Create and test PhraseMatcher patterns from core gazetteers (artist, song, album).")
    parser.add_argument(
        "--gazetteer_dir",
        default="data/gazetteers",
        help="Directory containing the gazetteer text files."
    )
    parser.add_argument(
        "--sample_text_file",
        help="Optional: Path to a JSONL file with texts to test the matcher on (expects 'text' key from script 00)."
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default='Death Cab For Cutie will play the Showbox on Friday, November 22nd. Their new album is "Plans", released last month.',
        help="A sample text string to test the matcher if no file is provided."
    )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_trf",
        help="Name of the spaCy model to use."
    )

    args = parser.parse_args()

    try:
        nlp = load_spacy_model(args.spacy_model)

        # Define which gazetteers to load for PhraseMatcher
        # Focused on Artist, Song, Album as per user request for pre-labeling
        gazetteer_files_for_phrase_matcher = {
            "ARTIST_NAME": os.path.join(args.gazetteer_dir, "artists.txt"),
            "SONG_TITLE": os.path.join(args.gazetteer_dir, "songs.txt"),
            "ALBUM_TITLE": os.path.join(args.gazetteer_dir, "albums.txt"),
            # VENUE_NAME and EVENT_NAME (formerly FESTIVAL_NAME) gazetteers will be manually annotated for NER training,
            # not used for pre-labeling by this script.
        }

        gazetteers_data = {}
        for label, filepath in gazetteer_files_for_phrase_matcher.items():
            gazetteers_data[label] = load_gazetteer(filepath)

        # Create the PhraseMatcher
        phrase_matcher = create_phrase_matcher(nlp, gazetteers_data)

        # Rule-based Matcher for dates is removed from this script's pre-labeling focus.
        # Date expressions will be manually annotated or handled by spaCy's default DATE NER.

        # Test the PhraseMatcher
        texts_to_process = []
        if args.sample_text_file:
            if not os.path.exists(args.sample_text_file):
                logging.error(
                    f"Sample text file not found: {args.sample_text_file}")
                return
            with open(args.sample_text_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        texts_to_process.append(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(
                            f"Skipping malformed line {i+1} in sample text file: {line.strip()} - Error: {e}")
            logging.info(
                f"Loaded {len(texts_to_process)} texts from {args.sample_text_file} for testing.")
        else:
            texts_to_process.append(args.sample_text)
            logging.info(f"Using provided sample text for testing.")

        # Test on first 5 samples if from file, or the single sample_text
        for text_to_test in texts_to_process[:5]:
            logging.info(f"\n--- Testing on text: --- \n'{text_to_test}'")
            doc = nlp(text_to_test)

            phrase_matches_output = []

            # PhraseMatcher results
            phrase_matches = phrase_matcher(doc)
            logging.info(f"PhraseMatcher found {len(phrase_matches)} matches:")
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                label = nlp.vocab.strings[match_id]
                logging.info(
                    f"  Label: {label}, Span: '{span.text}', Start: {span.start_char}, End: {span.end_char}")
                phrase_matches_output.append(
                    {"start": span.start_char, "end": span.end_char, "label": label})

            if phrase_matches_output:
                # Sort by start character to make it easier to read / process
                phrase_matches_sorted = sorted(
                    phrase_matches_output, key=lambda x: x["start"])
                prodigy_like_output = {
                    "text": text_to_test, "spans": phrase_matches_sorted}
                logging.info(
                    f"Prodigy-like output for this text (from PhraseMatcher only): {json.dumps(prodigy_like_output, indent=2)}")
            elif not phrase_matches:
                logging.info("  No matches found by PhraseMatcher.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: scripts/04_prelabel_for_prodigy.py
================
#!/usr/bin/env python3
import spacy
from spacy.tokens import Doc  # Import Doc for creating spaCy documents
from spacy.util import filter_spans
import os
import argparse
import json
import logging
from dotenv import load_dotenv
# Still useful if a metadata term matches multiple times
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
NLP = None

# Define label priorities - simplified as we are only dealing with metadata sources now
# but keeping structure in case of future expansion or if a term is in multiple meta fields.
LABEL_PRIORITY = {
    "METADATA_ARTIST_NAME": 0,  # Highest priority
    "METADATA_ALBUM_TITLE": 1,
    "METADATA_SONG_TITLE": 2
    # No general gazetteer labels needed here for priority
}


def get_label_priority(label):
    return LABEL_PRIORITY.get(label, 99)


def load_spacy_model(model_name="en_core_web_trf"):
    global NLP
    if NLP is None:
        try:
            # We only need tokenization for character offsets and creating Doc objects.
            # Disabling other components for speed.
            NLP = spacy.load(model_name, disable=[
                             "parser", "tagger", "ner", "lemmatizer"])
            logging.info(
                f"spaCy model '{model_name}' loaded (most components disabled for pre-labeling).")
        except OSError:
            logging.error(
                f"Model '{model_name}' not found. `python -m spacy download {model_name}`")
            raise
    return NLP


def find_exact_string_matches(text_content, search_string, label_prefix, source_suffix):
    """
    Finds all exact, case-insensitive matches of search_string in text_content,
    being robust to internal whitespace variations in text_content.
    Args:
        text_content (str): The text to search within.
        search_string (str): The string to search for (from metadata).
        label_prefix (str): The base label (e.g., "ARTIST_NAME").
        source_suffix (str): Suffix for the source (e.g., "artist").
    Returns:
        list: List of match dictionaries.
    """
    matches = []
    if not search_string or not text_content:  # Ensure search_string is not None or empty
        return matches

    # Ensure it's a string and strip it
    search_string_str = str(search_string).strip()
    if not search_string_str:  # If stripping makes it empty
        return matches

    try:
        # Create a regex pattern that allows for flexible whitespace
        # between words in the search string.
        # Split metadata term by any whitespace
        words = re.split(r'\s+', search_string_str)
        # Escape each part, ignore empty parts
        escaped_words = [re.escape(word) for word in words if word]

        if not escaped_words:  # If search string was only whitespace
            return matches

        # Join with \s+ to match one or more whitespace characters between words in the target text
        flexible_space_pattern = r'\s+'.join(escaped_words)

        # Use word boundaries (\b) if you want to match whole words only,
        # but this can be tricky if search_string itself starts/ends with non-word chars.
        # For titles/names, sometimes it's better without \b for partial but correct phrase matches.
        # Example: If search_string is "Live", \bLive\b won't match "Live at KEXP".
        # For now, let's try without strict word boundaries on the overall pattern for flexibility.
        # If you find it's too greedy, you can add them:
        # flexible_space_pattern = r'\b' + flexible_space_pattern + r'\b'
        # However, re.escape already handles individual words, so this should be fairly specific.

        for match in re.finditer(flexible_space_pattern, text_content, re.IGNORECASE):
            matches.append({
                "start_char": match.start(),
                "end_char": match.end(),
                # The actual matched text from the comment (which might have varied whitespace)
                "text": match.group(0),
                # Temporary label for prioritization
                "label": f"METADATA_{label_prefix}",
                "source": f"metadata_{source_suffix}"
            })
    except re.error as e:
        logging.warning(
            f"Regex error matching '{search_string_str}' in '{text_content[:50]}...': {e}")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Pre-labels KEXP comments with ARTIST_NAME, SONG_TITLE, ALBUM_TITLE using ONLY comment metadata."
    )
    parser.add_argument("input_jsonl_file",
                        help="Path to input JSONL (from script 00)")
    parser.add_argument("output_jsonl_file",
                        help="Path to output pre-labeled JSONL for Prodigy")
    parser.add_argument(
        "--spacy_model", default="en_core_web_trf", help="spaCy model.")
    parser.add_argument("--limit", type=int, help="Limit number of comments.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input_jsonl_file):
        logging.error(f"Input JSONL file not found: {args.input_jsonl_file}")
        return

    try:
        nlp = load_spacy_model(args.spacy_model)

        processed_count = 0
        with open(args.input_jsonl_file, 'r', encoding='utf-8') as infile, \
                open(args.output_jsonl_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                if args.limit and i >= args.limit:
                    logging.info(
                        f"Reached processing limit of {args.limit} records.")
                    break
                try:
                    record = json.loads(line)
                    text_to_annotate = record.get("text")
                    meta = record.get("meta", {})
                    if not text_to_annotate:
                        logging.warning(
                            f"Skipping line {i+1} due to missing or empty 'text' field: {line.strip()}")
                        continue

                    doc = nlp(text_to_annotate)

                    candidate_matches_info = []

                    # 1. Metadata-based matches ONLY
                    if meta.get("db_artist_name"):
                        candidate_matches_info.extend(find_exact_string_matches(
                            doc.text, meta["db_artist_name"], "ARTIST_NAME", "artist"
                        ))
                    if meta.get("db_song_title"):
                        candidate_matches_info.extend(find_exact_string_matches(
                            doc.text, meta["db_song_title"], "SONG_TITLE", "song"
                        ))
                    if meta.get("db_album_title"):
                        candidate_matches_info.extend(find_exact_string_matches(
                            doc.text, meta["db_album_title"], "ALBUM_TITLE", "album"
                        ))

                    # 2. Resolve conflicts for *identical spans* first by priority
                    spans_by_offset = defaultdict(list)
                    for match_info in candidate_matches_info:
                        spans_by_offset[(match_info["start_char"], match_info["end_char"])].append(
                            match_info)

                    unique_spans_for_filtering = []
                    for (start_char, end_char), match_group in spans_by_offset.items():
                        if not match_group:
                            continue
                        match_group.sort(key=lambda m: (
                            get_label_priority(m["label"]), m["label"]))
                        best_match = match_group[0]

                        span_obj = doc.char_span(
                            best_match["start_char"], best_match["end_char"], label=best_match["label"])
                        if span_obj:
                            unique_spans_for_filtering.append(span_obj)
                        else:
                            logging.debug(
                                f"Could not create span for: {best_match} in doc '{doc.text[:50]}...' (start: {best_match['start_char']}, end: {best_match['end_char']})")

                    # 3. Filter overlapping span boundaries using filter_spans
                    final_spans_obj = filter_spans(unique_spans_for_filtering)

                    # 4. Convert final spaCy Span objects to Prodigy format
                    spans_for_prodigy = []
                    for span in final_spans_obj:
                        final_label = span.label_
                        if final_label.startswith("METADATA_"):
                            final_label = final_label.replace("METADATA_", "")

                        spans_for_prodigy.append({
                            "start": span.start_char,
                            "end": span.end_char,
                            "label": final_label,
                            "text": span.text,
                            "source": "metadata_match_v2"  # Updated source
                        })

                    spans_for_prodigy.sort(key=lambda x: x["start"])

                    output_record = {
                        "text": text_to_annotate,
                        "meta": meta,
                        "spans": spans_for_prodigy
                    }
                    outfile.write(json.dumps(
                        output_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logging.info(
                            f"Pre-labeled {processed_count} comments...")

                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON line {i+1}: {line.strip()}")
                except Exception as e:
                    logging.error(
                        f"Error processing line {i+1}: '{text_to_annotate[:100] if text_to_annotate else 'N/A'}...' - {e}", exc_info=True)

        logging.info(
            f"Successfully pre-labeled {processed_count} comments to: {args.output_jsonl_file}")

    except Exception as e:
        logging.error(f"An critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

================
File: .gitignore
================
.vscode
.env
.venv
__pycache__
*.pyc
*.pyo
*.pyd
*.pyw
*.pyz
*.pywz
/data/raw_kexp_data

================
File: .repomixignore
================
data/

================
File: requirements.txt
================
aiofiles==24.1.0
annotated-types==0.7.0
anyio==4.9.0
blis==0.7.11
cachetools==5.5.2
catalogue==2.0.10
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.1.8
cloudpathlib==0.21.1
confection==0.1.5
curated-tokenizers==0.0.9
curated-transformers==0.1.1
cymem==2.0.11
en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl#sha256=272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5
fastapi==0.110.3
filelock==3.18.0
fsspec==2025.3.2
h11==0.16.0
idna==3.10
Jinja2==3.1.6
langcodes==3.5.0
language_data==1.3.0
marisa-trie==1.2.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
murmurhash==1.0.12
networkx==3.4.2
numpy==1.26.4
packaging==25.0
peewee==3.16.3
preshed==3.0.9
prodigy @ file:///Users/pooks/Dev/kexp_data/prodigy-1.17.5-py3-none-any.whl#sha256=a4c4bf09295ac418956c4f28f2d22c47c4d1708b99874bb9af8d917b0b56ebc2
pydantic==2.11.4
pydantic_core==2.33.2
Pygments==2.19.1
PyJWT==2.10.1
python-dotenv==1.1.0
radicli==0.0.25
regex==2024.11.6
requests==2.32.3
rich==14.0.0
setuptools==80.7.1
shellingham==1.5.4
smart-open==7.1.0
sniffio==1.3.1
spacy==3.7.5
spacy-curated-transformers==0.3.0
spacy-legacy==3.0.12
spacy-llm==0.7.3
spacy-loggers==1.0.5
srsly==2.5.1
starlette==0.37.2
sympy==1.14.0
thinc==8.2.5
thinc_apple_ops==1.0.0
toolz==0.12.1
torch==2.7.0
tqdm==4.67.1
typeguard==3.0.2
typer==0.15.4
typing-inspection==0.4.0
typing_extensions==4.13.2
urllib3==2.4.0
uvicorn==0.34.2
wasabi==1.1.3
weasel==0.4.1
wrapt==1.17.2

================
File: setup.sh
================
# Create the main project directory
mkdir kexp_knowledge_base
cd kexp_knowledge_base

# Create subdirectories
mkdir -p data/raw_kexp_data
mkdir -p data/gazetteers
mkdir -p data/prodigy_exports
mkdir -p data/spacy_training_data
mkdir models
mkdir scripts
mkdir notebooks
mkdir -p src/kexp_processing
mkdir configs

# Create some initial files (you'll populate these later)
touch data/raw_kexp_data/.gitkeep  # .gitkeep makes empty dirs trackable by git
touch data/gazetteers/artists.txt
touch data/gazetteers/songs.txt
touch data/gazetteers/albums.txt
touch data/gazetteers/venues.txt
touch data/gazetteers/festivals.txt
touch data/prodigy_exports/.gitkeep
touch data/spacy_training_data/.gitkeep
touch models/.gitkeep
touch scripts/01_extract_db_terms.py
touch scripts/02_compile_gazetteers.py
touch scripts/03_create_matchers.py
touch scripts/04_prelabel_for_prodigy.py
touch scripts/05_train_spacy_model.py
touch scripts/06_extract_relations.py
touch notebooks/data_exploration.ipynb
touch src/__init__.py
touch src/kexp_processing/__init__.py
touch src/kexp_processing/entity_extractors.py
touch configs/ner_config.cfg
touch .gitignore
touch README.md
touch requirements.txt

echo "Project directory structure for 'kexp_knowledge_base' created."
echo "Next steps:"
echo "1. cd kexp_knowledge_base"
echo "2. python3 -m venv venv  # Create a virtual environment"
echo "3. source venv/bin/activate # Activate it (on Linux/macOS)"
echo "4. pip install -r requirements.txt # (After you add dependencies to requirements.txt)"
echo "5. git init && git add . && git commit -m 'Initial project structure' # Initialize Git"



================================================================
End of Codebase
================================================================

================
File: requirements.txt
================
aiofiles==24.1.0
annotated-types==0.7.0
anyio==4.9.0
blis==0.7.11
cachetools==5.5.2
catalogue==2.0.10
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.1.8
cloudpathlib==0.21.1
confection==0.1.5
curated-tokenizers==0.0.9
curated-transformers==0.1.1
cymem==2.0.11
en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl#sha256=272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5
fastapi==0.110.3
filelock==3.18.0
fsspec==2025.3.2
h11==0.16.0
idna==3.10
Jinja2==3.1.6
langcodes==3.5.0
language_data==1.3.0
marisa-trie==1.2.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
murmurhash==1.0.12
networkx==3.4.2
numpy==1.26.4
packaging==25.0
peewee==3.16.3
preshed==3.0.9
prodigy @ file:///Users/pooks/Dev/kexp_data/prodigy-1.17.5-py3-none-any.whl#sha256=a4c4bf09295ac418956c4f28f2d22c47c4d1708b99874bb9af8d917b0b56ebc2
pydantic==2.11.4
pydantic_core==2.33.2
Pygments==2.19.1
PyJWT==2.10.1
python-dotenv==1.1.0
radicli==0.0.25
regex==2024.11.6
requests==2.32.3
rich==14.0.0
setuptools==80.7.1
shellingham==1.5.4
smart-open==7.1.0
sniffio==1.3.1
spacy==3.7.5
spacy-curated-transformers==0.3.0
spacy-legacy==3.0.12
spacy-llm==0.7.3
spacy-loggers==1.0.5
srsly==2.5.1
starlette==0.37.2
sympy==1.14.0
thinc==8.2.5
thinc_apple_ops==1.0.0
toolz==0.12.1
torch==2.7.0
tqdm==4.67.1
typeguard==3.0.2
typer==0.15.4
typing-inspection==0.4.0
typing_extensions==4.13.2
urllib3==2.4.0
uvicorn==0.34.2
wasabi==1.1.3
weasel==0.4.1
wrapt==1.17.2

================
File: setup.sh
================
# Create the main project directory
mkdir kexp_knowledge_base
cd kexp_knowledge_base

# Create subdirectories
mkdir -p data/raw_kexp_data
mkdir -p data/gazetteers
mkdir -p data/prodigy_exports
mkdir -p data/spacy_training_data
mkdir models
mkdir scripts
mkdir notebooks
mkdir -p src/kexp_processing
mkdir configs

# Create some initial files (you'll populate these later)
touch data/raw_kexp_data/.gitkeep  # .gitkeep makes empty dirs trackable by git
touch data/gazetteers/artists.txt
touch data/gazetteers/songs.txt
touch data/gazetteers/albums.txt
touch data/gazetteers/venues.txt
touch data/gazetteers/festivals.txt
touch data/prodigy_exports/.gitkeep
touch data/spacy_training_data/.gitkeep
touch models/.gitkeep
touch scripts/01_extract_db_terms.py
touch scripts/02_compile_gazetteers.py
touch scripts/03_create_matchers.py
touch scripts/04_prelabel_for_prodigy.py
touch scripts/05_train_spacy_model.py
touch scripts/06_extract_relations.py
touch notebooks/data_exploration.ipynb
touch src/__init__.py
touch src/kexp_processing/__init__.py
touch src/kexp_processing/entity_extractors.py
touch configs/ner_config.cfg
touch .gitignore
touch README.md
touch requirements.txt

echo "Project directory structure for 'kexp_knowledge_base' created."
echo "Next steps:"
echo "1. cd kexp_knowledge_base"
echo "2. python3 -m venv venv  # Create a virtual environment"
echo "3. source venv/bin/activate # Activate it (on Linux/macOS)"
echo "4. pip install -r requirements.txt # (After you add dependencies to requirements.txt)"
echo "5. git init && git add . && git commit -m 'Initial project structure' # Initialize Git"



================================================================
End of Codebase
================================================================

================
File: requirements.txt
================
aiofiles==24.1.0
annotated-types==0.7.0
anyio==4.9.0
blis==0.7.11
cachetools==5.5.2
catalogue==2.0.10
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.1.8
cloudpathlib==0.21.1
confection==0.1.5
curated-tokenizers==0.0.9
curated-transformers==0.1.1
cymem==2.0.11
en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl#sha256=272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5
fastapi==0.110.3
filelock==3.18.0
fsspec==2025.3.2
h11==0.16.0
idna==3.10
Jinja2==3.1.6
langcodes==3.5.0
language_data==1.3.0
marisa-trie==1.2.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
murmurhash==1.0.12
networkx==3.4.2
numpy==1.26.4
packaging==25.0
peewee==3.16.3
preshed==3.0.9
prodigy @ file:///Users/pooks/Dev/kexp_data/prodigy-1.17.5-py3-none-any.whl#sha256=a4c4bf09295ac418956c4f28f2d22c47c4d1708b99874bb9af8d917b0b56ebc2
pydantic==2.11.4
pydantic_core==2.33.2
Pygments==2.19.1
PyJWT==2.10.1
python-dotenv==1.1.0
radicli==0.0.25
regex==2024.11.6
requests==2.32.3
rich==14.0.0
setuptools==80.7.1
shellingham==1.5.4
smart-open==7.1.0
sniffio==1.3.1
spacy==3.7.5
spacy-curated-transformers==0.3.0
spacy-legacy==3.0.12
spacy-llm==0.7.3
spacy-loggers==1.0.5
srsly==2.5.1
starlette==0.37.2
sympy==1.14.0
thinc==8.2.5
thinc_apple_ops==1.0.0
toolz==0.12.1
torch==2.7.0
tqdm==4.67.1
typeguard==3.0.2
typer==0.15.4
typing-inspection==0.4.0
typing_extensions==4.13.2
urllib3==2.4.0
uvicorn==0.34.2
wasabi==1.1.3
weasel==0.4.1
wrapt==1.17.2

================
File: setup.py
================
from setuptools import setup, find_packages

setup(
    name="kexp_data",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="./src"),
)

================
File: setup.sh
================
# Create the main project directory
mkdir kexp_knowledge_base
cd kexp_knowledge_base

# Create subdirectories
mkdir -p data/raw_kexp_data
mkdir -p data/gazetteers
mkdir -p data/prodigy_exports
mkdir -p data/spacy_training_data
mkdir models
mkdir scripts
mkdir notebooks
mkdir -p src/kexp_processing
mkdir configs

# Create some initial files (you'll populate these later)
touch data/raw_kexp_data/.gitkeep  # .gitkeep makes empty dirs trackable by git
touch data/gazetteers/artists.txt
touch data/gazetteers/songs.txt
touch data/gazetteers/albums.txt
touch data/gazetteers/venues.txt
touch data/gazetteers/festivals.txt
touch data/prodigy_exports/.gitkeep
touch data/spacy_training_data/.gitkeep
touch models/.gitkeep
touch scripts/01_extract_db_terms.py
touch scripts/02_compile_gazetteers.py
touch scripts/03_create_matchers.py
touch scripts/04_prelabel_for_prodigy.py
touch scripts/05_train_spacy_model.py
touch scripts/06_extract_relations.py
touch notebooks/data_exploration.ipynb
touch src/__init__.py
touch src/kexp_processing/__init__.py
touch src/kexp_processing/entity_extractors.py
touch configs/ner_config.cfg
touch .gitignore
touch README.md
touch requirements.txt

echo "Project directory structure for 'kexp_knowledge_base' created."
echo "Next steps:"
echo "1. cd kexp_knowledge_base"
echo "2. python3 -m venv venv  # Create a virtual environment"
echo "3. source venv/bin/activate # Activate it (on Linux/macOS)"
echo "4. pip install -r requirements.txt # (After you add dependencies to requirements.txt)"
echo "5. git init && git add . && git commit -m 'Initial project structure' # Initialize Git"



================================================================
End of Codebase
================================================================
