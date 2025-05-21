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
