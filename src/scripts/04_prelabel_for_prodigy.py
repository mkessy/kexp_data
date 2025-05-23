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
from src.scripts import prelabel_config  # Import the new config module

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()

# Global NLP object (load once)
# nlp_global = None # Will be removed

# Define label priorities
# LABEL_PRIORITY = {
# "METADATA_ARTIST_NAME": 0,
# "METADATA_ALBUM_TITLE": 1,
# "METADATA_SONG_TITLE": 2
# }


def get_label_priority(label):
    return prelabel_config.LABEL_PRIORITY.get(label, 99)


def load_spacy_model(model_name=None):
    effective_model_name = model_name if model_name else prelabel_config.DEFAULT_SPACY_MODEL
    try:
        nlp = spacy.load(effective_model_name,
                         disable=prelabel_config.SPACY_DISABLED_COMPONENTS)
        logging.info(
            f"spaCy model '{effective_model_name}' loaded (components {prelabel_config.SPACY_DISABLED_COMPONENTS} disabled).")
        return nlp
    except OSError:
        logging.error(
            f"Model '{effective_model_name}' not found. Please run: python -m spacy download {effective_model_name}")
        raise


def _build_regex_pattern(normalized_search_term: str) -> str | None:
    """Helper to build the regex pattern for find_exact_string_matches."""
    if not normalized_search_term:
        return None
    try:
        words = normalized_search_term.split(' ')
        escaped_words = [re.escape(word) for word in words if word]
        if not escaped_words:
            return None

        flexible_space_pattern = r'\s+'.join(escaped_words)

        prefix_boundary = r'(?<!\w)' if normalized_search_term[0].isalnum(
        ) else ''
        suffix_boundary = r'(?!\w)' if normalized_search_term[-1].isalnum(
        ) else ''

        return prefix_boundary + flexible_space_pattern + suffix_boundary
    except Exception as e:
        logging.warning(
            f"Error building regex pattern for '{normalized_search_term}': {e}")
        return None


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
    pattern_to_search = _build_regex_pattern(normalized_search_term)
    if not pattern_to_search:
        return matches  # Could not build pattern

    try:
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
                f"Regex error for term '{normalized_search_term}' with pattern '{pattern_to_search}' in '{normalized_doc_text[:50]}...': {e}")
        else:
            logging.warning(
                f"Regex error for term '{normalized_search_term}' (pattern couldn't be built) in '{normalized_doc_text[:50]}...': {e}")
    return matches


def extract_and_normalize_metadata(meta_dict: dict, metadata_fields_config: dict) -> list:
    """
    Extracts and normalizes metadata terms based on the configuration.
    Returns a list of tuples: (normalized_meta_term, label_prefix, source_suffix)
    """
    normalized_metadata = []
    for meta_key, (label_prefix, source_suffix) in metadata_fields_config.items():
        raw_meta_value = meta_dict.get(meta_key)
        if raw_meta_value:
            normalized_meta_term = normalize_text(raw_meta_value)
            if normalized_meta_term:  # Ensure not empty after normalization
                normalized_metadata.append(
                    (normalized_meta_term, label_prefix, source_suffix)
                )
    return normalized_metadata


def find_all_candidate_matches(normalized_doc_text: str, metadata_to_match: list) -> list:
    """
    Finds all candidate matches for the given metadata terms in the document text.
    metadata_to_match is a list of (normalized_meta_term, label_prefix, source_suffix)
    """
    all_matches = []
    for normalized_meta_term, label_prefix, source_suffix in metadata_to_match:
        all_matches.extend(find_exact_string_matches(
            normalized_doc_text,
            normalized_meta_term,
            label_prefix,
            source_suffix
        ))
    return all_matches


def resolve_span_conflicts(candidate_matches_info: list, normalized_doc: Doc, label_priority_config: dict) -> list:
    """
    Resolves conflicts for identical and overlapping spans.
    Returns a list of spaCy Span objects.
    """
    spans_by_offset = defaultdict(list)
    for match_info in candidate_matches_info:
        spans_by_offset[(match_info["start_char"],
                         match_info["end_char"])].append(match_info)

    unique_spans_for_filtering = []
    for (start_char, end_char), match_group in spans_by_offset.items():
        if not match_group:
            continue
        match_group.sort(key=lambda m: (
            get_label_priority(m["label"]), m["label"]))
        best_match = match_group[0]

        span_obj = normalized_doc.char_span(
            best_match["start_char"], best_match["end_char"], label=best_match["label"]
        )
        if span_obj:
            unique_spans_for_filtering.append(span_obj)
        else:
            logging.debug(
                f"Could not create span for: {best_match} in doc (len {len(normalized_doc.text)}): '{normalized_doc.text[:70]}...' "
                f"(start: {best_match['start_char']}, end: {best_match['end_char']}). "
                f"Original text for this span was: '{normalized_doc.text[best_match['start_char']:best_match['end_char']]}'"
            )

    # Filter overlapping span boundaries using filter_spans
    return filter_spans(unique_spans_for_filtering)


def format_spans_for_prodigy(final_spans_obj: list, source_info: str) -> list:
    """
    Converts final spaCy Span objects to Prodigy format.
    """
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
            "source": source_info
        })
    spans_for_prodigy.sort(key=lambda x: x["start"])
    return spans_for_prodigy


def process_record(record_line: str, nlp_model: spacy.Language, config) -> dict | None:
    """
    Processes a single record (JSON line) from the input file.
    Returns a dictionary ready for Prodigy output, or None if skipping/error.
    """
    original_comment_text_for_error_logging = "N/A"
    try:
        record = json.loads(record_line)
        original_comment_text = record.get("text")
        meta = record.get("meta", {})
        original_comment_text_for_error_logging = original_comment_text if original_comment_text else "N/A"

        if not original_comment_text:
            logging.warning(
                f"Skipping record due to missing or empty 'text' field: {record_line.strip()}")
            return None

        normalized_comment_text = normalize_text(original_comment_text)
        if not normalized_comment_text:
            logging.info(
                f"Skipping record as text became empty after normalization. Original: '{original_comment_text[:100]}...'")
            return None

        doc = nlp_model(normalized_comment_text)

        # 1. Extract and normalize metadata
        metadata_to_match = extract_and_normalize_metadata(
            meta, config.METADATA_FIELDS_TO_CHECK)

        # 2. Find all candidate matches
        candidate_matches_info = find_all_candidate_matches(
            doc.text, metadata_to_match)

        # 3. Resolve span conflicts
        # Pass doc (the spaCy object from normalized_comment_text)
        final_spans_obj_list = resolve_span_conflicts(
            candidate_matches_info, doc, config.LABEL_PRIORITY)

        # 4. Format spans for Prodigy
        spans_for_prodigy = format_spans_for_prodigy(
            final_spans_obj_list, config.PRODIGY_SPAN_SOURCE)

        return {
            "text": normalized_comment_text,
            "meta": {**meta, "original_comment_text": original_comment_text},
            "spans": spans_for_prodigy
        }

    except json.JSONDecodeError:
        logging.warning(f"Skipping malformed JSON line: {record_line.strip()}")
        return None
    except Exception as e:
        logging.error(
            f"Error processing record (original text: '{original_comment_text_for_error_logging[:100]}...'): {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-labels KEXP comments with ARTIST_NAME, SONG_TITLE, ALBUM_TITLE using ONLY comment metadata, with text normalization."
    )
    parser.add_argument("input_jsonl_file",
                        help="Path to input JSONL (from script 00)")
    parser.add_argument("output_jsonl_file",
                        help="Path to output pre-labeled JSONL for Prodigy")
    parser.add_argument(
        "--spacy_model", default=None, help=f"spaCy model. Defaults to '{prelabel_config.DEFAULT_SPACY_MODEL}' from config.")
    parser.add_argument("--limit", type=int, help="Limit number of comments.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input_jsonl_file):
        logging.error(f"Input JSONL file not found: {args.input_jsonl_file}")
        return

    try:
        nlp_model = load_spacy_model(args.spacy_model)
        processed_count = 0

        with open(args.input_jsonl_file, 'r', encoding='utf-8') as infile, \
                open(args.output_jsonl_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                if args.limit and i >= args.limit:
                    logging.info(
                        f"Reached processing limit of {args.limit} records.")
                    break

                record = process_record(line, nlp_model, prelabel_config)
                if record:
                    outfile.write(json.dumps(
                        record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logging.info(
                            f"Pre-labeled {processed_count} comments...")

        logging.info(
            f"Successfully pre-labeled {processed_count} comments to: {args.output_jsonl_file}")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
