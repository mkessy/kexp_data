#!/usr/bin/env python3
import json
import re
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Text Normalization (applied to each segment *after* splitting) ---


def normalize_text(text_content):
    if text_content is None:  # Should not happen with new split logic, but good for safety
        return ""
    text_str = str(text_content)
    text_str = text_str.replace("—", " ")  # em-dash U+2014
    text_str = text_str.replace("–", " ")  # en-dash U+2013
    text_str = re.sub(r'\s+', ' ', text_str)
    text_str = text_str.strip()
    return text_str


# --- Regex Patterns ---
# URL_REGEX_STR: Define with non-capturing groups internally if complex, or ensure no optional captures if used in re.split's main pattern.
# For finditer, internal captures in URL_REGEX_STR are fine as we use match.group(0) for the main pattern.
URL_REGEX_STR = r'(https?://[^\s/$.?#].[^\s]*|[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|ly|eu|info|biz|ws|us|ca|uk|au|de|jp|fr|ch|fm|tv|me|sh|stream|live|watch|listen|download|video|audio|pics|photo|img|image|gallery|news|blog|shop|store|app|co|info|online|site|website|xyz|club|dev|page|link|art|bandcamp|soundcloud|spotify|youtube|youtu\.be|vimeo|tiktok|instagram|facebook|twitter|patreon|kexp)(?:/(?:[^\s()<>]+|(?:\([^\s()<>]+(?:\([^\s()<>]+\))*\)))*)?(?:\?[^\s]*)?)'

# 1. Pattern to FIND <text>:<url> lines (for isolating them as special segments).
#    match.group(0) will be the whole matched line.
TEXT_COLON_URL_LINE_PATTERN_FOR_FINDITER = re.compile(
    r'^[ \t]*[\w\s\(\)\[\].,!?\"-&\u2019\u201c\u201d;<>]+?:\s*' +
    URL_REGEX_STR + r'[ \t]*$',
    re.IGNORECASE | re.MULTILINE
)

# 2. Standard Separators (hyphens on own line, multiple newlines) for further splitting non-special text.
HYPHEN_SEP_PATTERN_STR = r'(?:^\s*---\s*$|^\s*--\s*$|^\s*-\s*$)'
MULTI_NEWLINE_SEP_PATTERN_STR = r'\n{3,}'
STANDARD_SEP_PATTERN = re.compile(
    f'(?:{HYPHEN_SEP_PATTERN_STR}|{MULTI_NEWLINE_SEP_PATTERN_STR})', re.MULTILINE)

# 3. Patterns for DETECTING features *within* final segments (to set meta flags)
DETECT_URL_ONLY_PATTERN = re.compile(URL_REGEX_STR, re.IGNORECASE)
DETECT_TEXT_COLON_URL_PATTERN = re.compile(
    # No ^ $ for detection within a segment
    r'[\w\s\(\)\[\].,!?\"-&\u2019\u201c\u201d;<>]+?:\s*' + URL_REGEX_STR,
    re.IGNORECASE
)

# --- Text Segmentation (Revised) ---


def split_text_into_segments(text_content):
    if not text_content:
        return []

    text_content = text_content.replace('\r\n', '\n')
    final_segments = []
    last_end = 0

    for match in TEXT_COLON_URL_LINE_PATTERN_FOR_FINDITER.finditer(text_content):
        # Part before this special match
        before_match_str = text_content[last_end:match.start()]
        # No strip here yet, pass to standard splitter which will handle it
        if before_match_str:  # Process only if there's content
            sub_segments_before = STANDARD_SEP_PATTERN.split(before_match_str)
            for sub_seg in sub_segments_before:
                sub_seg_stripped = sub_seg.strip()
                if sub_seg_stripped:
                    final_segments.append(sub_seg_stripped)

        # The matched <text>:<url> line itself (it's a special segment)
        special_segment = match.group(0).strip()  # group(0) is the whole match
        if special_segment:
            final_segments.append(special_segment)

        last_end = match.end()

    # Part after the last special match
    after_last_match_str = text_content[last_end:]
    # No strip here yet
    if after_last_match_str:  # Process only if there's content
        sub_segments_after = STANDARD_SEP_PATTERN.split(after_last_match_str)
        for sub_seg in sub_segments_after:
            sub_seg_stripped = sub_seg.strip()
            if sub_seg_stripped:
                final_segments.append(sub_seg_stripped)

    # Fallback: If no special <text>:<url> lines were found at all,
    # the whole content is one big "standard" part.
    # This condition (`last_end == 0`) means finditer had no matches.
    if last_end == 0 and text_content:  # text_content itself could be just whitespace
        original_text_stripped = text_content.strip()
        if original_text_stripped:  # Check if the original text wasn't just whitespace
            sub_segments_full = STANDARD_SEP_PATTERN.split(
                original_text_stripped)  # Apply to stripped text
            for sub_seg in sub_segments_full:
                sub_seg_stripped = sub_seg.strip()
                if sub_seg_stripped:
                    final_segments.append(sub_seg_stripped)

    return [seg for seg in final_segments if seg]


# --- Main Preprocessing Function (largely same as before, uses new split_text_into_segments) ---
def preprocess_for_textcat_multilabel(input_file_path, output_file_path, text_categories):
    processed_count = 0
    segment_count = 0

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
            open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line_idx, line in enumerate(infile):
            original_record_for_logging = {}  # For logging in case of error before full parse
            try:
                original_record = json.loads(line)
                original_record_for_logging = original_record  # Store for logging if error
                original_text_field = original_record.get("text")
                original_meta = original_record.get("meta", {})
                input_hash = original_record.get("_input_hash")
                task_hash = original_record.get("_task_hash")

                if not original_text_field:
                    logging.debug(
                        f"Line {line_idx+1}: Skipping record due to missing 'text' field.")
                    continue

                segments = split_text_into_segments(original_text_field)

                if not segments:
                    stripped_original = original_text_field.strip()
                    if stripped_original:
                        segments = [stripped_original]
                        logging.debug(
                            f"Line {line_idx+1}: Original text had no complex separators, using as single segment: '{stripped_original[:100]}...'")
                    else:
                        logging.debug(
                            f"Line {line_idx+1}: Skipping record as text resulted in no usable segments. Original: '{original_text_field[:100]}...'")
                        continue

                processed_count += 1

                for segment_idx, segment_text in enumerate(segments):
                    segment_count += 1
                    normalized_segment_text = normalize_text(segment_text)

                    if not normalized_segment_text:
                        logging.debug(
                            f"Line {line_idx+1}, Segment {segment_idx}: Skipped as it became empty after normalization. Original segment: '{segment_text[:100]}...'")
                        continue

                    pattern_matches_meta = {}
                    suggested_categories = []

                    if DETECT_URL_ONLY_PATTERN.search(normalized_segment_text):
                        pattern_matches_meta["pattern_has_any_link"] = True
                        if "IS_LINK_SHARE" in text_categories and "IS_LINK_SHARE" not in suggested_categories:
                            suggested_categories.append("IS_LINK_SHARE")

                    if DETECT_TEXT_COLON_URL_PATTERN.search(normalized_segment_text):
                        pattern_matches_meta["pattern_has_descriptive_link"] = True
                        if "IS_LINK_SHARE" in text_categories and "IS_LINK_SHARE" not in suggested_categories:
                            suggested_categories.append("IS_LINK_SHARE")
                        if "HAS_CALL_TO_ACTION_LINK" in text_categories and "HAS_CALL_TO_ACTION_LINK" not in suggested_categories:
                            suggested_categories.append(
                                "HAS_CALL_TO_ACTION_LINK")

                    prodigy_segment_record = {
                        "text": normalized_segment_text,
                        "meta": {
                            **original_meta,
                            "original_comment_text_hash": hash(original_text_field + str(line_idx)),
                            "segment_index": segment_idx,
                            "total_segments_from_original": len(segments)
                        }
                    }
                    prodigy_segment_record["meta"].update(pattern_matches_meta)
                    if suggested_categories:
                        prodigy_segment_record["meta"]["suggested_categories_by_pattern"] = suggested_categories

                    if input_hash is not None:
                        prodigy_segment_record["_input_hash"] = hash(
                            normalized_segment_text)
                    if task_hash is not None:
                        prodigy_segment_record["_task_hash"] = hash(
                            normalized_segment_text)

                    outfile.write(json.dumps(prodigy_segment_record) + '\n')

            except json.JSONDecodeError:
                logging.error(
                    f"Line {line_idx+1}: Malformed JSON, skipping line: {line.strip()}")
            except Exception as e:
                # Use original_record_for_logging to get text if parsing failed early
                error_text_preview = original_record_for_logging.get(
                    'text', 'N/A')[:100]
                # Set exc_info=True for full traceback
                logging.error(
                    f"Line {line_idx+1}: Error processing record: '{error_text_preview}...' - {e}", exc_info=True)

    logging.info(
        f"Successfully preprocessed {processed_count} original records into {segment_count} segments, saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess KEXP comments for Prodigy textcat_multilabel with advanced segmentation.")
    parser.add_argument(
        "input_jsonl_file", help="Path to input JSONL (e.g., from 00_extract_kexp_comments.py)")
    parser.add_argument("output_jsonl_file",
                        help="Path to output preprocessed JSONL for Prodigy")
    parser.add_argument(
        "--labels", help="Comma-separated list of text categories used in your Prodigy project (e.g., IS_LINK_SHARE,HAS_ARTIST_BIO). This helps the script suggest relevant categories.", default="")

    args = parser.parse_args()
    text_categories_list = [label.strip()
                            for label in args.labels.split(',') if label.strip()]

    preprocess_for_textcat_multilabel(
        args.input_jsonl_file, args.output_jsonl_file, text_categories_list)
