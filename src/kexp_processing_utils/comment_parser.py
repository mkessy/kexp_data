import re
from typing import List, Dict, Iterator, Any
import logging

# Configure basic logging for this module, if needed for debugging parsing.
# Keep it minimal as Prodigy will handle main recipe logging.
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')

# --- Text Normalization (originally from 00c_segment_and_normalize_comments.py) ---
# This is applied to *each segment* after splitting.


def normalize_text_segment(text_content: str) -> str:
    if text_content is None:
        return ""
    text_str = str(text_content)
    text_str = text_str.replace("—", " ")  # em-dash U+2014
    text_str = text_str.replace("–", " ")  # en-dash U+2013
    text_str = re.sub(r'\s+', ' ', text_str)
    text_str = text_str.strip()
    return text_str


# --- Regex Patterns (originally from 00c_segment_and_normalize_comments.py) ---
URL_REGEX_STR = r'(https?://[^\s/$.?#].[^\s]*|[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|ly|eu|info|biz|ws|us|ca|uk|au|de|jp|fr|ch|fm|tv|me|sh|stream|live|watch|listen|download|video|audio|pics|photo|img|image|gallery|news|blog|shop|store|app|co|info|online|site|website|xyz|club|dev|page|link|art|bandcamp|soundcloud|spotify|youtube|youtu\.be|vimeo|tiktok|instagram|facebook|twitter|patreon|kexp)(?:/(?:[^\s()<>]+|(?:\([^\s()<>]+(?:\([^\s()<>]+\))*\)))*)?(?:\?[^\s]*)?)'

# TEXT_COLON_URL_LINE_PATTERN_FOR_FINDITER is not strictly needed if we simplify the splitting logic as discussed
# but can be kept if we want to refine segment identification later.
# For now, the primary split is by STANDARD_SEP_PATTERN.
TEXT_COLON_URL_LINE_PATTERN_FOR_FINDITER = re.compile(
    r'^[ \t]*[\w\s\(\)\[\].,!?"-&\u2019\u201c\u201d;<>]+?:\s*' +
    URL_REGEX_STR + r'[ \t]*$',
    re.IGNORECASE | re.MULTILINE
)

HYPHEN_SEP_PATTERN_STR = r'(?:^\s*---\s*$|^\s*--\s*$|^\s*-\s*$)'
# Changed to catch one or more empty lines
MULTI_NEWLINE_SEP_PATTERN_STR = r'\n{2,}'
STANDARD_SEP_PATTERN = re.compile(
    f'(?:{HYPHEN_SEP_PATTERN_STR}|{MULTI_NEWLINE_SEP_PATTERN_STR})', re.MULTILINE
)

# --- Text Segmentation (Revised for Simplicity and Robustness) ---


def _split_raw_text_into_segments(raw_text_content: str) -> List[str]:
    """
    Splits raw text into segments based on standard separators (empty lines, hyphens).
    Strips and ensures no empty segments are returned.
    """
    if not raw_text_content or not raw_text_content.strip():
        return []

    # Standardize line endings first
    text_content = raw_text_content.replace('\r\n', '\n')  # Corrected replace

    # Primary split using standard separators
    initial_potential_segments = STANDARD_SEP_PATTERN.split(text_content)

    final_segments = []
    for potential_segment in initial_potential_segments:
        stripped_segment = potential_segment.strip()

        if stripped_segment:  # Only add non-empty segments
            final_segments.append(stripped_segment)

    return final_segments  # No need for a final list comprehension if already filtering


# --- Main Parsing Function for this Module ---
def parse_comment_to_prodigy_tasks(
    raw_comment_text: str,
    original_db_meta: Dict[str, Any]
) -> Iterator[Dict[str, Any]]:
    """
    Takes a raw comment string and its original database metadata,
    segments the comment using _split_raw_text_into_segments,
    normalizes each segment, and yields Prodigy-ready
    task dictionaries for each valid segment.
    Segments consisting *only* of a URL are filtered out.
    The original full comment text is added to the task metadata.
    """
    if not raw_comment_text or not raw_comment_text.strip():
        return

    segments = _split_raw_text_into_segments(raw_comment_text)

    if not segments:
        # logging.debug(f"Comment ID {original_db_meta.get('play_id', 'N/A')}: _split_raw_text_into_segments resulted in no segments. Raw text: '{raw_comment_text[:100]}...'")
        return  # Yield nothing if no segments were produced

    total_segments = len(segments)
    for segment_idx, segment_text in enumerate(segments):
        normalized_segment = normalize_text_segment(
            segment_text)  # segment_text is already stripped

        if not normalized_segment:
            # logging.debug(f"Comment ID {original_db_meta.get('play_id', 'N/A')}, original segment {segment_idx} ('{segment_text[:50]}...') became empty after normalization.")
            continue

        # Filter out segments that are ONLY a URL
        if re.fullmatch(URL_REGEX_STR, normalized_segment, re.IGNORECASE):
            # logging.debug(f"Comment ID {original_db_meta.get('play_id', 'N/A')}, segment {segment_idx} ('{normalized_segment[:50]}...') is URL-only, skipping.")
            continue

        task_meta: Dict[str, Any] = {
            **original_db_meta,
            "segment_index_in_comment": segment_idx,
            "total_segments_from_comment": total_segments,
            "original_full_comment_text": raw_comment_text  # Add original full comment
        }

        yield {
            "text": normalized_segment,
            "meta": task_meta
        }
