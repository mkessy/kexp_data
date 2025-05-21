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
