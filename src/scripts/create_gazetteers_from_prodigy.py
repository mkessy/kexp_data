import json
import os
import pathlib
import logging
import argparse
from kexp_processing_utils.normalization import normalize_text_for_gazetteer, clean_term, is_well_formed

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_prodigy_annotations(input_file_path, target_labels):
    """
    Processes annotations from a Prodigy JSONL file.
    Extracts, cleans, normalizes, and deduplicates terms for specified labels.
    """
    extracted_entities_by_label = {label: [] for label in target_labels}
    # For deduplication based on normalized form, keeping the first encountered 'text'
    # {label: {normalized_form: original_text_object}}
    unique_terms_tracker = {label: {} for label in target_labels}

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    annotation = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON line {line_number} in {input_file_path}")
                    continue

                if "text" not in annotation or "spans" not in annotation:
                    # logging.debug(f"Skipping line {line_number} due to missing 'text' or 'spans' field.")
                    continue

                original_text_content = annotation["text"]

                for span in annotation.get("spans", []):
                    span_label = span.get("label")
                    if span_label in target_labels:
                        start, end = span["start"], span["end"]
                        raw_term_text = original_text_content[start:end]

                        if not raw_term_text or not str(raw_term_text).strip():
                            continue

                        cleaned_original_term = clean_term(str(raw_term_text))

                        if cleaned_original_term and is_well_formed(cleaned_original_term):
                            normalized_match_term = normalize_text_for_gazetteer(
                                cleaned_original_term)
                            if normalized_match_term:
                                term_data = {
                                    "text": cleaned_original_term,
                                    "normalized": normalized_match_term,
                                    "label": span_label
                                }
                                # Add to tracker for deduplication by normalized form
                                if normalized_match_term not in unique_terms_tracker[span_label]:
                                    unique_terms_tracker[span_label][normalized_match_term] = term_data
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing file {input_file_path}: {e}")
        raise

    # Convert tracked unique terms back to list format for each label
    for label in target_labels:
        extracted_entities_by_label[label] = list(
            unique_terms_tracker[label].values())
        logging.info(
            f"Found {len(extracted_entities_by_label[label])} unique normalized terms for label {label}.")

    return extracted_entities_by_label


def write_gazetteers(processed_entities_by_label, output_dir_path_str):
    """
    Writes the processed and deduplicated entities to gazetteer files.
    One JSONL file is created per label in the output directory.
    """
    output_dir = pathlib.Path(output_dir_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir.resolve()}")

    for label, items in processed_entities_by_label.items():
        if not items:  # Skip creating empty files if no items for a label
            logging.info(
                f"No items found for label {label}. Skipping gazetteer file creation.")
            continue

        output_file_path = output_dir / f"{label.lower()}_gazetteer.jsonl"
        count = 0
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for item in items:
                    # Using "pattern" key for PhraseMatcher compatibility, "text" for original
                    f.write(json.dumps(
                        {"label": item["label"], "text": item["text"], "pattern": item["normalized"]}) + '\n')
                    count += 1
            logging.info(f"Wrote {count} entries to {output_file_path}")
        except IOError as e:
            logging.error(f"Could not write to file {output_file_path}: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while writing {output_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generates gazetteers from Prodigy annotated JSONL data."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the Prodigy JSONL input file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="gazetteers",
        help="Path to the directory where gazetteer files will be saved (default: 'gazetteers')."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        required=True,
        help="A list of labels to generate gazetteers for (e.g., GENRE_TAG ROLE_TAG)."
    )
    args = parser.parse_args()

    logging.info(f"Starting gazetteer generation from Prodigy annotations.")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Target labels: {args.labels}")

    # Placeholder for core logic
    # process_and_write_gazetteers(args.input_file, args.output_dir, args.labels)
    try:
        processed_entities = process_prodigy_annotations(
            args.input_file, args.labels)
        # Placeholder for writing logic
        # write_gazetteers(processed_entities, args.output_dir, args.labels)
        # logging.info(f"Processed entities: { {k: len(v) for k, v in processed_entities.items()} }") # Temp log
        write_gazetteers(processed_entities, args.output_dir)

    except FileNotFoundError:
        logging.error(f"Exiting due to file not found: {args.input_file}")
        return  # Exit if file not found
    except Exception as e:
        logging.error(f"An critical error occurred during processing: {e}")
        return  # Exit on other critical errors

    logging.info("Gazetteer generation finished successfully.")


if __name__ == "__main__":
    main()
