#!/usr/bin/env python3
import argparse
import json
import os
import random
import logging
from collections import defaultdict

try:
    import prodigy
    prodigy_available = True
except ImportError:
    prodigy_available = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_jsonl(input_file):
    """Load records from a JSONL file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed JSON: {line[:100]}...")


def load_prodigy_dataset(dataset_name):
    """Load records from a Prodigy dataset. Only works if Prodigy is installed."""
    if not prodigy_available:
        raise ImportError(
            "Prodigy is not installed. Please install Prodigy to use this feature.")
    from prodigy.components.db import connect
    db = connect()
    for eg in db.get_dataset_examples(dataset_name):
        yield eg


def get_label(record, label_field=None):
    """Extract label(s) from a record, supporting several common formats, including Prodigy textcat-multi."""
    # Prodigy textcat-multi: 'accept' is a list of label ids
    if 'accept' in record and isinstance(record['accept'], list):
        return record['accept']
    if label_field:
        value = record.get(label_field)
    else:
        # Try common fields
        for field in ['label', 'labels', 'spans']:
            if field in record:
                value = record[field]
                break
        else:
            value = None
    # Normalize to a list of labels
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        # For spans, extract 'label' from each dict
        if value and isinstance(value[0], dict) and 'label' in value[0]:
            return [span['label'] for span in value if 'label' in span]
        return value
    return []


def sample_by_label(records, samples_per_label, label_field=None):
    """Group records by label and sample N per label."""
    label_to_records = defaultdict(list)
    for rec in records:
        labels = get_label(rec, label_field)
        for label in labels:
            label_to_records[label].append(rec)
    sampled = []
    for label, recs in label_to_records.items():
        n = min(samples_per_label, len(recs))
        sampled_recs = random.sample(recs, n)
        sampled.extend(sampled_recs)
        logging.info(
            f"Sampled {n} records for label '{label}' (out of {len(recs)} available).")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Sample annotation data by label from a JSONL file or Prodigy dataset for inspection.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str,
                       help='Path to input JSONL file.')
    group.add_argument('--prodigy-dataset', type=str,
                       help='Name of Prodigy dataset to sample from.')
    parser.add_argument('--label-field', type=str, default=None,
                        help="Field containing the label(s). Defaults to 'label', 'labels', or 'spans'.")
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional: Path to output sampled JSONL. If not set, prints to stdout.')
    parser.add_argument('--samples-per-label', type=int, default=5,
                        help='Number of samples to draw per label.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.input_file:
        records = list(load_jsonl(args.input_file))
    else:
        records = list(load_prodigy_dataset(args.prodigy_dataset))

    logging.info(f"Loaded {len(records)} records.")

    sampled = sample_by_label(
        records, args.samples_per_label, args.label_field)
    logging.info(f"Total sampled records: {len(sampled)}")

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for rec in sampled:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        logging.info(f"Wrote sampled records to {args.output_file}")
    else:
        for rec in sampled:
            print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
