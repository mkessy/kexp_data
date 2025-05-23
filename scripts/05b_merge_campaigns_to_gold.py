import argparse
import datetime
import json
import os
import sys
import yaml
from collections import defaultdict

# Assuming load_annotation_schema and update_dataset_registry might be shared
# For now, keep update_dataset_registry here, or move to a common util file later.

from scripts.utilities.common_utils import load_annotation_schema, update_dataset_registry, load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Merge consolidated campaign annotations into a gold standard dataset.")
    parser.add_argument("--campaign-annotated-files", required=True,
                        nargs='+', help="List of paths to campaign_annotated/*.jsonl files.")
    parser.add_argument("--gold-version-tag", required=True,
                        help="A string tag for this version of the gold standard (e.g., 'v1.0_alpha', 'v1.0').")
    parser.add_argument("--target-annotation-schema-version", required=True,
                        help="The primary kexp_annotation_schema_v{X.Y}.yaml version this gold set aims to conform to.")
    parser.add_argument("--output-base-filename", required=True,
                        help="Base for the output gold file (e.g., 'gold_spancat'). Final name will include schema and gold versions.")
    parser.add_argument("--workspace-root", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..')), help="Path to the workspace root.")
    # parser.add_argument("--build-ann-index", action="store_true", help="Optionally build an ANN index for the created gold set.") # TODO: Implement ANN index building call

    args = parser.parse_args()

    print(f"Creating gold standard version: {args.gold_version_tag}")
    print(f"Target schema version: {args.target_annotation_schema_version}")

    # Load target annotation schema to get valid labels
    schema_path = os.path.join(args.workspace_root, "config", "schemas",
                               f"kexp_annotation_schema_v{args.target_annotation_schema_version}.yaml")
    target_schema = load_annotation_schema(schema_path)
    valid_labels_from_target_schema = {
        label_def['name'] for label_def in target_schema['labels']}
    print(
        f"Loaded target schema. Valid labels: {len(valid_labels_from_target_schema)}")

    # 1. Load Data from campaign_annotated files
    all_examples = []
    campaign_file_schema_versions = {}
    for ca_file in args.campaign_annotated_files:
        if not os.path.exists(ca_file):
            print(
                f"Warning: Campaign annotated file {ca_file} not found. Skipping.")
            continue

        # Infer schema version from filename (e.g., {campaign_name}_asv{X.Y}_annotated_final.jsonl)
        try:
            filename = os.path.basename(ca_file)
            schema_ver_str = filename.split(
                '_asv')[-1].split('_annotated_final.jsonl')[0]
            campaign_file_schema_versions[ca_file] = schema_ver_str
        except Exception as e:
            print(
                f"Warning: Could not parse schema version from filename {ca_file}. This might affect conflict resolution. Error: {e}")
            # Default to a low version if unparsable
            campaign_file_schema_versions[ca_file] = "0.0"

        print(
            f"Loading campaign annotated file: {ca_file} (schema version inferred: {campaign_file_schema_versions[ca_file]})")
        examples = load_jsonl(ca_file)
        for ex in examples:
            # Track origin for conflict resolution
            ex['_source_campaign_file'] = ca_file
        all_examples.extend(examples)

    if not all_examples:
        print("No valid examples found in campaign annotated files. Exiting.")
        sys.exit(0)

    # 2. Group by Example ID
    grouped_examples = defaultdict(list)
    for ex in all_examples:
        # Assuming _input_hash is the consistent ID
        example_id = ex.get('_input_hash')
        if not example_id and 'text' in ex:  # Fallback if needed
            example_id = hash(ex['text'])
        elif not example_id:
            print(
                f"Warning: Example missing 'text' and '_input_hash', cannot group: {ex.get('_id', 'Unknown example')}")
            continue
        grouped_examples[example_id].append(ex)

    # 3. Merge Annotations for Each Example
    gold_standard_annotations = []
    print("Merging annotations for gold standard...")

    for example_id, example_versions in grouped_examples.items():
        if not example_versions:
            continue

        # Master version of the example (text, metadata). Take the first one for now.
        # More sophisticated merging of text/metadata could be done if necessary.
        # Start with a copy of the first instance
        merged_example = example_versions[0].copy()
        del merged_example['_source_campaign_file']
        merged_example['spans'] = []

        spans_by_label = defaultdict(list)

        for ex_version in example_versions:
            source_file = ex_version['_source_campaign_file']
            for span in ex_version.get('spans', []):
                # Only consider spans whose labels are in the *target* schema
                if span['label'] not in valid_labels_from_target_schema:
                    # print(f"Info: Label '{span['label']}' in {source_file} for example {example_id} not in target schema v{args.target_annotation_schema_version}. Skipping span.")
                    continue
                spans_by_label[span['label']].append({
                    'span_data': span,
                    'source_file': source_file,
                    'source_schema_version': campaign_file_schema_versions.get(source_file, "0.0")
                })

        final_spans_for_example = []
        for label, span_occurrences in spans_by_label.items():
            if not span_occurrences:
                continue

            chosen_span_info = span_occurrences[0]  # Default
            if len(span_occurrences) > 1:
                # Conflict Resolution: Same label annotated for the same text snippet from different campaigns.
                # Prioritize the one from the campaign file that used the latest schema version for that label.
                span_occurrences.sort(
                    key=lambda s_info: s_info['source_schema_version'], reverse=True)
                chosen_span_info = span_occurrences[0]
                # Log this conflict and resolution
                print(
                    f"Conflict for label '{label}' in example ID {example_id}. Multiple sources. Choosing from {chosen_span_info['source_file']} (schema v{chosen_span_info['source_schema_version']}).")
            final_spans_for_example.append(chosen_span_info['span_data'])

        # 4. Deduplicate Spans within the example (after conflict resolution)
        # A span is defined by start, end, label. (token_start, token_end too if present)
        unique_spans_for_example = []
        seen_span_tuples = set()
        for span in final_spans_for_example:
            # Add token_start/end if used
            span_tuple = (span['start'], span['end'], span['label'])
            if span_tuple not in seen_span_tuples:
                unique_spans_for_example.append(span)
                seen_span_tuples.add(span_tuple)

        merged_example['spans'] = unique_spans_for_example
        gold_standard_annotations.append(merged_example)

    # 5. Final Validation (Placeholder)
    # - Validate all spans against the --target-annotation-schema-version (already partially done by filtering)
    # - Perform consistency checks (e.g., no overlapping spans of incompatible types, if such rules exist in schema).
    print(
        f"Final validation against schema v{args.target_annotation_schema_version} (Placeholder)...")

    # 6. Write Output
    output_filename = f"{args.output_base_filename}_asv{args.target_annotation_schema_version}_{args.gold_version_tag}.jsonl"
    output_path = os.path.join(
        args.workspace_root, "data", "annotated", output_filename)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_jsonl(output_path, gold_standard_annotations)
    print(f"Gold standard dataset saved to: {output_path}")

    # Optionally build ANN index (placeholder)
    # if args.build_ann_index:
    # print("Building ANN index for gold set (Placeholder)...")
    # Call scripts/utilities/build_ann_index.py --input-jsonl {output_path} ...
    ann_index_path_for_registry = "N/A"

    # 7. Update Dataset Registry
    print("Updating dataset registry...")
    dataset_entry = {
        # Logical name for gold set
        'prodigy_dataset_name': os.path.basename(output_path).replace('.jsonl', ''),
        'campaign_name': 'N/A',  # Gold sets are not directly a campaign
        'creation_date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'status': 'gold_created',
        'input_examples_file': 'N/A',  # Gold set is an output, not an input in this context
        'ann_index_file': ann_index_path_for_registry,  # Path to .ann_index if created
        'prodigy_export_file': os.path.relpath(output_path, args.workspace_root),
        'annotation_schema_version': args.target_annotation_schema_version,
        'description': f"Gold standard dataset version '{args.gold_version_tag}', schema v{args.target_annotation_schema_version}. Merged from {len(args.campaign_annotated_files)} campaign file(s).",
        'source_prodigy_datasets_merged': ";".join([os.path.basename(f) for f in args.campaign_annotated_files])
    }
    registry_file_path = os.path.join(
        args.workspace_root, "docs", "dataset_registry.csv")
    update_dataset_registry(registry_file_path, dataset_entry)

    print("Gold standard dataset creation complete.")


if __name__ == "__main__":
    main()
