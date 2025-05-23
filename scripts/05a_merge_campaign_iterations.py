import argparse
import datetime
import json
import os
import sys
import yaml
from collections import defaultdict

# Assuming load_annotation_schema and update_dataset_registry might be shared
# For now, keep update_dataset_registry here, or move to a common util file later.

from scripts.utilities.common_utils import update_dataset_registry, load_jsonl, save_jsonl, load_annotation_schema


def main():
    parser = argparse.ArgumentParser(
        description="Merge annotation export files from the same campaign.")
    parser.add_argument("--campaign-name", required=True,
                        help="The name of the campaign being consolidated.")
    parser.add_argument("--export-files", required=True, nargs='+',
                        help="List of paths to Prodigy export JSONL files for this campaign.")
    parser.add_argument("--annotation-schema-version", required=True,
                        help="The target kexp_annotation_schema_v{X.Y}.yaml version these merged annotations should conform to.")
    parser.add_argument("--output-file", required=True,
                        help="Path for the consolidated campaign annotation file (e.g., data/campaign_annotated/{campaign_name}_asv{X.Y}_annotated_final.jsonl).")
    # Add --workspace-root argument for clarity on relative paths
    parser.add_argument("--workspace-root", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..')), help="Path to the workspace root.")

    args = parser.parse_args()

    print(f"Merging exports for campaign: {args.campaign_name}")
    print(f"Target schema version: {args.annotation_schema_version}")

    # 1. Load Data
    all_examples = []
    export_file_timestamps = {}
    for export_file in args.export_files:
        if not os.path.exists(export_file):
            print(f"Warning: Export file {export_file} not found. Skipping.")
            continue
        try:
            # Try to extract timestamp from filename (e.g., prodigy_dataset_name_export_{timestamp}.jsonl)
            # This is a basic assumption, might need refinement.
            filename = os.path.basename(export_file)
            timestamp_str = filename.split('_export_')[-1].split('.jsonl')[0]
            export_file_timestamps[export_file] = datetime.datetime.strptime(
                timestamp_str, '%Y%m%d%H%M%S')  # Assuming YYYYMMDDHHMMSS format
        except Exception as e:
            print(
                f"Warning: Could not parse timestamp from {export_file}. Using file modification time. Error: {e}")
            export_file_timestamps[export_file] = datetime.datetime.fromtimestamp(
                os.path.getmtime(export_file))

        print(f"Loading export file: {export_file}")
        examples = load_jsonl(export_file)
        for ex in examples:
            # Keep track of origin for conflict resolution
            ex['_source_file'] = export_file
        all_examples.extend(examples)

    if not all_examples:
        print("No valid examples found in export files. Exiting.")
        sys.exit(0)

    # 2. Group by Example
    # Using _input_hash if available and consistent, otherwise hash of text.
    # Prodigy usually adds _input_hash and _task_hash.
    # If text is identical but hashes differ (e.g. due to metadata), this might still treat them as different.
    # A more robust grouping might involve normalizing text content first.
    grouped_examples = defaultdict(list)
    for ex in all_examples:
        example_id = ex.get('_input_hash')
        if not example_id and 'text' in ex:
            # Fallback to hashing the text if no _input_hash
            # Consider a more robust hashing function if needed
            example_id = hash(ex['text'])
        elif not example_id:
            print(
                f"Warning: Example missing 'text' and '_input_hash', cannot group: {ex.get('_id', 'Unknown example')}")
            continue
        grouped_examples[example_id].append(ex)

    # 3. Merge Annotations for Each Example
    consolidated_annotations = []
    print("Consolidating annotations...")
    for example_id, versions in grouped_examples.items():
        if not versions:
            continue

        chosen_version = versions[0]  # Default to first one
        if len(versions) > 1:
            # Conflict Resolution: Prioritize by latest export file timestamp
            versions.sort(
                key=lambda v: export_file_timestamps[v['_source_file']], reverse=True)
            chosen_version = versions[0]
            print(
                f"Conflict for example ID {example_id}. Choosing version from {chosen_version['_source_file']} (latest). Spans: {len(chosen_version.get('spans', []))}")

        # Remove temporary source file tracking
        del chosen_version['_source_file']
        consolidated_annotations.append(chosen_version)

    # 4. Schema Alignment
    print(
        f"Performing schema alignment against v{args.annotation_schema_version}...")
    try:
        schema_path = os.path.join(args.workspace_root, "config", "schemas",
                                   f"kexp_annotation_schema_v{args.annotation_schema_version}.yaml")
        annotation_schema = load_annotation_schema(schema_path)
        valid_labels = {label_def['name']
                        for label_def in annotation_schema.get('labels', [])}
        if not valid_labels:
            print(
                f"Warning: No labels found in schema v{args.annotation_schema_version}. Cannot perform schema alignment for labels.")
            # Potentially exit or handle as an error depending on desired strictness
        else:
            print(
                f"Loaded {len(valid_labels)} valid labels from schema v{args.annotation_schema_version} for alignment.")

        aligned_annotations = []
        for ex in consolidated_annotations:
            # Ensure spans exist and is not None
            if 'spans' in ex and ex['spans'] is not None:
                original_span_count = len(ex['spans'])
                ex['spans'] = [span for span in ex['spans']
                               if span.get('label') in valid_labels]
                if len(ex['spans']) != original_span_count:
                    print(
                        f"Info: Example ID {ex.get('_input_hash', ex.get('text', 'Unknown example')[:30])} had spans filtered due to schema alignment. Original: {original_span_count}, New: {len(ex['spans'])}")
            aligned_annotations.append(ex)
        consolidated_annotations = aligned_annotations
        print("Schema alignment complete.")

    except FileNotFoundError as e:
        print(
            f"ERROR during schema alignment: {e} - Target schema file not found. Cannot align labels.")
        # Depending on strictness, you might want to sys.exit(1) or proceed with unaligned labels but a warning.
        print("Warning: Proceeding without label alignment due to missing schema.")
    except ValueError as e:
        print(
            f"ERROR during schema alignment: {e} - Invalid target schema file. Cannot align labels.")
        print("Warning: Proceeding without label alignment due to invalid schema.")
    except Exception as e:
        print(f"An unexpected ERROR occurred during schema alignment: {e}")
        print("Warning: Proceeding without label alignment due to unexpected error.")

    # 5. Write Output
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_jsonl(args.output_file, consolidated_annotations)
    print(f"Consolidated annotations saved to: {args.output_file}")

    # 6. Update Dataset Registry
    print("Updating dataset registry...")
    dataset_entry = {
        # Construct a logical name
        'prodigy_dataset_name': os.path.basename(args.output_file).replace('_annotated_final.jsonl', ''),
        'campaign_name': args.campaign_name,
        'creation_date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'status': 'merged_campaign',
        'input_examples_file': 'N/A',
        'ann_index_file': 'N/A',
        'prodigy_export_file': os.path.relpath(args.output_file, args.workspace_root),
        'annotation_schema_version': args.annotation_schema_version,
        'description': f"Consolidated annotations for campaign '{args.campaign_name}' from {len(args.export_files)} export(s).",
        'source_prodigy_datasets_merged': ";".join([os.path.basename(f) for f in args.export_files])
    }
    registry_file_path = os.path.join(
        args.workspace_root, "docs", "dataset_registry.csv")
    update_dataset_registry(registry_file_path, dataset_entry)

    print("Campaign iteration merging complete.")


if __name__ == "__main__":
    main()
