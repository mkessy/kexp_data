# Implementation Plan: Gold Standard Dataset Creation Workflow

## 1. Introduction

This document outlines the implementation plan for a revised workflow to create, manage, and version gold-standard datasets for SpanCat model training. The primary goals are to:

- Improve modularity in the annotation process by allowing focus on specific label subsets ("Campaigns").
- Establish clear version control for a unified annotation schema (definitions, guidelines, examples) and datasets.
- Enhance traceability and reproducibility of the gold data.
- Streamline the process of merging annotations from different efforts.
- Integrate an iterative, ANN-driven example selection process.
- Utilize YAML for a structured and consolidated annotation schema and for campaign configurations.

## 2. Configuration Management (YAML)

Core annotation schema (definitions, guidelines, examples) and campaign-specific configurations will be managed using YAML files, enabling clear structure, versioning, and easier programmatic access.

### 2.1. Consolidated Annotation Schema

- **Purpose:** To provide a single source of truth for all label definitions, comprehensive annotation guidelines, illustrative examples (with rationales), and edge case clarifications. This ensures consistent understanding for both human annotators and downstream modeling tasks.
- **Location:** `config/schemas/`
- **Structure:** Each version of the complete annotation schema will be in a separate file.
  - Filename: `kexp_annotation_schema_v{X.Y}.yaml` (e.g., `kexp_annotation_schema_v1.0.yaml`).
- **Versioning:** Semantic versioning (Major `X` for breaking changes, Minor `Y` for clarifications or additions to labels or guidelines).
- **Content (YAML Structure per file):**

  ```yaml
  schema_version: "1.0" # Matches the X.Y in filename
  date: "YYYY-MM-DD" # Date this schema version was finalized
  description: "Consolidated annotation schema for KEXP DJ comments. Contains all label definitions, detailed guidelines, and illustrative examples."

  general_annotation_rules:
    - rule: "Annotate based on explicit information in the text."
    # ... other global rules

  workflow_overview:
    segmentation_notes: "Comments are pre-segmented. Annotate within these segments."
    # ... other global workflow notes

  labels:
    - name: "ARTIST_TAG"
      description: "The name of a musical artist, band, or group."
      guidance: |
        - Annotate full names of artists, bands, or groups.
        - Include common aliases if directly mentioned as such and part of the name being identified.
        - Do not annotate possessive forms (e.g., "artist's song").
        - **Process Note**: If an artist has multiple common spellings, prefer the one most frequently used by KEXP sources.
        - **Edge Case**: For collaborations like "Artist A ft. Artist B", tag both "Artist A" and "Artist B" individually.
      examples: # List of positive examples
        - text: "And that was 'Come As You Are' by Nirvana, a classic from their album 'Nevermind'."
          results: ["Nirvana"]
          reason: "Direct mention of the band name 'Nirvana'."
        # ... more positive examples
    # ... all other labels follow this comprehensive structure
  ```

- **Management:**
  - A changelog section within each schema file or a separate `annotation_schema_changelog.md` should track changes between versions.

### 2.2. Campaign Configurations (Formerly 2.3)

- **Location:** `config/campaign_configs/`
- **Structure:** One YAML file per defined annotation campaign.
  - Filename: `{campaign_name}.yaml` (e.g., `artist_biographical_campaign.yaml`).
- **Content (YAML Structure per file):**

  ```yaml
  campaign_name: "artist_biographical_details_v1"
  description: "Focus on annotating artist biographical information and associated tags."

  # Version of the consolidated annotation schema this campaign adheres to
  annotation_schema_version: "1.0" # e.g., points to kexp_annotation_schema_v1.0.yaml

  target_labels_for_ui:
    - "ARTIST_TAG"
    - "ARTIST_BIO_SPAN"
    # ...

  source_data_specification:
    type: "jsonl_file"
    path: "data/source_material/unannotated_batch1.jsonl"

  pre_labeling:
    use_script_04: true

  ann_bootstrapping:
    sentence_transformer_model: "all-MiniLM-L6-v2"
  ```

## 3. Directory Structure (Revised)

```
kexp_data/
├── config/
│   ├── schemas/
│   │   ├── kexp_annotation_schema_v1.0.yaml
│   │   └── kexp_annotation_schema_v1.1.yaml
│   ├── campaign_configs/
│   │   ├── artist_biographical_campaign.yaml
│   │   └── release_info_campaign.yaml
│   ├── prelabel_config.py         # General config for 04_prelabel_for_prodigy.py
│   └── gazetteers/                # Existing
│   └── labels.txt                 # Master list of all labels (can be generated from the annotation schema)
├── data/
│   ├── raw_kexp_data/             # Raw DB extracts (existing)
│   ├── source_material/           # Cleaned/filtered raw data ready for campaign input
│   ├── processed_examples/        # Per-Prodigy-dataset input
│   ├── prodigy_db/                # Prodigy SQLite databases
│   ├── prodigy_exports/           # Raw exports from Prodigy
│   ├── campaign_annotated/        # Consolidated annotations per campaign
│   └── annotated/                 # Final merged gold standard
│   └── models/                    # Trained models
├── docs/
│   ├── implementation_plan_gold_standard_workflow.md # This file
│   ├── dataset_registry.csv       # Or dataset_registry.yaml
│   ├── (existing .md files like annotation_schema.md - content now migrated to YAML schema)
├── prodigy_recipes/
│   └── prelabel_ent_spans_manual.py
├── scripts/
│   ├── 00_extract_kexp_comments.py
│   ├── 00b_segment_and_normalize_comments.py
│   ├── 04_prelabel_for_prodigy.py
│   ├── run_prodigy_campaign.py        # New
│   ├── 05a_merge_campaign_iterations.py # New
│   ├── 05b_merge_campaigns_to_gold.py   # New
│   └── utilities/
│       ├── build_ann_index.py           # New/Refined
│       ├── validate_schema.py           # Renamed from validate_definitions.py
│       └── examine_prodigy_export.py
├── src/
│   └── kexp_processing_utils/
│       └── normalization.py
│       └── comment_parser.py
├── .env
└── README.md
```

## 4. Dataset Registry

A central log to track the lineage and status of all generated datasets.

- **Fields (CSV example - updated to reflect schema versioning):**
  ```csv
  prodigy_dataset_name,campaign_name,creation_date,status,input_examples_file,ann_index_file,prodigy_export_file,annotation_schema_version,description,source_prodigy_datasets_merged
  artist_bio_details_v1_asv1.0_iter1_20231115,artist_biographical_details_v1,2023-11-15,annotated,processed_examples/artist_bio_details_v1_asv1.0_iter1_20231115_examples.jsonl,processed_examples/artist_bio_details_v1_asv1.0_iter1_20231115_examples.ann_index,prodigy_exports/artist_bio_details_v1_asv1.0_iter1_20231115_export_20231116.jsonl,1.0,"Initial bootstrap for artist bio",
  # ...
  campaign_annotated_artist_bio_details_v1_asv1.0,artist_biographical_details_v1,2023-11-20,merged_campaign,N/A,N/A,campaign_annotated/artist_bio_details_v1_asv1.0_annotated_final.jsonl,1.0,"Consolidated artist bio annotations for v1 campaign","artist_bio_details_v1_asv1.0_iter1_20231115;artist_bio_details_v1_asv1.0_iter2_20231118"
  # ...
  gold_standard_v1_asv1.0,N/A,2023-11-25,gold_created,N/A,annotated/gold_spancat_asv1.0_v1.ann_index,annotated/gold_spancat_asv1.0_v1.jsonl,1.0,"First gold standard release","campaign_annotated_artist_bio_details_v1_asv1.0;campaign_annotated_release_info_v1_asv1.0"
  ```
  (Removed `label_definitions_version` and `annotation_guidelines_version`, replaced with `annotation_schema_version`)

## 5. Script Specifications

### 5.1. `scripts/run_prodigy_campaign.py` (New)

- **Core Logic:**
  1.  **Load Configurations:**
      - Read the specified campaign config YAML.
      - Read the referenced `kexp_annotation_schema_v{X.Y}.yaml`.
  2.  **Determine Prodigy Dataset Name:**
      - (logic unchanged, but constructed name might use `_asv{schema_ver}` instead of `_ldv..._agv...`)
      - Example: `{campaign_name}_asv{schema_ver}_iter{iter_num}_{iter_desc}_{date}`.
  3.  (Prepare Input Stream & ANN Index - logic largely unchanged)
  4.  **Configure Prodigy UI Labels:**
      - From campaign config, get `target_labels_for_ui`.
      - From the loaded annotation schema, get full details (description, examples) for these target labels to potentially pass to or display within Prodigy (if recipe supports it).
  5.  (Launch Prodigy - logic largely unchanged)
  6.  (Update Dataset Registry - logic largely unchanged, uses new schema version field)

### 5.2. `scripts/05a_merge_campaign_iterations.py` (New)

- **Inputs (Command Line Arguments):**
  - (Removed `--label-definitions-version` and `--annotation-guidelines-version`)
  - Add `--annotation-schema-version`: The target `kexp_annotation_schema_v{X.Y}.yaml` version these merged annotations should conform to.
  - Update `--output-file` example: `data/campaign_annotated/{campaign_name}_asv{X.Y}_annotated_final.jsonl`.
- **Core Logic:** 4. **Schema Alignment (Optional but Recommended):** - Verify that all labels in the merged spans conform to the specified `--annotation-schema-version`.

### 5.3. `scripts/05b_merge_campaigns_to_gold.py` (New)

- **Inputs (Command Line Arguments):**
  - (Removed `--target-label-definitions-version` and `--target-annotation-guidelines-version`)
  - Add `--target-annotation-schema-version`: The primary `kexp_annotation_schema_v{X.Y}.yaml` version this gold set aims to conform to.
  - Update `--output-base-filename` example for final name: `gold_spancat_asv{X.Y}_{gold_version_tag}.jsonl`.
- **Core Logic:** 2. **Identify Annotation Schema Versions:** For each input file, determine the `annotation_schema_version` it was created with. 4. **Merge Annotations for Each Example - Conflict Resolution:** - Prioritize annotation from campaign file using the _latest `annotation_schema_version`_ for that label. - Filter out spans whose labels are not in the `--target-annotation-schema-version`. 6. **Final Validation:** - Validate all spans against the `--target-annotation-schema-version`.

### 5.4. `scripts/utilities/build_ann_index.py` (New/Refined)

(No changes directly related to schema file structure)

### 5.5. `scripts/utilities/validate_schema.py` (Renamed from `validate_definitions.py`)

- **Purpose:** Validate the structure, content, and versioning consistency of the consolidated `kexp_annotation_schema_v{X.Y}.yaml` files and `campaign_configs`.
- **Inputs (Command Line Arguments):**
  - `--file-type`: Type of file to validate (`schema`, `campaign_conf`).
  - `--path`: Path to a specific YAML file or a directory.
- **Core Logic:**
  - **Common Checks:** Valid YAML syntax.
  - **Annotation Schema (`schema`):**
    - Presence of `schema_version`, `date`, `description`, `general_annotation_rules`, `labels` keys.
    - `schema_version` matches filename pattern.
    - Each label has `name`, `description`, `guidance`, `examples` (list of objects with `text`, `results`, `reason`).
  - **Campaign Configurations (`campaign_conf`):**
    - Presence of `campaign_name`, `description`, `annotation_schema_version`, `target_labels_for_ui`, `source_data_specification`.
    - Check if referenced `annotation_schema_version` file exists.
    - Check if all labels in `target_labels_for_ui` are defined in the referenced schema file.

### 5.6. Existing Scripts (`04_prelabel_for_prodigy.py`)

(No changes directly related to schema file structure, but its config in `prelabel_config.py` might eventually reference the consolidated schema for label lists if needed).

## 6. Workflow Overview (Step-by-Step)

1.  **Setup & Definition (YAML):**
    - Define/update `config/schemas/kexp_annotation_schema_v{X.Y}.yaml`. Validate with `validate_schema.py`.
2.  **Campaign Planning:**
    - Define an annotation campaign by creating `config/campaign_configs/{campaign_name}.yaml`. Specify target labels, source data, and the `annotation_schema_version` it adheres to. Validate.
      (Steps 3-10 will need minor adjustments in filenames and version parameters to reflect `annotation_schema_version` or `asv{X.Y}` instead of separate label def/guideline versions, similar to changes in Section 5 and 4).

## 7. Versioning Details Summary

- **Consolidated Annotation Schema:** File-based semantic versioning (`config/schemas/kexp_annotation_schema_vX.Y.yaml`).
- (Removed Label Definitions and Annotation Guidelines as separate items)
- **Campaign Configs:** Named by campaign; internal field links to specific `annotation_schema_version`.
- **Prodigy Datasets:** Named dynamically, including campaign name, `annotation_schema_version`, iteration, and date.
- **Processed Examples/ANN Indexes:** Named identically to their corresponding Prodigy dataset.
- **Prodigy Exports:** Named after Prodigy dataset with a timestamp.
- **Consolidated Campaign Data:** Named by campaign, including `annotation_schema_version`.
- **Gold Standard Data:** Named with base, `annotation_schema_version`, and a gold version tag.
- **Models:** Filenames should reflect the version of the gold data and schema they were trained on.

## 8. Migration of Existing Documentation

- Review existing `.md` files in `docs/` (e.g., `annotation_schema.md`, `phase_a_ner_annotation.md`).
- All relevant label definitions, guidance, and examples from these files should now be migrated and consolidated into the new YAML-based `config/schemas/kexp_annotation_schema_vX.Y.yaml` structure.
- The old `.md` files can then be archived or kept as supplementary material if they contain information not suited for the YAML format (e.g., extensive prose on project history or very high-level conceptual discussions not tied to specific label mechanics).

This detailed plan should provide a solid foundation for implementing the new workflow.
