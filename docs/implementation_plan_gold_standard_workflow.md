# Implementation Plan: Gold Standard Dataset Creation Workflow

## 1. Introduction

This document outlines the implementation plan for a revised workflow to create, manage, and version gold-standard datasets for SpanCat model training. The primary goals are to:

- Improve modularity in the annotation process by allowing focus on specific label subsets ("Campaigns").
- Establish clear version control for label definitions, annotation guidelines, and datasets.
- Enhance traceability and reproducibility of the gold data.
- Streamline the process of merging annotations from different efforts.
- Integrate an iterative, ANN-driven example selection process.
- Utilize YAML for structured configuration files (label definitions, guidelines, campaign configurations).

## 2. Configuration Management (YAML)

All core definitions and configurations will be managed using YAML files, enabling clear structure, versioning, and easier programmatic access.

### 2.1. Label Definitions

- **Location:** `config/label_definitions/`
- **Structure:** Each version of label definitions will be in a separate file.
  - Filename: `label_definitions_v{X.Y}.yaml` (e.g., `label_definitions_v1.0.yaml`, `label_definitions_v1.1.yaml`).
- **Versioning:** Semantic versioning (Major `X` for breaking changes in meaning, Minor `Y` for clarifications or additions).
- **Content (YAML Structure per file):**
  ```yaml
  version: "1.0" # Matches the X.Y in filename
  # Date this version was finalized
  date: "YYYY-MM-DD"
  description: "Initial set of label definitions for KEXP DJ comments."
  labels:
    - name: "ARTIST_TAG"
      description: "Proper name of a musical artist or band."
      guidance: |
        - Annotate full names.
        - Include common aliases if directly mentioned as such.
        - Do not annotate possessive forms (e.g., "artist's song").
      examples:
        positive:
          - text_snippet: "Nirvana played last night."
            rationale: "Direct mention of band name."
          - text_snippet: "Tune from an artist, Taylor Swift."
            rationale: "Artist name clearly identified."
        negative:
          - text_snippet: "The show was nirvanic."
            rationale: "Adjective, not the band name."
          - text_snippet: "She's a great artist."
            rationale: "Generic reference, not a specific name."
    - name: "ALBUM_TAG"
      # ... similar structure ...
    - name: "ARTIST_BIO_SPAN"
      description: "Span of text containing biographical information about an artist."
      guidance: |
        - Should cover sentences or clauses primarily discussing an artist's history, background, or significant life events.
        - Avoid including general music critique unless it's tied to a biographical detail.
      examples:
        # ...
    # ... other 20+ labels
  ```
- **Management:**
  - A changelog section within each file or a separate `label_definitions_changelog.md` should track changes between versions.

### 2.2. Annotation Guidelines

- **Location:** `config/annotation_guidelines/`
- **Structure:** Each version of general annotation guidelines will be in a separate file.
  - Filename: `annotation_guidelines_v{A.B}.yaml` (e.g., `annotation_guidelines_v1.0.yaml`).
- **Versioning:** Semantic versioning.
- **Content (YAML Structure per file):**

  ```yaml
  version: "1.0" # Matches the A.B in filename
  date: "YYYY-MM-DD"
  description: "General annotation guidelines for KEXP project."
  # Link to the specific version of label definitions these guidelines operate with.
  # This ensures that guidelines are interpreted in the context of specific label meanings.
  applies_to_label_definitions_version: "1.0" # Points to label_definitions_v1.0.yaml

  general_rules:
    - rule: "Annotate based on explicit information in the text."
    - rule: "If unsure, do not annotate and flag the example if possible."
    - rule: "Span boundaries should be minimal but complete."
    # ... more general rules

  workflow_overview:
    segmentation_notes: "Comments are pre-segmented. Annotate within these segments."
    iteration_process: "Annotation will be done in campaigns. Each campaign focuses on a subset of labels..."
    # ...

  # Optional: Per-label nuances that are more about process/priority than core definition,
  # or specific instructions for the annotator not covered in label definitions.
  # However, core definitions should remain in label_definitions.
  label_specific_process_notes:
    - label: "ARTIST_BIO_SPAN"
      note: "Prioritize capturing complete thoughts or facts. If a bio spans multiple sentences, try to capture all related ones if coherent."
    - label: "SHOW_DATE_SPAN"
      note: "Be mindful of relative dates and ensure context is captured."
  ```

### 2.3. Campaign Configurations

- **Location:** `config/campaign_configs/`
- **Structure:** One YAML file per defined annotation campaign.
  - Filename: `{campaign_name}.yaml` (e.g., `artist_biographical_campaign.yaml`, `technical_tags_campaign.yaml`).
- **Content (YAML Structure per file):**

  ```yaml
  campaign_name: "artist_biographical_details_v1" # Unique identifier for the campaign
  description: "Focus on annotating artist biographical information (ARTIST_BIO_SPAN) and associated artist tags (ARTIST_TAG), and origin locations (ARTIST_LOC_ORGIN_SPAN)."

  # Version of label definitions this campaign adheres to
  label_definitions_version: "1.0" # e.g., points to label_definitions_v1.0.yaml

  # Version of annotation guidelines this campaign adheres to
  annotation_guidelines_version: "1.0" # e.g., points to annotation_guidelines_v1.0.yaml

  # Labels to be actively presented and annotated in the Prodigy UI for this campaign
  target_labels_for_ui:
    - "ARTIST_TAG"
    - "ARTIST_BIO_SPAN"
    - "ARTIST_LOC_ORGIN_SPAN"
    # - "ARTIST_ALIAS_SPAN" # Example: can be added in a later version of this campaign config

  # Specification for the source data for this campaign's initial iteration
  # This can be flexible, e.g., a path to a pre-filtered JSONL, or parameters for DB query
  source_data_specification:
    type: "jsonl_file" # or "database_query"
    path: "data/source_material/unannotated_batch1.jsonl"
    # Or for DB:
    # query_params:
    #   min_comment_length: 50
    #   random_sample_size: 1000

  # Optional: Pre-labeling strategy for this campaign
  pre_labeling:
    use_script_04: true # Whether to run 04_prelabel_for_prodigy.py
    # script_04_config: # Optional: if 04_prelabel needs specific overrides for this campaign
    #   metadata_fields_to_check: # Override METADATA_FIELDS_TO_CHECK from prelabel_config.py
    #     db_artist_name: ["ARTIST_NAME", "artist"] # Only pre-label artists for this campaign

  # Optional: ANN bootstrapping configuration
  ann_bootstrapping:
    initial_ann_index: null # Path to an existing index if starting from one
    # If null, an index will be built from source_data_specification after pre-labeling
    sentence_transformer_model: "all-MiniLM-L6-v2" # Model for embedding
  ```

## 3. Directory Structure (Revised)

```
kexp_data/
├── config/
│   ├── label_definitions/
│   │   ├── label_definitions_v1.0.yaml
│   │   └── label_definitions_v1.1.yaml
│   ├── annotation_guidelines/
│   │   ├── annotation_guidelines_v1.0.yaml
│   │   └── annotation_guidelines_v1.1.yaml
│   ├── campaign_configs/
│   │   ├── artist_biographical_campaign.yaml
│   │   └── release_info_campaign.yaml
│   ├── prelabel_config.py         # General config for 04_prelabel_for_prodigy.py
│   └── gazetteers/                # Existing
│   └── labels.txt                 # Master list of all labels (can be generated from label_definitions)
├── data/
│   ├── raw_kexp_data/             # Raw DB extracts (existing)
│   ├── source_material/           # Cleaned/filtered raw data ready for campaign input (e.g. large JSONL files)
│   ├── processed_examples/        # Per-Prodigy-dataset input: {prodigy_dataset_name}_examples.jsonl & .ann_index
│   ├── prodigy_db/                # Location for Prodigy SQLite databases (if used)
│   ├── prodigy_exports/           # Raw exports from Prodigy: {prodigy_dataset_name}_export_{timestamp}.jsonl
│   ├── campaign_annotated/        # Consolidated annotations per campaign:
│   │                              # {campaign_name}_ldv{X.Y}_agv{A.B}_annotated_final.jsonl
│   └── annotated/                 # Final merged gold standard:
│                                  # gold_spancat_ldv{X.Y}_agv{A.B}_v{gold_version}.jsonl & .ann_index
│   └── models/                    # Trained models, linked to gold data versions
├── docs/
│   ├── implementation_plan_gold_standard_workflow.md # This file
│   ├── dataset_registry.csv       # Or dataset_registry.yaml
│   ├── (existing .md files like annotation_schema.md - can be converted/referenced)
├── prodigy_recipes/
│   └── prelabel_ent_spans_manual.py # Existing recipe
├── scripts/
│   ├── 00_extract_kexp_comments.py    # Existing
│   ├── 00b_segment_and_normalize_comments.py # Existing
│   ├── 04_prelabel_for_prodigy.py     # Existing, may need minor adjustments or be called by wrapper
│   ├── run_prodigy_campaign.py        # New: Prodigy recipe wrapper/launcher
│   ├── 05a_merge_campaign_iterations.py # New
│   ├── 05b_merge_campaigns_to_gold.py   # New
│   └── utilities/
│       ├── build_ann_index.py           # New/Refined
│       ├── validate_definitions.py      # New: YAML validation
│       └── examine_prodigy_export.py    # Existing idea
├── src/                             # Existing Python modules
│   └── kexp_processing_utils/
│       └── normalization.py
│       └── comment_parser.py
├── .env
└── README.md
```

## 4. Dataset Registry

A central log to track the lineage and status of all generated datasets.

- **Location:** `docs/dataset_registry.csv` (or `dataset_registry.yaml` if preferred for more complex fields).
- **Format:** CSV is simpler for manual updates, YAML for richer structure.
- **Fields (CSV example):**
  ```csv
  prodigy_dataset_name,campaign_name,creation_date,status,input_examples_file,ann_index_file,prodigy_export_file,label_definitions_version,annotation_guidelines_version,description,source_prodigy_datasets_merged
  artist_bio_details_v1_ldv1.0_agv1.0_iter1_20231115,artist_biographical_details_v1,2023-11-15,annotated,processed_examples/artist_bio_details_v1_ldv1.0_agv1.0_iter1_20231115_examples.jsonl,processed_examples/artist_bio_details_v1_ldv1.0_agv1.0_iter1_20231115_examples.ann_index,prodigy_exports/artist_bio_details_v1_ldv1.0_agv1.0_iter1_20231115_export_20231116.jsonl,1.0,1.0,"Initial bootstrap for artist bio",
  # ...
  campaign_annotated_artist_bio_details_v1_ldv1.0_agv1.0,artist_biographical_details_v1,2023-11-20,merged_campaign,N/A,N/A,campaign_annotated/artist_bio_details_v1_ldv1.0_agv1.0_annotated_final.jsonl,1.0,1.0,"Consolidated artist bio annotations from all iterations for v1 campaign","artist_bio_details_v1_ldv1.0_agv1.0_iter1_20231115;artist_bio_details_v1_ldv1.0_agv1.0_iter2_20231118"
  # ...
  gold_standard_v1_ldv1.0_agv1.0,N/A,2023-11-25,gold_created,N/A,annotated/gold_spancat_ldv1.0_agv1.0_v1.ann_index,annotated/gold_spancat_ldv1.0_agv1.0_v1.jsonl,1.0,1.0,"First gold standard release","campaign_annotated_artist_bio_details_v1_ldv1.0_agv1.0;campaign_annotated_release_info_v1_ldv1.0_agv1.0"
  ```
- **`status` field values:** `defined`, `processing_input`, `annotation_in_progress`, `annotated`, `exported`, `merged_campaign`, `gold_created`, `deprecated`.
- **`source_prodigy_datasets_merged`:** Semicolon-separated list of Prodigy dataset names or campaign annotated file names that were merged to create this entry.

## 5. Script Specifications

### 5.1. `scripts/run_prodigy_campaign.py` (New)

- **Purpose:** Standardize launching of Prodigy annotation sessions for specific campaigns, ensuring correct label filtering and data flow.
- **Inputs (Command Line Arguments):**
  - `--campaign-config`: Path to the campaign's YAML configuration file (e.g., `config/campaign_configs/artist_biographical_campaign.yaml`).
  - `--prodigy-dataset-name`: Optional. If provided, use this exact name. If not, generate one based on campaign name, versions, iteration, and date.
  - `--iteration-number`: Integer for the current iteration within the campaign (e.g., 1, 2).
  - `--iteration-description`: Short string (e.g., "bootstrap", "ann_refined_pass1").
  - `--input-examples-file`: Optional. Path to a specific JSONL file to use as input. If not provided, behavior depends on `source_data_specification` in campaign config.
  - `--force-rebuild-ann-index`: Flag, if input examples are provided, force rebuilding its ANN index.
  - Other standard Prodigy arguments (e.g., `--port`, `--host`).
- **Core Logic:**
  1.  **Load Configurations:**
      - Read the specified campaign config YAML.
      - Read the referenced `label_definitions_{version}.yaml` and `annotation_guidelines_{version}.yaml`.
  2.  **Determine Prodigy Dataset Name:**
      - If `--prodigy-dataset-name` is given, use it.
      - Else, construct: `{campaign_name}_ldv{label_def_ver}_agv{guideline_ver}_iter{iter_num}_{iter_desc}_{date}`.
  3.  **Prepare Input Stream & ANN Index:**
      - If `--input-examples-file` is provided:
        - Use this file. Check if a corresponding `.ann_index` exists.
        - If index doesn't exist or `--force-rebuild-ann-index` is true, call `scripts/utilities/build_ann_index.py`.
      - Else (no explicit input file):
        - Use `source_data_specification` from campaign config.
        - If `type: "jsonl_file"`, use that path. Build ANN index if needed.
        - If `type: "database_query"`, implement logic to fetch data (similar to current recipe) and save to a temporary file in `data/processed_examples/` (named after the Prodigy dataset). Then build ANN index.
        - If `pre_labeling.use_script_04` is true, run `04_prelabel_for_prodigy.py` on the source data before ANN indexing. The output of script 04 becomes the input for Prodigy.
      - Store paths to the final input JSONL and its ANN index. These will be logged in `dataset_registry.csv`.
  4.  **Configure Prodigy UI Labels:**
      - From campaign config, get `target_labels_for_ui`.
      - From `label_definitions`, get full details for these target labels (e.g., for colors if Prodigy config supports it per label).
  5.  **Launch Prodigy:**
      - Construct the `prodigy` command using `subprocess`.
      - Command will be e.g.: `python -m prodigy {prodigy_dataset_name} {path_to_input_stream.jsonl} --recipe prodigy_recipes.prelabel_ent_spans_manual ...other_args...`
      - Crucially, pass the `target_labels_for_ui` to the recipe if it's modified to accept them, OR generate a temporary Prodigy config file (`prodigy.json`) with these labels and point Prodigy to it.
        ```json
        // Example temporary prodigy.json
        {
          "labels": ["ARTIST_TAG", "ARTIST_BIO_SPAN"],
          "custom_theme": {
            "labels": { "ARTIST_TAG": "#FFD700", "ARTIST_BIO_SPAN": "#ADFF2F" }
          }
          // other prodigy settings
        }
        ```
  6.  **Update Dataset Registry:**
      - After Prodigy starts (or before, if dataset name is confirmed), add/update an entry in `dataset_registry.csv` with status `annotation_in_progress`.
- **Outputs:**
  - A running Prodigy session.
  - Entries in `dataset_registry.csv`.
  - Potentially new files in `data/processed_examples/` if data was prepared.

### 5.2. `scripts/05a_merge_campaign_iterations.py` (New)

- **Purpose:** Consolidate annotations from multiple Prodigy export files that belong to the _same_ annotation campaign (i.e., same label focus, but different iterations).
- **Inputs (Command Line Arguments):**
  - `--campaign-name`: The name of the campaign being consolidated.
  - `--export-files`: A list of paths to `data/prodigy_exports/*.jsonl` files for this campaign.
  - `--label-definitions-version`: The target `label_definitions_v{X.Y}.yaml` version these merged annotations should conform to.
  - `--annotation-guidelines-version`: The target `annotation_guidelines_v{A.B}.yaml` version.
  - `--output-file`: Path for the consolidated campaign annotation file (e.g., `data/campaign_annotated/{campaign_name}_ldv{X.Y}_agv{A.B}_annotated_final.jsonl`).
- **Core Logic:**
  1.  **Load Data:** Read all specified export JSONL files.
  2.  **Group by Example:** Group annotations by a unique example identifier (e.g., `_input_hash` if consistent, or hash of `text` content).
  3.  **Merge Annotations for Each Example:**
      - For each unique example, collect all its annotated versions from the different export files.
      - **Conflict Resolution (if same example annotated multiple times, e.g., in review iterations):**
        - Prioritize the version from the export file with the latest timestamp (assuming later means more reviewed/correct).
        - If versions are tied to guideline sub-versions within a campaign (less likely but possible), use that as a tie-breaker.
        - The spans list from the chosen version becomes the canonical one for this example.
  4.  **Schema Alignment (Optional but Recommended):**
      - Verify that all labels in the merged spans conform to the specified `--label-definitions-version`. Log warnings/errors if discrepancies.
  5.  **Write Output:** Save the consolidated list of unique examples with their merged/chosen annotations to the `--output-file` in JSONL format.
  6.  **Update Dataset Registry:** Add an entry for this consolidated campaign file, noting the source export files.
- **Outputs:**
  - A single JSONL file in `data/campaign_annotated/`.
  - Updates to `dataset_registry.csv`.

### 5.3. `scripts/05b_merge_campaigns_to_gold.py` (New)

- **Purpose:** Merge the final, consolidated annotations from _different_ campaigns into a single, comprehensive gold standard dataset.
- **Inputs (Command Line Arguments):**
  - `--campaign-annotated-files`: A list of paths to `data/campaign_annotated/*.jsonl` files (output from `05a_merge_campaign_iterations.py`).
  - `--gold-version-tag`: A string tag for this version of the gold standard (e.g., "v1.0_alpha", "v1.0_final").
  - `--target-label-definitions-version`: The primary `label_definitions_v{X.Y}.yaml` version this gold set aims to conform to.
  - `--target-annotation-guidelines-version`: The primary `annotation_guidelines_v{A.B}.yaml` version.
  - `--output-base-filename`: Base for the output gold file (e.g., `gold_spancat`). Final name will include versions.
- **Core Logic:**
  1.  **Load Data:** Read all specified campaign-annotated JSONL files.
  2.  **Identify Label Definition Versions:** For each input file, determine the `label_definitions_version` it was created with (from filename or registry).
  3.  **Group by Example:** Group annotations by a unique example identifier.
  4.  **Merge Annotations for Each Example:**
      - For each unique example, collect its annotations from all campaign files it appears in.
      - Since each campaign _should_ have focused on a distinct set of `target_labels_for_ui`, the `spans` from different campaign files for the same example text are typically concatenated.
      - **Conflict Resolution (Critical):**
        - If the _same label_ was accidentally (or intentionally due to evolving definitions) annotated for the _same text snippet_ in two different campaign files:
          - Prioritize the annotation from the campaign file that used the _latest `label_definitions_version`_ for that specific label.
          - Log these conflicts and their resolution in detail.
        - Filter out any spans whose labels are not present in the `--target-label-definitions-version`.
  5.  **Deduplicate Spans:** After merging spans for an example, ensure no identical spans (same start, end, label) exist.
  6.  **Final Validation:**
      - Validate all spans against the `--target-label-definitions-version`.
      - Perform consistency checks (e.g., no overlapping spans of incompatible types, if such rules exist).
  7.  **Write Output:**
      - Save the gold standard data to `data/annotated/{output_base_filename}_ldv{X.Y}_agv{A.B}_{gold_version_tag}.jsonl`.
      - Optionally, build an ANN index for this gold set using `scripts/utilities/build_ann_index.py`.
  8.  **Update Dataset Registry:** Add an entry for the new gold standard file.
- **Outputs:**
  - A gold standard JSONL file in `data/annotated/`.
  - Optionally, an `.ann_index` file.
  - Detailed logs, especially of any conflicts resolved.
  - Updates to `dataset_registry.csv`.

### 5.4. `scripts/utilities/build_ann_index.py` (New/Refined)

- **Purpose:** Create or update an HNSWLib ANN index from a JSONL file containing text examples.
- **Inputs (Command Line Arguments):**
  - `--input-jsonl`: Path to the JSONL file (must contain a "text" field).
  - `--output-index-path`: Path to save the HNSWLib index.
  - `--model-name`: Name of the SentenceTransformer model to use (default: "all-MiniLM-L6-v2").
  - `--force`: Flag to overwrite if index already exists.
- **Core Logic:** (Similar to `build_and_save_ann_index` in current Prodigy recipe)
  1.  Load SentenceTransformer model.
  2.  Read "text" fields from input JSONL.
  3.  Encode texts to embeddings.
  4.  Initialize HNSWLib index.
  5.  Add items to index.
  6.  Save index to file.
- **Outputs:** An HNSWLib index file.

### 5.5. `scripts/utilities/validate_definitions.py` (New)

- **Purpose:** Validate the structure, content, and versioning consistency of YAML definition files (`label_definitions`, `annotation_guidelines`, `campaign_configs`).
- **Inputs (Command Line Arguments):**
  - `--file-type`: Type of file to validate (`label_def`, `guideline`, `campaign_conf`).
  - `--path`: Path to a specific YAML file or a directory containing them.
- **Core Logic:**
  - **Common Checks:** Valid YAML syntax.
  - **Label Definitions:**
    - Presence of `version`, `date`, `labels` keys.
    - `version` matches filename.
    - Each label has `name`, `description`, `guidance`, `examples` (with `positive`/`negative` lists).
  - **Annotation Guidelines:**
    - Presence of `version`, `date`, `description`, `applies_to_label_definitions_version`.
    - `version` matches filename.
    - Check if referenced `label_definitions_version` exists.
  - **Campaign Configurations:**
    - Presence of `campaign_name`, `description`, `label_definitions_version`, `annotation_guidelines_version`, `target_labels_for_ui`, `source_data_specification`.
    - Check if referenced definition/guideline versions exist.
    - Check if all labels in `target_labels_for_ui` are defined in the referenced `label_definitions` file.
- **Outputs:**
  - Console output: Success messages or detailed error/warning messages.
  - Return code: 0 for success, non-zero for validation failures.

### 5.6. Existing Scripts (`04_prelabel_for_prodigy.py`)

- **Modifications/Usage:**
  - This script can still be used for broad pre-labeling of a large dataset.
  - The `run_prodigy_campaign.py` script might call it to prepare an input stream if specified in the campaign config.
  - No major internal changes required unless its pre-labeling logic needs to be dynamically filtered based on campaign labels (which could be an advanced optimization).

## 6. Workflow Overview (Step-by-Step)

1.  **Setup & Definition (YAML):**
    - Define/update `config/label_definitions/label_definitions_v{X.Y}.yaml`. Validate with `validate_definitions.py`.
    - Define/update `config/annotation_guidelines/annotation_guidelines_v{A.B}.yaml` (linking to a label definitions version). Validate.
2.  **Campaign Planning:**
    - Define an annotation campaign by creating `config/campaign_configs/{campaign_name}.yaml`. Specify target labels, source data, and the definition/guideline versions it adheres to. Validate.
3.  **Launch Prodigy Annotation Session:**
    - Use `python scripts/run_prodigy_campaign.py --campaign-config config/campaign_configs/{campaign_name}.yaml --iteration-number 1 --iteration-description "initial_bootstrap"`
    - This script:
      - Prepares source data (e.g., from `source_material/` or DB).
      - Runs `04_prelabel_for_prodigy.py` if configured in campaign.
      - Saves prepared data to `data/processed_examples/{prodigy_dataset_name}_examples.jsonl`.
      - Builds an ANN index: `data/processed_examples/{prodigy_dataset_name}_examples.ann_index` using `build_ann_index.py`.
      - Launches Prodigy (`prelabel_ent_spans_manual.py` recipe) with the UI filtered for the campaign's `target_labels_for_ui`.
      - Logs entry in `docs/dataset_registry.csv`.
4.  **Annotation:** Annotator works in Prodigy. Data is saved to the named Prodigy dataset.
5.  **Export Annotations:**
    - `python -m prodigy db-out {prodigy_dataset_name} > data/prodigy_exports/{prodigy_dataset_name}_export_{timestamp}.jsonl`
    - Update status in `dataset_registry.csv`.
6.  **Iterative Refinement (within the same campaign):**
    - Use the exported data (or the original processed examples augmented with new annotations) to build an improved/updated ANN index using `build_ann_index.py`.
    - Launch a new Prodigy session for the same campaign (e.g., `... --iteration-number 2 --iteration-description "ann_refined_pass1"`), potentially using the Prodigy ANN plugin with the new index to find similar examples. The UI remains filtered for the _same campaign labels_.
    - Repeat steps 4-6 as needed for the campaign.
7.  **Consolidate Campaign Annotations:**
    - Once a campaign's iterations are complete:
      `python scripts/05a_merge_campaign_iterations.py --campaign-name {campaign_name} --export-files data/prodigy_exports/{relevant_exports*.jsonl} --label-definitions-version {X.Y} --annotation-guidelines-version {A.B} --output-file data/campaign_annotated/{campaign_name}_ldv{X.Y}_agv{A.B}_annotated_final.jsonl`
    - Update `dataset_registry.csv`.
8.  **Repeat for Other Campaigns:** Repeat steps 2-7 for different campaigns focusing on other label subsets.
9.  **Create Gold Standard Dataset:**
    - When sufficient campaigns are consolidated:
      `python scripts/05b_merge_campaigns_to_gold.py --campaign-annotated-files data/campaign_annotated/{files_to_merge*.jsonl} --gold-version-tag "v1.0" --target-label-definitions-version {X.Y} --target-annotation-guidelines-version {A.B} --output-base-filename "gold_spancat"`
    - This creates `data/annotated/gold_spancat_ldv{X.Y}_agv{A.B}_v1.0.jsonl`.
    - Build an ANN index for this gold set: `python scripts/utilities/build_ann_index.py --input-jsonl {gold_file_path} ...`
    - Update `dataset_registry.csv`.
10. **Model Training:** Train SpanCat models using the versioned gold standard dataset from `data/annotated/`. Model artifacts should also be versioned and ideally named to reflect the gold data version used.

## 7. Versioning Details Summary

- **Label Definitions:** File-based semantic versioning (`label_definitions_vX.Y.yaml`).
- **Annotation Guidelines:** File-based semantic versioning (`annotation_guidelines_vA.B.yaml`).
- **Campaign Configs:** Named by campaign; internal fields link to specific definition/guideline versions.
- **Prodigy Datasets:** Named dynamically by `run_prodigy_campaign.py` including campaign name, definition versions, iteration, and date. Tracked in `dataset_registry.csv`.
- **Processed Examples/ANN Indexes:** Named identically to their corresponding Prodigy dataset.
- **Prodigy Exports:** Named after Prodigy dataset with a timestamp.
- **Consolidated Campaign Data:** Named by campaign, including definition/guideline versions.
- **Gold Standard Data:** Named with base, definition/guideline versions, and a gold version tag.
- **Models:** Filenames should reflect the version of the gold data they were trained on.

## 8. Migration of Existing Documentation

- Review existing `.md` files in `docs/` (e.g., `annotation_schema.md`, `phase_a_ner_annotation.md`).
- Extract relevant label definitions and guidance.
- Convert and integrate this information into the new YAML-based `label_definitions_v1.0.yaml` and `annotation_guidelines_v1.0.yaml` structures.
- The old `.md` files can then be archived or kept as supplementary material if they contain information not suited for the YAML format (e.g., extensive prose on project history).

This detailed plan should provide a solid foundation for implementing the new workflow.
