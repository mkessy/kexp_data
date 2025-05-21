# Phase A: Named Entity Recognition (NER) Annotation

This document details the steps and specifications for Phase A of the annotation process, focusing on Named Entity Recognition (NER). The goal of this phase is to create a high-quality dataset (`kexp_ner_annotations`) of text segments with accurately labeled named entities.

## 1. Objective

To identify and label a limited set of core named entities within the KEXP DJ comments. These entities provide foundational geographical and temporal context. The majority of descriptive tagging (like artist names, album titles, genres, roles) is now handled in Phase B (SpanCat Annotation).

## 2. NER Entities to Annotate

The following NER labels will be used, as defined in `docs/annotation_schema.md`:

1.  **`LOCATION`**: Geographical locations mentioned in the text.
2.  **`DATE`**: Specific dates, years, or relative time expressions.

_Self-correction: Removed previous NER entities like `ALBUM_TITLE`, `ARTIST_NAME`, `SONG_TITLE`, `RECORD_LABEL_NAME`, `QUOTE_TEXT`, `RELEASE_DESCRIPTOR`, and `URL` as per the latest schema revision. These are either moved to SpanCat or removed entirely._

## 3. Data Sources & Preparation

- **Primary Input Data**: `data/processed/04_final_segments_for_annotation.jsonl`
  - This file is the output of the data processing pipeline, which includes:
    1.  Extraction of comments from the database (`00_extract_db_comments.py`).
    2.  Segmentation of comments into smaller, coherent units (`01_segment_comments.py`).
    3.  _Self-correction: The script `02_prelabel_ner_entities.py` previously pre-labeled `ALBUM_TITLE`, `ARTIST_NAME`, and `SONG_TITLE`. Its role is now re-evaluated. For Phase A, we assume this script no longer adds these (now SpanCat) labels as NER pre-labels to the stream, or it only pre-labels `LOCATION` and `DATE` if feasible and desired. If it only pre-labeled the moved entities, it might be bypassed for NER data preparation or modified._
    4.  Filtering out segments that consist only of URLs (`03_filter_link_only_segments.py`).
  - Each line in this JSONL file represents a text segment ready for NER annotation for `LOCATION` and `DATE`.

## 4. Prodigy Datasets for NER

- **`kexp_ner_annotations`**: The main Prodigy dataset where all verified and manually added NER annotations for `LOCATION` and `DATE` will be stored.
- **`kexp_ner_ann_bootstrap_[label]_idx`**: Temporary Prodigy ANN index datasets created for `LOCATION` or `DATE` (e.g., `kexp_ner_ann_bootstrap_location_idx`).
- **`kexp_ner_ann_bootstrap_[label]`**: Temporary Prodigy datasets for annotations collected during `ann`-driven bootstrapping sessions (e.g., `kexp_ner_ann_bootstrap_location`). These will be merged into `kexp_ner_annotations`.
- **`kexp_ner_lunr_bootstrap_[label]`**: Temporary Prodigy datasets for annotations collected during `lunr`-driven bootstrapping sessions (e.g., `kexp_ner_lunr_bootstrap_date`). These will be merged into `kexp_ner_annotations`.

## 5. Annotation Steps & Prodigy Commands

### Step 5.1: Initial Manual NER Annotation

- **Objective**: To create an initial seed set of high-quality, manually annotated data for `LOCATION` and `DATE`.
- **Prodigy Recipe**: `ner.manual`
- **Input Data**: `data/processed/04_final_segments_for_annotation.jsonl`
- **Output Dataset**: `kexp_ner_annotations`
- **Command**:
  ```bash
  prodigy ner.manual kexp_ner_annotations \
      data/processed/04_final_segments_for_annotation.jsonl \
      --label LOCATION,DATE
  ```
- **Process**: Annotate a diverse batch of several hundred segments to ensure a good representation of `LOCATION` and `DATE` entities.

### Step 5.2: Bootstrapping NER with `prodigy-lunr-plugin` (Keyword-based Search)

- **Objective**: To quickly find and annotate segments containing `LOCATION` or `DATE` entities using keyword searches if applicable (e.g., for date patterns or common location indicators).
- **Prodigy Recipe**: `lunr.ner.search` (from the `prodigy-lunr-plugin`).
- **Input Data**: `data/processed/04_final_segments_for_annotation.jsonl`
- **Output Dataset**: Temporary datasets like `kexp_ner_lunr_bootstrap_date`, then merged into `kexp_ner_annotations`.
- **Command Example (for `DATE`)**:
  ```bash
  # Example keywords for dates - actual patterns might be better handled by `ner.match` or manual annotation
  # For lunr, broad keywords might be month names, days of week, year indicators if not too noisy.
  prodigy lunr.ner.search kexp_ner_lunr_bootstrap_date \
      data/processed/04_final_segments_for_annotation.jsonl \
      "January,February,March,April,May,June,July,August,September,October,November,December,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday,202,199,198,197" \
      --label DATE --keyword-meta-keys text --allow-spans
  ```
  _(Adjust keywords as needed. For `LOCATION`, this might be less effective than ANN or manual annotation due to diversity.)_
- **Process**: Use cautiously for `DATE` if specific keywords are helpful. For `LOCATION`, `ann.ner.manual` or direct manual annotation is likely more robust.

### Step 5.3: Bootstrapping NER with `prodigy-ann` (Similarity-based Search)

- **Objective**: To find and annotate segments semantically similar to already annotated examples of `LOCATION` or `DATE`.
- **Prodigy Recipes**: `ann.ner.index` and `ann.ner.manual` (from `prodigy-ann`).
- **Source of Examples for ANN**: The `kexp_ner_annotations` dataset.
- **Input Data for Annotation**: `data/processed/04_final_segments_for_annotation.jsonl`
- **Output Dataset**: Temporary datasets like `kexp_ner_ann_bootstrap_location`, then merged into `kexp_ner_annotations`.
- **Commands**:
  1.  **Create an ANN index (once per focused label session)**:
      _Example for `LOCATION` entity:_
      ```bash
      prodigy ann.ner.index kexp_ner_ann_bootstrap_location_idx \
          kexp_ner_annotations --label LOCATION
      ```
  2.  **Annotate using the ANN index**:
      _Example for `LOCATION` entity:_
      ```bash
      prodigy ann.ner.manual kexp_ner_ann_bootstrap_location \
          data/processed/04_final_segments_for_annotation.jsonl \
          kexp_ner_ann_bootstrap_location_idx \
          --label LOCATION,DATE # Still allow annotating both core NER types
      ```
- **Process**: Use this for `LOCATION` and potentially for `DATE` if diverse contextual examples exist.

### Step 5.4: Iteration and Merging

- **Objective**: To continuously improve the `kexp_ner_annotations` dataset for `LOCATION` and `DATE`.
- **Process**: Same as previously described, but focused only on `LOCATION` and `DATE` entities.

## 6. Final Output of Phase A

- **Primary Output File**: `data/annotations/ner_annotations.jsonl`
- **Command to Export**: After sufficient annotation and review:
  ```bash
  prodigy db-out kexp_ner_annotations > data/annotations/ner_annotations.jsonl
  ```
- **Content**: This JSONL file will contain the text segments along with a `"spans"` array detailing all manually verified and added `LOCATION` and `DATE` NER annotations. This file serves as input for Phase B (SpanCat Annotation), providing this contextual NER information.

## 7. Quality Control

- Regularly review annotations for `LOCATION` and `DATE`.
- Focus on consistency in applying these NER labels according to the `annotation_schema.md`.
- Pay attention to span boundaries.

This revised plan for Phase A significantly simplifies the NER task, focusing on core entities, and aligns with the updated overall annotation strategy.
