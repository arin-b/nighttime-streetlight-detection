# Review Workflow

This note defines the manual review expected by the annotation automation pipeline.

## 1. Hard-negative review

Primary manifest:

- [hard_negative_review_manifest.csv](F:\RBCCPS_Directory\datasets\derived\annotation_automation\reviews\hard_negative_review_manifest.csv)

Reviewer image batch:

- [hard_negative](F:\RBCCPS_Directory\datasets\derived\annotation_automation\reviews\batches\hard_negative)

Edit only these columns:

- `review_label`
- `notes`

Allowed `review_label` values:

- `clean_negative`
- `ambiguous`
- `missed_positive`

Decision rule:

- Use `clean_negative` only when you are confident there is no visible streetlight that should have been annotated.
- Use `missed_positive` when at least one real streetlight is visible and the frame should not enter the negative pool.
- Use `ambiguous` when the frame is too uncertain to trust as a clean negative.

## 2. Calibration subset lock

Primary manifest:

- [calibration_subset_manifest.csv](F:\RBCCPS_Directory\datasets\derived\annotation_automation\reviews\calibration_subset_manifest.csv)

Reviewer image batch:

- [calibration](F:\RBCCPS_Directory\datasets\derived\annotation_automation\reviews\batches\calibration)

Edit these columns:

- `review_status`
- `locked`
- `notes`

Recommended `review_status` values:

- `locked_clean`
- `locked_needs_fix`
- `locked_negative`

Lock rule:

- Set `locked` to `1` only after the image and its annotations are acceptable as calibration gold data.
- Keep `locked` at `0` for anything that still needs correction or discussion.

## 3. After review

Run:

```powershell
& 'C:\Users\ahuja\AppData\Local\Programs\Python\Python312\python.exe' `
  'F:\RBCCPS_Directory\scripts\annotation_automation\integrate_reviewed_data.py'
```

That script will:

- admit only `clean_negative` rows into the derived YOLO corpus
- separate `ambiguous`, `missed_positive`, and still-pending rows into separate CSVs

## 4. What not to change

Do not edit:

- `review_candidate_id`
- `calibration_id`
- `dataset_id`
- `clip_id`
- `frame_id`
- `image_path`

These fields are part of the traceability chain.
