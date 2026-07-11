# Annotation Automation Scripts

This directory now contains only retained helper scripts that have not yet been folded into the package, plus launchers and reviewer-facing notes.

## Current Status

The old embedded streetlight review app has been replaced by the modular measurement annotator. New review work should use:

```bash
python -m rbccps_annotator <command>
```

The compatibility command `python -m rbccps_od review-app` now forwards to the modular annotator command surface.

The default supported workflow is the v3 local-night path:

1. review click-review items with `python -m rbccps_od review-app`
2. sync v3 review manifests with `python -m rbccps_od sync-v3-reviews`
3. build the guarded v3 corpus with `python -m rbccps_od build-v3-corpus`
4. train and compare detector runs with `python -m rbccps_od train`

The canonical command surface is now:

```bash
python -m rbccps_od <command>
```

Direct mapping:

- `python -m rbccps_od review-app`
- `python -m rbccps_od sync-v3-reviews`
- `python -m rbccps_od build-v3-corpus`
- `python -m rbccps_od train`
- `python -m rbccps_od validate`
- `python -m rbccps_od build-tiled`
- `python -m rbccps_od build-mixed`
- `python -m rbccps_od build-review-subset`
- `python -m rbccps_od propagate-reviews`
- `python -m rbccps_od export-candidates`
- `python -m rbccps_od score-reliability`
- `python -m rbccps_od evaluate-gate`
- `python -m rbccps_od integrate-reviewed-data`
- `python -m rbccps_od materialize-review-batches`

## Retained Helpers

- `extract_mapillary_vistas_streetlights.py`
  - external dataset extraction helper
- `extract_openimages_streetlights.py`
  - external dataset extraction helper
- `REVIEW_WORKFLOW.md`
  - reviewer-facing instructions for filling the hard-negative and calibration manifests
- `make_hard_negative_contact_sheets.ps1`
  - retained review helper
  - creates image contact sheets for reviewed hard-negative buckets
- `make_calibration_contact_sheets.ps1`
  - retained review helper
  - creates image contact sheets for pending calibration rows
- `generate_calibration_overlays.ps1`
  - retained review helper
  - draws streetlight boxes on the calibration images for visual review
## Current Typical Order

1. Run `python -m rbccps_od review-app` and complete:
   - `Jobin Positive Review`
   - `Arindam Positive Review`
   - `Hard-Negative Review`
2. Run `python -m rbccps_od sync-v3-reviews` to sync the v3 review workspace.
3. Run `python -m rbccps_od build-v3-corpus` for the next strict local-night corpus.
4. Train the local baseline with `python -m rbccps_od train`.
5. If recall is still weak, run `python -m rbccps_od build-tiled` on the v3 YOLO export and compare against the full-frame run.
6. Only after a strong local baseline exists, run `python -m rbccps_od build-mixed` to test light external augmentation.

## Review labels

- Hard-negative manifest allowed labels:
  - `clean_negative`
  - `ambiguous`
  - `missed_positive`
- Calibration manifest should only be considered locked when:
  - `review_status` is no longer `pending_manual_lock`
  - `locked` is set to `1`

## Review app

Launch locally with:

```bash
python -m rbccps_od review-app
```

Convenience launchers:

```bash
bash scripts/annotation_automation/run_review_app.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\annotation_automation\run_review_app.ps1
```

The app writes its working outputs under:

- `datasets/derived/annotation_click_review/app_data`
- `datasets/derived/annotation_click_review/reviews`

## Cleanup Notes

- Treat the package under `src/rbccps_od` as the only active OD codebase.
- Treat this directory as helper-script overflow only.
- Keep local runtime clutter such as `__pycache__/` out of version control.
