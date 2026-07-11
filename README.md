# RBCCPS Streetlight Auditing Project

## Overview

This repository is a runnable research codebase for developing an automated streetlight auditing system based on computer vision. The target application is night-time assessment of streetlight condition and performance from road imagery and mobile video captured in urban environments.

The intended system is being developed for deployment in Bangalore under the municipal corporation of Bangalore. The broader objective is to support scalable infrastructure auditing by identifying streetlights, assessing whether they are functioning, and estimating the adequacy of the illumination they provide to roads and nearby public areas.

## Institutional Context

This work is being carried out under RBCCPS, Indian Institute of Science (IISc).

## Team

- Angad Singh Ahuja, Research Intern, RBCCPS, IISc
- Arindam Bhaduri, Research Intern, RBCCPS, IISc

## Academic Supervision

The work is being conducted under the guidance of:

- Prof. Prasant Misra
- Prof. Pandrasamy Arjunan

## Project Scope

The current repository contents indicate work across the following areas:

- literature review on object detection, low-light enhancement, domain adaptation, tracking, illumination measurement, and streetlight-related auditing
- dataset collection and organization for night-time road scenes and related imported image corpora
- annotation automation scripts for rebuilding training corpora, review manifests, and detector handoff packages
- system-design material for a streetlight auditing pipeline
- measurement-block planning for estimating useful illumination from detected and tracked streetlights on edge devices such as mobile phones

## Current Supported Workflow

The primary active workflow is the v3 local-night corpus path:

1. review local positives and negatives in `datasets/derived/annotation_click_review`
2. sync those reviews into `datasets/derived/annotation_automation_v3`
3. build the guarded v3 corpus
4. train and compare v3 detector runs under `runs/`

The main entrypoints for that workflow are:

- [scripts/annotation_automation/README.md](scripts/annotation_automation/README.md)
- [documentation/system_design/new_design/run3_accuracy_recovery_implementation.md](documentation/system_design/new_design/run3_accuracy_recovery_implementation.md)
- [documentation/system_design/README.md](documentation/system_design/README.md)

Retained older generations are still present because the v3 path depends on prior review outputs and corpus history, but they should be treated as supporting lineage rather than the default operating surface.

The package entrypoint for the refactored detector codebase is now:

```bash
python -m rbccps_od <command>
```

Primary detector commands:

- `review-app`
- `sync-v3-reviews`
- `build-v3-corpus`
- `train`
- `validate`
- `build-tiled`
- `build-mixed`
- `download-models`
- `run-baseline`
- `run-advanced-pipeline`

## Repository Structure

```text
RBCCPS_Directory/
  datasets/
  documentation/
  final_paper/
  scripts/
  .gitignore
  README.md
```

### `datasets/`

Contains local dataset storage used for research and experimentation, including raw videos, extracted frames, imported datasets, and derived artifacts. The current active derived areas are `datasets/derived/annotation_click_review/` and `datasets/derived/annotation_automation_v3/`. The current organization and retention notes for this directory are documented in [datasets/README.md](datasets/README.md).

### `documentation/`

Contains literature-review material, downloaded references, bibliography files, system-design notes, and repository-maintenance notes. The active system-design index lives at [documentation/system_design/README.md](documentation/system_design/README.md).

### `final_paper/`

Contains the current paper source and compiled PDF. Treat this as a core project output rather than disposable build clutter.

### `scripts/`

Contains tracked implementation code for local workflows. The active annotation automation workflow is documented under [scripts/annotation_automation/README.md](scripts/annotation_automation/README.md).

## Version Control Notes

This directory is initialized as a Git repository. `.gitignore` has been prepared to keep large datasets, regenerated artifacts, and local-only research assets out of version control by default.

## Ignored Folders

The following folders are intentionally listed in `.gitignore` because they contain large binary assets, regenerated data, or imported reference material that should usually remain outside version control.

| Path | Category | Primary Contents | Source / Provenance | Reason for Exclusion |
| --- | --- | --- | --- | --- |
| `datasets/raw/` | Raw capture storage | Original night-drive video clips (`.mp4`) | Phone or vehicle-mounted capture sessions | Large binary source inputs; retained locally rather than versioned |
| `datasets/extracted_frames/` | Derived dataset storage | Frame batches extracted from raw videos (`.jpg`) | Generated during preprocessing | Regenerable from source videos and large in volume |
| `datasets/imported/` | External dataset storage | Imported datasets and platform exports | Roboflow exports and other third-party collections | Mixed provenance, large size, and potentially separate redistribution constraints |
| `datasets/derived/` | Generated artifacts | Preview videos, outputs, intermediate artifacts, and annotation automation products | Produced by scripts, experiments, and review workflows | Rebuildable outputs that should not clutter version history |
| `documentation/literature_review/papers/` | Literature asset storage | Downloaded research papers and PDF references | Manually collected literature sources | Large binary reference assets, not primary project source files |

## Cleanup Status

- The active surface is defined around the v3 review and corpus workflow.
- Older workflow generations are retained, but they are no longer the default entrypoint.
- Local runtime clutter such as `.venvs/`, `_tmp_*`, `runs/`, and `_ultralytics_config/` remains local-only state.
- The current preservation and deletion triage lives in [documentation/repository_maintenance/cleanup_inventory.md](documentation/repository_maintenance/cleanup_inventory.md).

## Status

At its current stage, this repository should be treated as an active research codebase with a defined current workflow, preserved local data/model assets, and explicit cleanup boundaries for legacy or rebuildable material.
