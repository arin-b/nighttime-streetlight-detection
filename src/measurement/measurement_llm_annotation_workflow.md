# Token-Efficient LLM Draft Annotation Workflow

## Goal

Use the visual LLM as the primary draft annotator while minimizing token usage, browser interaction, and repeated reasoning. Human reviewers then verify and correct the draft labels.

This workflow treats LLM output as **draft labels**, not final ground truth.

## Core Principle

Do not ask the LLM to think from scratch for every frame. Give it a fixed protocol, process images in batches, and only spend extra tokens on difficult frames.

This should be run as a model-assisted annotation loop, not as pure manual labeling. CVAT explicitly supports automatic pre-annotation from models, label matching, thresholds, mask-to-polygon conversion, and optional cleaning of old annotations before a model run. Label Studio similarly treats pre-annotations as model predictions that can be imported and reviewed by annotators.

Sources:

- CVAT automatic annotation: https://docs.cvat.ai/docs/annotation/auto-annotation/automatic-annotation/
- Label Studio pre-annotations: https://labelstud.io/guide/predictions.html
- Label Studio ML backend / prediction workflow: https://labelstud.io/guide/ml.html

## Batch Structure

Use batches of `25-50` frames.

For each batch:

1. Load sampled frames into the annotator.
2. Precompute any available proposals:
   - YOLO lamp-head / pole boxes.
   - SAM or polygon proposals for surfaces.
   - Road/footpath segmentation if available.
   - Prior track IDs if frames came from video.
3. The LLM reviews one screenshot per frame.
4. The LLM edits only what is necessary.
5. Every saved frame gets `needs_human_verification`.

Do not do open-ended discussion during annotation. Keep annotation transactional.

## Data Selection Strategy

Do not annotate `500-800` images in random order.

Use staged active selection:

1. Start with `50` diverse frames selected by route, scene type, lighting condition, exposure, weather, traffic, and confounder density.
2. Train or tune weak pre-labelers.
3. For the next batches, prioritize:
   - low detector confidence,
   - disagreement between YOLO/SAM/LLM proposals,
   - high confounder scenes,
   - multiple-lamp scenes,
   - frames with glare/saturation,
   - frames with new camera/exposure settings,
   - frames from clips/routes not represented in the current labeled set.
4. Keep a small random sample in every batch to detect blind spots.

Label Studio’s ML backend workflow supports retrieving predictions and using model confidence/task ordering to mimic active learning. For this project, that means sorting candidate frames by uncertainty before sending them to the LLM/human review queue.

## Minimal Per-Frame Protocol

For each frame, the LLM should perform this exact sequence:

1. Inspect image once.
2. Mark lamp-head boxes.
3. Mark pole/fixture boxes only when visible.
4. Mark obvious other-light/confounder boxes:
   - shopfronts
   - signs
   - windows
   - headlights
   - reflections
5. Mark road/footpath region coarsely.
6. Mark lit area only if visually clear.
7. If lit area is unclear, use `not properly visible`.
8. Add per-lamp visibility/status guesses only when obvious.
9. Add uncertainty flags.
10. Save as `needs_human_verification`.

Do not attempt calibrated lux values from the image.

## Prediction Handling

Treat all automated labels as predictions until a human accepts them.

Store for every predicted object:

- `source_model`
- `model_version`
- `confidence`
- `created_at`
- `needs_human_verification`

Do not overwrite human-approved labels with new model output unless the reviewer explicitly requests reset/relabel.

This follows the Label Studio distinction between predictions/pre-annotations and final annotations: imported predictions are shown to annotators for review, and must match the project labeling schema.

## Screenshot Policy

To reduce tokens:

- Use one full-page screenshot per frame by default.
- Do not request zoomed crops unless:
  - lamp head is tiny,
  - multiple lamps overlap,
  - object identity is ambiguous,
  - box placement failed.
- Do not re-open saved JSON unless verification fails.
- Do not inspect every exported CSV during annotation.

Expected token cost:

- Easy frame: `2k-4k`
- Normal frame: `4k-7k`
- Hard frame: `8k-15k`

Avoid workflows that repeatedly screenshot before and after every click unless debugging UI.

## Label Confidence Rules

Use these confidence buckets internally:

- `high`: visually obvious, reviewer likely only checks.
- `medium`: likely correct, reviewer should inspect.
- `low`: uncertain, reviewer must inspect.

Store uncertainty through QA flags and notes, not long prose.

Recommended flags:

- `needs_human_verification`
- `uncertain_lamp_identity`
- `uncertain_pole_extent`
- `uncertain_lit_area`
- `multiple_lamp_confounding`
- `headlight_confounder`
- `shopfront_confounder`
- `wet_reflection`
- `tree_occlusion`
- `exposure_problem`
- `no_lux_reference`

When confidence is low, avoid forcing a hard label. Use `unknown`, `mixed`, `uncertain`, or `not properly visible`. This follows the weak-labeling lesson from auto-labeling literature: ambiguous regions should be left weak/uncertain rather than converted into confident wrong labels.

## Fast Decision Rules

### Lamp Head

Draw a tight box around the luminous lamp/source head.

If the fixture is visible but not lit, still annotate if it is clearly a streetlight head and mark status accordingly.

### Pole / Fixture

Draw pole/fixture only if visible enough.

If hidden by darkness, tree, crop, or glare, do not invent it. Mark:

`Pole/fixture is not visible enough`

### Other Lights

Mark only major confounders that could affect measurement:

- bright shopfront
- illuminated sign
- strong headlight
- bright window
- wet road reflection
- large facade glow

Ignore tiny distant lights unless they affect the measured area.

### Road / Footpath

Draw coarse usable public-space area. Precision is less important than covering the correct public surface.

Do not include:

- building walls
- vehicles
- private storefront interiors
- sky

### Lit Area

Mark lit area only when the illuminated region can be reasonably attributed to the selected lamp.

If several lights mix together, mark:

`mixed` attribution or `not properly visible`.

### Visibility / Status

Rate per lamp/track, not globally.

For every lamp that matters:

- lamp status: `on`, `dim`, `off`, `occluded`, `saturated`, or `unknown`
- visibility: `good`, `adequate`, `marginal`, `poor`, `dark`, or `unknown`
- attribution: `certain`, `mixed`, `uncertain`, or `impossible_due_to_confounding`

## Human Review Workflow

The LLM should save drafts with:

`review_status = needs_review`

Human reviewer then:

1. Opens each frame.
2. Checks boxes first.
3. Checks confounders.
4. Checks road/footpath.
5. Checks lit area.
6. Checks per-lamp ratings.
7. Deletes bad labels.
8. Accepts final frame.

The reviewer should not need to annotate from scratch unless the frame is very hard.

## Quality Control

Use three QA layers.

### 1. Gold / Honeypot Frames

Create a fixed validation subset of `5-10%` of frames with trusted human labels. Mix these into normal batches without telling the LLM/reviewer which frames they are.

CVAT documents this as a ground-truth or honeypot approach: a small trusted validation subset can estimate quality without relabeling the full dataset.

Source:

- CVAT Automated QA, Review & Honeypots: https://docs.cvat.ai/docs/qa-analytics/auto-qa/

### 2. Consensus Review For Hard Frames

Use double review only for frames marked:

- `uncertain_lamp_identity`
- `uncertain_lit_area`
- `multiple_lamp_confounding`
- `exposure_problem`
- `not properly visible`

Do not double-review easy frames. Consensus review is expensive, so reserve it for labels that affect model correctness.

CVAT supports consensus-based annotation as part of its QA and analytics workflow.

Source:

- CVAT QA & Analytics: https://docs.cvat.ai/docs/qa-analytics/

### 3. Correction Metrics

For every reviewed batch, record:

- boxes deleted per frame,
- boxes resized per frame,
- polygons deleted per frame,
- lit-area labels changed,
- visibility labels changed,
- frames accepted without edit,
- frames rejected as unusable,
- time per frame for human verification.

If humans edit more than `30-40%` of LLM labels in a category, pause scaling and improve the prompt/pre-labeler/UI for that category.

## Batch QA Sampling

After each batch:

1. Inspect all honeypot frames.
2. Randomly inspect `10%` of non-honeypot frames.
3. Inspect all frames with uncertainty flags.
4. Compute correction types:
   - missing lamp
   - wrong lamp box
   - bad pole box
   - bad confounder
   - bad road polygon
   - bad lit-area attribution
   - wrong visibility rating
5. Update prompt/protocol before next batch.

Stop scaling if QA shows systematic errors. Fix the generator before producing more drafts.

## Stop Conditions

The LLM should stop and mark `needs_review` instead of overthinking when:

- glare hides the lamp head,
- multiple lamps overlap,
- road surface is not visible,
- lit area cannot be separated from headlights/shopfronts,
- exposure is blown out,
- the image is too blurry,
- the scene is not public-road lighting.

## Efficient Prompt Template

Use this prompt for each batch:

```text
You are draft-annotating streetlight measurement frames.

For each frame:
1. Draw tight lamp-head boxes.
2. Draw pole/fixture boxes only when visible.
3. Mark major other-light confounders.
4. Mark coarse road/footpath public space.
5. Mark lit area per selected lamp only if visible; otherwise mark lit area not properly visible.
6. Add per-lamp status, visibility, and attribution.
7. Add uncertainty flags.
8. Save as needs_human_verification.

Do not infer lux values.
Do not over-label tiny distant lights.
Prefer coarse correct polygons over slow perfect polygons.
When uncertain, flag for human review and move on.
```

## Recommended Scale-Up Plan

1. LLM drafts `50` diverse frames.
2. Human reviews all `50`.
3. Add `5-10` gold/honeypot frames.
4. Measure correction rate by label type.
5. Train/tune pre-labelers on accepted labels.
6. Draft `100-150` more, prioritized by active selection.
7. Repeat review.
8. Scale to `500-800` only after the protocol stabilizes.

Suggested gates:

- Lamp-head boxes: reviewer changes under `15%`.
- Pole boxes: reviewer changes under `25%`.
- Confounder boxes: reviewer changes under `30%`.
- Road/footpath polygons: reviewer changes under `25%`.
- Lit-area labels: reviewer changes under `35%`.
- Per-lamp visibility/attribution: reviewer changes under `35%`.

If a category exceeds the gate, keep it as draft-only and require mandatory human verification.

## Practical Token-Saving Rules

- One screenshot per frame unless hard.
- No long explanations during annotation.
- No repeated JSON inspection unless failure is suspected.
- No recompiling docs during annotation.
- No UI debugging during production annotation.
- Use delete-all/reset controls instead of reasoning through stale state.
- Prefer `needs_review` over spending many tokens on uncertain cases.
- Use model pre-annotations before LLM review.
- Show only active label layers when possible.
- Hide low-priority detections while reviewing lit area.
- Never ask the LLM to inspect exports unless a save/export failure is suspected.

## Production Queue Design

Use queues instead of one huge folder.

Recommended queues:

- `draft_easy`: high-confidence prelabels, one screenshot, quick LLM pass.
- `draft_hard`: multi-lamp, glare, occlusion, or high-confounder scenes.
- `human_required`: LLM abstained or frame has critical uncertainty.
- `gold_qa`: trusted frames used for quality estimation.
- `accepted`: human-approved labels.
- `rework`: labels rejected by reviewer.

This keeps token use low because easy frames do not trigger the same inspection depth as hard frames.

## Annotation State Rules

To avoid hidden-state bugs:

- Every saved object must be listed in the UI.
- Every list must support individual delete and delete-all.
- Switching annotation mode must not silently mutate existing labels.
- Lit-area labels must not inherit road/footpath category state.
- Visibility must be per lamp/track.
- Every “not visible” state must be explicit and reversible.

These rules were added after browser-based mistake-recovery tests exposed stale state and selection issues in the custom annotator.

## Expected Cost

For `500-800` images:

- Efficient draft pass: about `2M-5M` tokens.
- With hard cases and QA: about `4M-8M+` tokens.

This is only worthwhile if human review time is substantially reduced.

## External Advice Integrated

The workflow above incorporates these practices from established annotation tools and data-labeling workflows:

- Use model-assisted pre-labeling before human/LLM review.
- Keep predictions/pre-annotations distinct from final accepted annotations.
- Match model labels to the project schema before import.
- Use validation subsets / honeypot frames for quality estimation.
- Use consensus or double review only on hard/uncertain frames.
- Use active selection instead of random-only annotation.
- Track correction metrics by label type, not just frame acceptance.

Sources:

- CVAT automatic annotation: https://docs.cvat.ai/docs/annotation/auto-annotation/automatic-annotation/
- CVAT automated QA / honeypots: https://docs.cvat.ai/docs/qa-analytics/auto-qa/
- CVAT QA & analytics overview: https://docs.cvat.ai/docs/qa-analytics/
- Label Studio pre-annotations: https://labelstud.io/guide/predictions.html
- Label Studio ML backend / predictions: https://labelstud.io/guide/ml.html
- Kili honeypot overview: https://docs.kili-technology.com/docs/honeypot-overview
