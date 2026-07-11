# RBCCPS Portable Annotator

1. Put raw phone images/videos in `input_raw/`.
2. Put optional detector weights at `models/detector/best.pt`.
3. Put tutorial examples in `tutorial_examples/` as image + JSON files.
4. Run `Setup-And-Launch.bat`.
5. Complete or skip the browser walkthrough, then annotate the sampled frames.

The launcher extracts sparse video frames, smart-samples a diverse review set, creates a workspace, starts the annotator, and opens the browser.

Exports are updated automatically after saves. To force export manually, run `Export-Now.bat`.

Outputs are written under:

```text
workspaces/
exports/
logs/
```

Tutorial JSON files should include:

```json
{
  "id": "example_01",
  "title": "Navigation basics",
  "lesson": "Select boxes, delete, redraw, and save.",
  "image": "example_01.jpg",
  "review": {
    "schema_version": "measurement_annotator_v1",
    "boxes": [],
    "confounder_boxes": [],
    "polygons": [],
    "measurement": {
      "lamp_status": [],
      "public_space_regions": [],
      "affected_regions": [],
      "visibility_labels": [],
      "attribution_labels": [],
      "lux_points": [],
      "qa_flags": []
    }
  }
}
```
