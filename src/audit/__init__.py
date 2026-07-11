"""Comprehensive streetlight audit pipeline.

Implements: YOLO26 detection → BoT-SORT/ByteTrack tracking → multi-cue
filtering → brightness measurement → temporal aggregation → audit report
generation → location-prior memory → evaluation metrics.

Usage:
    python -m audit_pipeline.run_audit --video VIDEO --model MODEL [options]
"""
