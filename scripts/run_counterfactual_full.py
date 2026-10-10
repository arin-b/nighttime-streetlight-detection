#!/usr/bin/env python3
"""Local bookkeeping for Codex built-in edits; no API calls or source writes.

prepare -> inspect/preflight -> begin -> ingest/fail -> report.
An IN_PROGRESS job must be reconciled after interruption, never auto-retried.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT / 'datasets/raw/mobile_night_videos/synth_images'
DEFAULT_RUN = PROJECT / 'datasets/derived/counterfactual_streetlights_full_v2'
SEED = 20260912
DISTRIBUTION = {'OFF': .20, 'FLICKER': .20, 'DIM': .25, 'SEVERE_DIM': .25, 'MIXED': .10}
MAX_ATTEMPTS = 3

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def atomic(path, text):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)

def save(run, rows):
    atomic(run/'manifest.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))
    temp = run/'manifest.csv.tmp'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in r.items()} for r in rows)
    temp.replace(run/'manifest.csv')

def load(run):
    return [json.loads(line) for line in (run/'manifest.jsonl').read_text().splitlines()]

def prepare(run):
    if (run/'manifest.jsonl').exists():
        return load(run)
    files = sorted(SOURCE_ROOT.rglob('*'))
    valid, invalid = [], []
    for path in files:
        if not path.is_file():
            continue
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}:
            continue
        try:
            with Image.open(path) as im:
                im.load()
                size, fmt = list(im.size), im.format
            valid.append((path, size, fmt))
        except Exception as exc:
            invalid.append({'path': str(path.relative_to(SOURCE_ROOT)), 'error': str(exc)})
    if not valid:
        raise ValueError('No valid sources')
    raw = {k: v*len(valid) for k,v in DISTRIBUTION.items()}
    counts = {k: int(v) for k,v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k]-counts[k], reverse=True)[:len(valid)-sum(counts.values())]:
        counts[k] += 1
    states = [k for k,v in counts.items() for _ in range(v)]
    random.Random(SEED).shuffle(states)
    rows = []
    for i, ((path, size, fmt), state) in enumerate(zip(valid, states), 1):
        rel = path.relative_to(SOURCE_ROOT)
        name = f'{i:05d}__{rel.parent.name}__{path.stem}.png'
        rows.append(dict(record_id=f'cf_{i:05d}', source_relative_path=str(rel), source_filename=path.name,
            source_sha256=digest(path), source_dimensions=size, source_format=fmt,
            output_filename=name, output_relative_path=f'images/{state}/{name}', transformation=state,
            severity={'OFF':'zero emission', 'FLICKER':'intermediate exposure of unstable lamp',
                'DIM':'clearly visible mild reduction', 'SEVERE_DIM':'strong reduction with faint residual glow',
                'MIXED':'per-lamp states recorded in target_description'}[state],
            confounder_category='UNREVIEWED', confounder_tags=[], source_video=None, source_timestamp=None,
            generation_status='PLANNED', generation_attempts=0, random_seed=None, assignment_seed=SEED,
            generation_timestamp=None, qa_status='PENDING', label_status='NO_SOURCE_ANNOTATION_LINKED',
            target_description=None, prompt=None, error=None))
    for state in DISTRIBUTION:
        (run/'images'/state).mkdir(parents=True, exist_ok=True)
    (run/'qc').mkdir(exist_ok=True)
    (run/'prompts').mkdir(exist_ok=True)
    atomic(run/'run_config.json', json.dumps(dict(source_root=str(SOURCE_ROOT), distribution=DISTRIBUTION,
        allocated_counts=counts, seed=SEED, max_attempts=MAX_ATTEMPTS, batch_size=50,
        backend='Codex built-in imagegen', generation_seed_supported=False,
        confounder_target_fraction=.40, confounder_note='Must be visually classified; all-source processing fixes prevalence.'), indent=2))
    atomic(run/'invalid_sources.json', json.dumps(invalid, indent=2))
    save(run, rows)
    return rows

def prompt(row):
    return ('Use case: lighting-weather. Input image is the edit target. Produce exactly ONE edited photograph '
        f'at {row["source_dimensions"][0]}x{row["source_dimensions"][1]}. Same exact physical scene. '
        f'Target physical streetlight(s): {row["target_description"]}. State: {row["transformation"]}; '
        f'severity: {row["severity"]}. The selected lamp emission and its attached bloom must change clearly '
        'at the lamp itself, with a visible change at normal viewing size. Preserve poles, fixture shape and location. '
        'OFF: zero emitted light and naturally absent bloom, still-visible physical fixture. DIM: visibly weaker '
        'emission with smaller, softer residual bloom. SEVERE_DIM: faint residual emission, strongly reduced bloom. '
        'FLICKER: a plausible partially illuminated instant of a failing lamp, no noise; a still image cannot show '
        'temporal flickering. MIXED: use the per-lamp states specified above. Reduce only illumination attributable '
        'to the chosen lamp, with natural falloff. Preserve exact viewpoint, framing, perspective, object positions, '
        'road markings, buildings, vehicles, pedestrians, trees, fixtures, text, signs, unrelated reflections, '
        'camera exposure, blur and grain. Keep vehicle headlights, brake lights, traffic lights, shop signs, '
        'storefronts and all unrelated lamps unchanged. No rectangular darkening, black patches, global exposure '
        'change, object removal, new objects or scene reconstruction.')

def composite_local(original, edited, regions):
    """Blend reviewed illumination regions, with exactly zero alpha outside.

    Regions are [center_x, center_y, radius_x, radius_y, solid_fraction].
    Smooth elliptical support avoids rectangular patches and has a finite edge.
    Multiple regions can describe bloom, streaks, and attributed ground spill.
    These supports are selected by visual review, not automatic lamp detection.
    """
    if original.size != edited.size:
        raise ValueError('Composite inputs must be aligned and have identical dimensions')
    if not regions:
        raise ValueError('At least one reviewed illumination region is required')
    width, height = original.size
    yy, xx = np.ogrid[:height, :width]
    alpha = np.zeros((height, width), dtype=np.float32)
    for region in regions:
        if len(region) != 5 or not all(np.isfinite(v) for v in region):
            raise ValueError('Expected five finite numbers per illumination region')
        cx, cy, rx, ry, solid = region
        if rx <= 0 or ry <= 0 or not 0 <= solid < 1 or not (0 <= cx < width and 0 <= cy < height):
            raise ValueError('Invalid illumination ellipse')
        radius = np.sqrt(((xx-cx)/rx)**2 + ((yy-cy)/ry)**2)
        ramp = np.clip((1-radius)/(1-solid), 0, 1)
        weight = ramp*ramp*(3-2*ramp)
        alpha = np.maximum(alpha, weight.astype(np.float32))
    src = np.asarray(original.convert('RGB'))
    gen = np.asarray(edited.convert('RGB'))
    result = np.rint(src.astype(np.float32)*(1-alpha[:,:,None])+gen.astype(np.float32)*alpha[:,:,None]).astype(np.uint8)
    outside = alpha == 0
    result[outside] = src[outside]
    stats = dict(outside_mask_changed_pixels=int(np.count_nonzero(np.any(result[outside] != src[outside], axis=1))),
                 mask_support_pixels=int(np.count_nonzero(~outside)),
                 changed_pixels=int(np.count_nonzero(np.any(result != src, axis=2))))
    return Image.fromarray(result), Image.fromarray(np.rint(alpha*255).astype(np.uint8)), stats

def repair(run, rows, args):
    row = next(r for r in rows if r['record_id'] == args.record_id)
    if row['generation_status'] != 'SUCCESS' or not args.notes or not args.regions:
        raise ValueError('Repair requires a saved output, reviewed regions, and reason')
    if row.get('local_compositing'):
        raise ValueError('Already composited; refusing repeated repair')
    source = SOURCE_ROOT/row['source_relative_path']
    destination = run/row['output_relative_path']
    if digest(source) != row['source_sha256']:
        raise ValueError('Source changed')
    regions = json.loads(args.regions)
    start = time.perf_counter()
    previous_hash = digest(destination)
    with Image.open(source) as original, Image.open(destination) as edited:
        result, mask, stats = composite_local(original, edited, regions)
    maskpath = run/'qc'/f'{row["record_id"]}_illumination_mask.png'
    mask.save(maskpath)
    temp = destination.with_suffix('.repair.tmp')
    with temp.open('wb') as stream:
        result.save(stream, format='PNG')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(destination)
    row['local_compositing'] = dict(method='compact_smooth_ellipses', regions=regions,
        reason=args.notes, previous_output_sha256=previous_hash, mask_path=str(maskpath.relative_to(run)),
        processing_seconds=round(time.perf_counter()-start,4), timestamp=now(), **stats)
    row.update(output_sha256=digest(destination), qa_status='PENDING',
               previous_qa_status=row['qa_status'], previous_qa_notes=row.get('qa_notes'),
               qa_notes='Local compositing applied; requires visual review')
    save(run, rows)
    print(json.dumps(row['local_compositing'],indent=2))

def act(run, rows, args):
    row = next(r for r in rows if r['record_id'] == args.record_id)
    if args.command == 'preflight':
        if not args.target or not args.category:
            raise ValueError('Target description and visual scene category are required')
        if row['generation_status'] not in {'PLANNED', 'RETRY'}:
            raise ValueError('Cannot change a dispatched job')
        row.update(target_description=args.target, confounder_category=args.category, confounder_tags=args.tags or [])
        row['prompt'] = prompt(row)
        atomic(run/'prompts'/f'{row["record_id"]}.txt', row['prompt'])
    elif args.command == 'begin':
        if row['generation_status'] not in {'PLANNED','RETRY'} or not row['prompt']:
            raise ValueError('Job must have visual preflight and be pending')
        if row['generation_attempts'] >= MAX_ATTEMPTS:
            raise ValueError('Retry limit reached')
        if digest(SOURCE_ROOT/row['source_relative_path']) != row['source_sha256']:
            raise ValueError('Source changed')
        row.update(generation_status='IN_PROGRESS', generation_attempts=row['generation_attempts']+1,
                   generation_timestamp=now())
    elif args.command == 'ingest':
        if row['generation_status'] != 'IN_PROGRESS':
            raise ValueError('Must reconcile an IN_PROGRESS job, cannot overwrite success')
        dest = run/row['output_relative_path']
        if dest.exists():
            raise ValueError('Output already exists: reconcile manually, do not regenerate')
        with Image.open(args.file) as im:
            im.load()
            row['native_output_dimensions'] = list(im.size)
            needs_resize = list(im.size) != row['source_dimensions']
            if needs_resize and not args.normalize:
                raise ValueError('Output dimensions differ; retain editor artifact for explicit normalization')
            if needs_resize:
                with dest.open('xb') as out:
                    im.convert('RGB').resize(tuple(row['source_dimensions']),Image.Resampling.LANCZOS).save(out,format='PNG')
            else:
                with args.file.open('rb') as src, dest.open('xb') as out:
                    import shutil
                    shutil.copyfileobj(src, out)
            row['resolution_normalized'] = needs_resize
        row.update(generation_status='SUCCESS', generated_artifact=str(args.file),
                   output_sha256=digest(dest), generation_timestamp=now(), error=None)
    elif args.command == 'fail':
        if row['generation_status'] != 'IN_PROGRESS':
            raise ValueError('Only dispatched jobs can fail')
        row.update(generation_status='FAILED' if row['generation_attempts'] >= MAX_ATTEMPTS else 'RETRY', error=args.error)
    elif args.command == 'qa':
        if row['generation_status'] != 'SUCCESS' or not args.verdict or not args.notes:
            raise ValueError('QA requires a saved image, verdict, and review notes')
        row.update(qa_status=args.verdict, qa_notes=args.notes)
    save(run, rows)

def report(run, rows):
    successful = [r for r in rows if r['generation_status']=='SUCCESS']
    counts = Counter(r['transformation'] for r in successful)
    conf = Counter(r['confounder_category'] for r in rows)
    data = dict(total_source_images=len(rows), successful_generations=len(successful),
        failed_generations=sum(r['generation_status']=='FAILED' for r in rows),
        status_counts=dict(Counter(r['generation_status'] for r in rows)),
        transformation_counts={k:counts[k] for k in DISTRIBUTION},
        planned_transformation_counts=dict(Counter(r['transformation'] for r in rows)),
        ordinary_count=conf['ordinary'], confounder_heavy_count=conf['confounder_heavy'],
        unreviewed_confounder_count=conf['UNREVIEWED'],
        actual_confounder_percentage=100*conf['confounder_heavy']/len(rows) if not conf['UNREVIEWED'] else None,
        qa_counts=dict(Counter(r['qa_status'] for r in successful)))
    atomic(run/'statistics_report.json', json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))

def sheet(run, rows, count):
    completed = [r for r in rows if r['generation_status']=='SUCCESS']
    random.Random(SEED).shuffle(completed)
    selected = completed[:count]
    if not selected:
        return
    canvas = Image.new('RGB',(980,300*len(selected)), '#202124')
    draw = ImageDraw.Draw(canvas)
    for i,r in enumerate(selected):
        for col,(label,path) in enumerate([('ORIGINAL',SOURCE_ROOT/r['source_relative_path']),('SYNTHETIC',run/r['output_relative_path'])]):
            with Image.open(path) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(480,270))
            canvas.paste(thumb,(col*490,i*300+30))
            draw.text((col*490+5,i*300+5),f'{r["record_id"]} {label} {r["transformation"]}',fill='white')
    canvas.save(run/'qc'/'random_contact_sheet.jpg',quality=92)
    atomic(run/'qc'/'sample_records.json',json.dumps(selected,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','next','preflight','begin','ingest','fail','qa','repair','report'])
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--record-id'); p.add_argument('--target')
    p.add_argument('--category',choices=['ordinary','confounder_heavy'])
    p.add_argument('--tags',nargs='*'); p.add_argument('--file',type=Path)
    p.add_argument('--normalize',action='store_true',help='Normalize editor output to the original pixel dimensions')
    p.add_argument('--error'); p.add_argument('--verdict',choices=['ACCEPTED','REJECTED','NEEDS_REVIEW'])
    p.add_argument('--notes'); p.add_argument('--batch-size',type=int,default=50)
    p.add_argument('--regions',help='Reviewed illumination ellipses as JSON: [[cx,cy,rx,ry,solid_fraction],...]')
    a=p.parse_args()
    if a.run_dir.resolve().is_relative_to(SOURCE_ROOT.resolve()):
        raise ValueError('Output must not be inside source directory')
    a.run_dir.mkdir(parents=True,exist_ok=True)
    with (a.run_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=prepare(a.run_dir)
        if a.command=='next':
            pending=[r for r in rows if r['generation_status'] in {'PLANNED','RETRY'}]
            random.Random(SEED+1).shuffle(pending)
            print(json.dumps(pending[:a.batch_size],indent=2))
        elif a.command=='repair':
            repair(a.run_dir,rows,a)
        elif a.command in {'preflight','begin','ingest','fail','qa'}:
            act(a.run_dir,rows,a)
        else:
            report(a.run_dir,rows)
            sheet(a.run_dir,rows,a.batch_size)

if __name__=='__main__':
    main()
