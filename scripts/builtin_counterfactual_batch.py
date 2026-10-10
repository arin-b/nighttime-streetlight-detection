"""Persistent bookkeeping for Codex's built-in image editor; no network/API calls.

Run init, then start; inspect each claimed source with view_image, call built-in
image_gen once with that path and the returned prompt, and finish with its file.
This script does not perform synthetic editing itself.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'synthetic_output'
MANIFEST = OUT / 'manifest.json'
SOURCE = ROOT / 'datasets/raw/mobile_night_videos/synth_images'
STATES = ['mild_dim', 'severe_dim', 'off', 'flicker']
RULES = '''Use case: lighting-weather. Input image 1 is the EDIT TARGET, an actual photograph. Perform image-to-image editing of this exact photograph; never recreate the scene. Produce exactly ONE counterpart, same 1280x720 resolution, framing, aspect ratio and instant. Change only a plausible subset of EXISTING roadside streetlights and their directly attributable local illumination. Preserve physical poles, fixtures and mounting arms even for OFF. Preserve every other pixel wherever no lamp edit is needed. Preserve camera, road geometry, lane markings, vehicles and positions, headlights, brake lights, license plates, traffic signals, buildings, windows, signs, shops, advertisements, vegetation, pedestrians, sky, unrelated poles and reflections. Do not confuse these lights, vehicle glare, decorative lights or billboards with streetlights. Never add, delete, move or redesign objects. Never globally darken, change exposure or apply a night filter/color grade. Minimum localized visual change; preservation takes priority over dramatic editing. No labels or watermarks.'''
STATE_RULES = {
 'mild_dim': 'Moderately reduce selected streetlight brightness and halos with a small natural reduction in their local illumination. They must clearly remain ON.',
 'severe_dim': 'Substantially dim selected streetlights without fully switching off: retain a faint residual glow and visible fixtures. Naturally reduce only their corresponding local illumination.',
 'off': 'Switch selected lamps OFF: strongly suppress lamp glow and halos and their attributable local illumination. Retain visible pole, luminaire, fixture and mounting arm. Do not turn the region black; preserve other-source illumination.',
 'flicker': 'One still photographic instant of unstable streetlights: different selected lamps have different reduced brightness, some moderate, some severe, with others unchanged. No trails, duplicate lamps, blinking graphics, temporal composites, streaks or animation artifacts.'}

def now():
 return datetime.now(timezone.utc).isoformat()

def save(m):
 OUT.mkdir(exist_ok=True)
 tmp = MANIFEST.with_suffix('.tmp')
 with tmp.open('w') as f:
  json.dump(m, f, indent=2); f.flush(); os.fsync(f.fileno())
 os.replace(tmp, MANIFEST)

def init():
 if MANIFEST.exists(): return json.loads(MANIFEST.read_text())
 files = sorted(p for p in SOURCE.iterdir() if p.is_file() and re.fullmatch(r'frame_\d+(?: \(\d+\))?\.jpg', p.name, re.I))
 rng = random.Random(42)
 states = STATES * (len(files)//4) + rng.sample(STATES, len(files)%4)
 rng.shuffle(states)
 rows=[]
 for p,s in zip(files,states):
  with Image.open(p) as im:
   im.verify()
  with Image.open(p) as im: size=list(im.size); fmt=im.format
  rows.append(dict(source_filename=p.name, assigned_state=s, output_filename=f'{p.stem}_{s}{p.suffix}', status='pending', attempt_count=0, failure_reason=None, source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), source_dimensions=size, source_format=fmt, attempts=[]))
 m=dict(version=1,seed=42,source_directory=str(SOURCE),method='Codex built-in image_gen only',created_at=now(),rules=RULES,state_rules=STATE_RULES,entries=rows,batches=[])
 save(m);return m

def report(m):
 b=m['batches'][-1] if m['batches'] else {'indices':[]}
 rows=[m['entries'][i] for i in b['indices']]
 attempted=[r for r in rows if any(a['batch_id']==b.get('id') for a in r['attempts'])]
 successes=[r for r in attempted if r['status']=='success']
 result=dict(images_attempted=len(attempted),successful_images=len(successes),failed_images=sum(r['status']=='failed' for r in attempted),retry_count=sum(max(0,sum(a['batch_id']==b.get('id') for a in r['attempts'])-1) for r in rows),state_successes={s:sum(r['assigned_state']==s for r in successes) for s in STATES},total_completed=sum(r['status']=='success' for r in m['entries']))
 result['total_remaining']=len(m['entries'])-result['total_completed']
 (OUT/'qc').mkdir(exist_ok=True)
 (OUT/'qc'/f"batch_{b.get('id',0):03d}_report.json").write_text(json.dumps(result,indent=2))
 reviewable=[r for r in attempted if r['status']=='success' or r.get('rejected_artifact')]
 sample=random.Random(42).sample(reviewable,min(20,len(reviewable)))
 if sample:
  canvas=Image.new('RGB',(800, len(sample)*250),'#202020'); draw=ImageDraw.Draw(canvas)
  for j,r in enumerate(sample):
   draw.text((8,j*250+5),f"{r['source_filename']} | {r['assigned_state']} | {r['status']}   SOURCE / EDIT",fill='white')
   edit=Path(r.get('output_path',str(OUT/r['output_filename']))) if r['status']=='success' else Path(r['rejected_artifact'])
   for k,p in enumerate([SOURCE/r['source_filename'],edit]):
    with Image.open(p) as im: canvas.paste(ImageOps.contain(im.convert('RGB'),(392,220)),(k*400+4,j*250+26))
  canvas.save(OUT/'qc'/f"batch_{b['id']:03d}_contact_sheet.jpg",quality=90)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('command',choices=['init','start','extend','claim','success','failed','report'])
 parser.add_argument('--index',type=int);parser.add_argument('--artifact');parser.add_argument('--reason')
 args=parser.parse_args();m=init()
 if args.command=='init':
  print(json.dumps(dict(count=len(m['entries']),states=dict(collections.Counter(r['assigned_state'] for r in m['entries'])))));return
 if args.command in ('start','extend'):
  if args.command=='extend' and m['batches']:
   b=m['batches'][-1]
   existing=set(b['indices'])
   b['indices'].extend(i for i,r in enumerate(m['entries']) if i not in existing and r['status'] in ('pending','processing'))
   b['extended_at']=now();save(m);print(json.dumps(b));return
  if not m['batches'] or all(m['entries'][i]['status'] in ['success','failed'] for i in m['batches'][-1]['indices']):
   m['batches'].append(dict(id=len(m['batches'])+1,started_at=now(),indices=[i for i,r in enumerate(m['entries']) if r['status']=='pending']));save(m)
  print(json.dumps(m['batches'][-1]));return
 if args.command=='report': print(json.dumps(report(m)));return
 b=m['batches'][-1]
 if args.command=='claim':
  active=[i for i in b['indices'] if m['entries'][i]['status']=='processing']
  if active: raise RuntimeError(f'Unfinished attempt(s): {active}; recover artifact or mark failed before proceeding')
  i=next((i for i in b['indices'] if m['entries'][i]['status']=='pending'),None)
  if i is None: print('null');return
  r=m['entries'][i]
  if hashlib.sha256((SOURCE/r['source_filename']).read_bytes()).hexdigest()!=r['source_sha256']: raise RuntimeError('Source changed')
  r['status']='processing';r['attempt_count']+=1;r['attempts'].append(dict(batch_id=b['id'],started_at=now(),status='processing'));save(m)
  print(json.dumps(dict(index=i,path=str(SOURCE/r['source_filename']),state=r['assigned_state'],prompt=RULES+'\nAssigned state: '+r['assigned_state']+'. '+STATE_RULES[r['assigned_state']])));return
 r=m['entries'][args.index]
 if args.index not in b['indices'] or r['status']!='processing': raise RuntimeError('Entry is not a current processing attempt')
 if args.command=='success':
  p=Path(args.artifact);dest=OUT/r['assigned_state']/r['output_filename'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists(): raise RuntimeError('Output exists; recover explicitly, never overwrite')
  with Image.open(p) as im:
   # Acceptance is intentionally unconditional for this run; preserve the
   # generated artifact dimensions per the user's no-QA instruction.
   tmp=dest.with_suffix('.tmp');im.convert('RGB').save(tmp,format='JPEG',quality=100,subsampling=0)
  os.replace(tmp,dest);r['artifact_path']=str(p);r['output_path']=str(dest);r['output_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();r['failure_reason']=None
 else:
  if not args.reason: raise RuntimeError('Failure reason required')
  r['failure_reason']=args.reason
  if args.artifact:
   dest=OUT/'qc'/'rejected'/(Path(r['output_filename']).stem+'.png')
   dest.parent.mkdir(exist_ok=True,parents=True)
   if dest.exists(): raise RuntimeError('Rejected artifact already exists')
   shutil.copy2(args.artifact,dest);r['rejected_artifact']=str(dest)
 r['status']=args.command;r['attempts'][-1].update(status=args.command,finished_at=now(),failure_reason=r['failure_reason']);save(m)
 print(json.dumps(dict(index=args.index,status=r['status'])))

if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 with (OUT/'.workflow.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  main()
