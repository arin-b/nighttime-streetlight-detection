import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import numpy as np

spec = importlib.util.spec_from_file_location('workflow', Path(__file__).resolve().parents[1]/'scripts/run_counterfactual_full.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def test_composite_preserves_unrelated_pixels(self):
        original=Image.new('RGB',(100,100),(200,160,120))
        edited=Image.new('RGB',(100,100),(10,20,30))
        result,mask,stats=workflow.composite_local(original,edited,[[50,50,20,30,.4]])
        arr=np.asarray(result)
        self.assertEqual(tuple(arr[50,50]),(10,20,30))
        self.assertEqual(tuple(arr[0,0]),(200,160,120))
        self.assertEqual(tuple(arr[50,70]),(200,160,120))
        self.assertEqual(stats['outside_mask_changed_pixels'],0)
        self.assertTrue(10 < arr[50,65,0] < 200)
        self.assertEqual(mask.getpixel((70,50)),0)
        with self.assertRaises(ValueError):
            workflow.composite_local(original,edited,[[50,50,-20,30,.4]])

    def test_resume_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            workflow.SOURCE_ROOT=base/'sources'
            workflow.SOURCE_ROOT.mkdir()
            source=workflow.SOURCE_ROOT/'one.jpg'
            Image.new('RGB',(32,32),'white').save(source)
            original=workflow.digest(source)
            run=base/'run'
            run.mkdir()
            rows=workflow.prepare(run)
            row=rows[0]
            args=SimpleNamespace(record_id=row['record_id'], command='preflight', target='lamp at center', category='ordinary', tags=[])
            workflow.act(run,rows,args)
            args.command='begin'
            workflow.act(run,rows,args)
            with self.assertRaises(ValueError):
                workflow.act(run,rows,args)
            args.command='ingest'
            args.file=source
            args.normalize=False
            workflow.act(run,rows,args)
            with self.assertRaises(ValueError):
                workflow.act(run,rows,args)
            reloaded=workflow.prepare(run)
            self.assertEqual(reloaded[0]['generation_attempts'],1)
            self.assertEqual(reloaded[0]['generation_status'],'SUCCESS')
            self.assertEqual(original,workflow.digest(source))
            self.assertEqual(len(list((run/'images').rglob('*.png'))),1)


if __name__=='__main__':
    unittest.main()
