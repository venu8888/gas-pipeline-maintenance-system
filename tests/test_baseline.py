import json, tempfile, unittest
from pathlib import Path
from inference.detector import Detection
from inference.tracker import IOUTracker, iou
from inference.event_manager import EventManager
from simulation.simulated_encoder import SimulatedDistanceSensor
from storage.database import InspectionDatabase
from training.dataset_audit import audit, prepare_dataset
class BaselineTests(unittest.TestCase):
 def test_iou_and_duplicate_tracking(self):
  self.assertGreater(iou((0,0,10,10),(1,1,9,9)),.5); tr=IOUTracker(.3); d=Detection(2,'Rupture',.9,(0,0,10,10)); a=tr.update([d]); b=tr.update([Detection(2,'Rupture',.8,(1,1,11,11))]); self.assertEqual(a[0].track_id,b[0].track_id)
 def test_confirmation(self):
  m=EventManager(3); t=[]
  for n in range(3):
   t=IOUTracker(.3).update([]) if False else t
  tr=IOUTracker(.3)
  for _ in range(2): self.assertEqual(m.update(tr.update([Detection(2,'Rupture',.8,(0,0,10,10))]),1),[])
  self.assertEqual(len(m.update(tr.update([Detection(2,'Rupture',.9,(0,0,10,10))]),1)),1)
 def test_distance(self): self.assertAlmostEqual(SimulatedDistanceSensor(.2).distance_at(10),2)
 def test_database(self):
  with tempfile.TemporaryDirectory() as d:
   db=InspectionDatabase(Path(d)/'x.db'); db.create_session('s','now'); db.add_defect({'defect_id':'D001','class_id':2,'class_name':'Rupture','confidence':.9,'distance_m':1,'distance_source':'simulation','timestamp':'now','bbox':[0,0,1,1],'status':'confirmed'},'s'); self.assertEqual(len(db.defects('s')),1); db.close()
 def test_audit(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'a.jpg').write_bytes(b'not-image'); r=audit(p); self.assertEqual(r['total_images'],1); self.assertEqual(len(r['corrupted_images']),1)
 def test_prepare_dataset_for_nested_image_label_layout(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); img_dir=root/'data'/'images'/'images'/'train'; lbl_dir=root/'data'/'labels'/'labels'/'train'; img_dir.mkdir(parents=True); lbl_dir.mkdir(parents=True)
   img_file=img_dir/'000001.jpg'; img_file.write_bytes(b'img')
   lbl_file=lbl_dir/'000001.txt'; lbl_file.write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
   prepared=prepare_dataset(root/'data'/'raw'/'train')
   self.assertTrue((prepared/'000001.jpg').exists()); self.assertTrue((prepared/'000001.txt').exists()); self.assertEqual(audit(prepared)['total_images'],1)
if __name__=='__main__': unittest.main()
