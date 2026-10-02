from __future__ import annotations
import argparse, datetime, uuid, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import load_config
from inference.detector import load_model, parse_result
from inference.frame_processor import FrameProcessor
from simulation.simulated_encoder import SimulatedDistanceSensor
from storage.database import InspectionDatabase
from storage.image_store import ConfirmedImageStore
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--weights',default='models/best.pt'); p.add_argument('--confidence',type=float,default=.60); p.add_argument('--display',action='store_true'); a=p.parse_args(); c=load_config(); model=load_model(a.weights); cap=None
 try:
  import cv2
  cap=cv2.VideoCapture(a.source); fps=cap.get(cv2.CAP_PROP_FPS) or 30; session=uuid.uuid4().hex[:12]; db=InspectionDatabase(Path(c['storage']['root'])/session/c['storage']['database']); db.create_session(session,datetime.datetime.now(datetime.timezone.utc).isoformat()); store=ConfirmedImageStore(c['storage']['root']); processor=FrameProcessor(c['tracking']['confirmation_frames'],c['tracking']['iou_threshold'],c['tracking']['max_age']); sensor=SimulatedDistanceSensor(c['simulation']['rover_speed_mps']); frame_no=0
  while True:
   ok,frame=cap.read()
   if not ok: break
   frame_no+=1; results=model.predict(frame,conf=a.confidence,iou=c['model']['iou'],verbose=False); detections=parse_result(results[0]); events=processor.process(detections,sensor.distance_at(frame_no/fps),datetime.datetime.now(datetime.timezone.utc).isoformat())
   for event in events:
    path=store.save(session,event['defect_id'],frame); db.add_defect(event,session,path)
   if a.display:
    cv2.imshow('replay',frame)
    if cv2.waitKey(1)&0xff==27: break
  db.finish_session(session, datetime.datetime.now(datetime.timezone.utc).isoformat(), sensor.distance_at(frame_no/fps))
  print('session_id=',session,'distance_m=',sensor.distance_at(frame_no/fps),'defects=',len(db.defects(session)))
 finally:
  if cap is not None: cap.release()
  try: cv2.destroyAllWindows()
  except Exception: pass
if __name__=='__main__': main()
