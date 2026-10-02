"""YOLO inference adapter for images, directories, and videos."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
@dataclass
class Detection:
 class_id:int; class_name:str; confidence:float; bbox:tuple[float,float,float,float]; track_id:int|None=None
NAMES=['Deformation','Obstacle','Rupture','Disconnect','Misalignment','Deposition']
def parse_result(result):
 out=[]; boxes=getattr(result,'boxes',None)
 if boxes is None: return out
 xyxy=boxes.xyxy.cpu().tolist(); conf=boxes.conf.cpu().tolist(); cls=boxes.cls.cpu().tolist(); ids=boxes.id.cpu().tolist() if getattr(boxes,'id',None) is not None else [None]*len(xyxy)
 for b,c,k,t in zip(xyxy,conf,cls,ids):
  cid=int(k); out.append(Detection(cid,NAMES[cid] if cid<len(NAMES) else str(cid),float(c),tuple(float(x) for x in b),int(t) if t is not None else None))
 return out
def load_model(weights='yolo26s.pt'):
 try: from ultralytics import YOLO
 except ImportError as e: raise RuntimeError('Inference requires ultralytics; install requirements.txt.') from e
 try: return YOLO(weights)
 except Exception as e: raise RuntimeError(f'Unable to load model {weights}: {e}') from e
def infer(source,weights='yolo26s.pt',confidence=.60,iou=.50,save=False):
 model=load_model(weights); return model.predict(source=source,conf=confidence,iou=iou,save=save)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--weights',default='models/best.pt'); p.add_argument('--confidence',type=float,default=.60); p.add_argument('--iou',type=float,default=.50); p.add_argument('--save',action='store_true'); a=p.parse_args()
 for result in infer(a.source,a.weights,a.confidence,a.iou,a.save):
  for d in parse_result(result): print({'class':d.class_name,'confidence':round(d.confidence,4),'bbox':d.bbox})
if __name__=='__main__': main()
