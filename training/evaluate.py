"""Evaluate a trained YOLO model; metrics are copied only from Ultralytics results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
NAMES=['Deformation','Obstacle','Rupture','Disconnect','Misalignment','Deposition']
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--weights',default='models/best.pt'); ap.add_argument('--data',default='data/dataset.yaml'); ap.add_argument('--split',default='test'); ap.add_argument('--imgsz',type=int,default=640); a=ap.parse_args()
 try:
  from ultralytics import YOLO
 except ImportError as e: raise RuntimeError('Evaluation requires ultralytics.') from e
 if not Path(a.weights).exists(): raise FileNotFoundError(a.weights)
 r=YOLO(a.weights).val(data=a.data,split=a.split,imgsz=a.imgsz,plots=True,project='training_outputs',name='evaluation')
 box=getattr(r,'box',None); metrics={}
 if box:
  for key in ('mp','mr','map50','map'):
   val=getattr(box,key,None)
   if val is not None: metrics[{'mp':'precision','mr':'recall','map50':'mAP50','map':'mAP50-95'}[key]]=float(val)
  apv=getattr(box,'ap',None)
  if apv is not None: metrics['per_class_AP']={NAMES[i] if i<len(NAMES) else str(i):float(v) for i,v in enumerate(apv)}
  if 'precision' in metrics and 'recall' in metrics and metrics['precision']+metrics['recall']>0: metrics['F1']=2*metrics['precision']*metrics['recall']/(metrics['precision']+metrics['recall'])
 Path('reports').mkdir(exist_ok=True); Path('reports/evaluation_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
