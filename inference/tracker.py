"""Lightweight dependency-free IoU tracker for replay; YOLO track IDs are accepted when available."""
from __future__ import annotations
from dataclasses import dataclass
from .detector import Detection
def iou(a,b):
 x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/(aa+ab-inter) if aa+ab-inter else 0
@dataclass
class Track:
 track_id:int; detection:Detection; age:int=0; hits:int=1
class IOUTracker:
 def __init__(self,threshold=.3,max_age=10): self.threshold=threshold; self.max_age=max_age; self.next_id=1; self.tracks={}
 def update(self,detections):
  used=set()
  for t in list(self.tracks.values()):
   choices=[(iou(t.detection.bbox,d.bbox),i,d) for i,d in enumerate(detections) if i not in used and d.class_id==t.detection.class_id]; best=max(choices,default=(0,None,None))
   if best[0]>=self.threshold: t.detection=best[2]; t.age=0; t.hits+=1; used.add(best[1])
   else: t.age+=1
  for i,d in enumerate(detections):
   if i not in used:
    tid=d.track_id or self.next_id; self.next_id=max(self.next_id,tid+1); self.tracks[tid]=Track(tid,d)
  self.tracks={k:v for k,v in self.tracks.items() if v.age<=self.max_age}
  return list(self.tracks.values())
