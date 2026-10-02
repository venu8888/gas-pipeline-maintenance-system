"""Temporal confirmation and duplicate suppression."""
from __future__ import annotations
from dataclasses import dataclass
from .tracker import Track,iou
@dataclass
class Candidate:
 track_id:int; hits:int=0; best:Track|None=None
class EventManager:
 def __init__(self,confirmation_frames=3): self.confirmation_frames=confirmation_frames; self.candidates={}; self.confirmed=set(); self.counter=0
 def update(self,tracks,distance_m,distance_source='simulation',timestamp=None):
  events=[]
  for t in tracks:
   c=self.candidates.setdefault(t.track_id,Candidate(t.track_id)); c.hits=max(c.hits,t.hits); c.best=t if c.best is None or t.detection.confidence>c.best.detection.confidence else c.best
   if c.hits>=self.confirmation_frames and t.track_id not in self.confirmed:
    self.counter+=1; self.confirmed.add(t.track_id); d=c.best.detection; events.append({'defect_id':f'D{self.counter:03d}','class_id':d.class_id,'class_name':d.class_name,'confidence':d.confidence,'distance_m':float(distance_m),'distance_source':distance_source,'timestamp':timestamp,'bbox':list(d.bbox),'status':'confirmed','track_id':t.track_id})
  return events
