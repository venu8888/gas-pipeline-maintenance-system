from __future__ import annotations
from .tracker import IOUTracker
from .event_manager import EventManager
class FrameProcessor:
 def __init__(self,confirmation_frames=3,iou_threshold=.3,max_age=10): self.tracker=IOUTracker(iou_threshold,max_age); self.events=EventManager(confirmation_frames)
 def process(self,detections,distance_m,timestamp=None): return self.events.update(self.tracker.update(detections),distance_m,'simulation',timestamp)
