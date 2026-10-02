from __future__ import annotations
import time
class DistanceSensor:
 def get_distance_m(self)->float: raise NotImplementedError
class SimulatedDistanceSensor(DistanceSensor):
 def __init__(self,speed_mps=.2): self.speed_mps=float(speed_mps); self.started=time.monotonic(); self.offset=0
 def reset(self): self.started=time.monotonic(); self.offset=0
 def get_distance_m(self): return self.offset+self.speed_mps*(time.monotonic()-self.started)
 def distance_at(self,elapsed_seconds): return self.offset+self.speed_mps*float(elapsed_seconds)
