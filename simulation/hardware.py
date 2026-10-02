from __future__ import annotations
class ObstacleSensor:
 def read_distance(self): raise NotImplementedError
class CameraSource:
 def read(self): raise NotImplementedError
class VideoFileCamera(CameraSource):
 def __init__(self,source): self.source=source
 def read(self):
  import cv2
  cap=cv2.VideoCapture(str(self.source))
  while True:
   ok,frame=cap.read()
   if not ok: break
   yield frame
  cap.release()
