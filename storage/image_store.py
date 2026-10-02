from __future__ import annotations
from pathlib import Path
class ConfirmedImageStore:
 def __init__(self,root='inspection'): self.root=Path(root)
 def save(self,session_id,defect_id,frame,extension='.jpg'):
  target=self.root/session_id/'defects'; target.mkdir(parents=True,exist_ok=True); path=target/(defect_id+extension)
  if hasattr(frame,'save'): frame.save(path)
  else:
   import cv2
   if not cv2.imwrite(str(path),frame): raise OSError('OpenCV could not write image')
  return path
