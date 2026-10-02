from __future__ import annotations
import json, sqlite3
from pathlib import Path
class InspectionDatabase:
 def __init__(self,path='inspection/events.db'):
  self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.connection=sqlite3.connect(self.path); self.connection.row_factory=sqlite3.Row; self._init()
 def _init(self):
  self.connection.executescript("""CREATE TABLE IF NOT EXISTS inspection_sessions(session_id TEXT PRIMARY KEY,start_time TEXT NOT NULL,end_time TEXT,distance REAL NOT NULL DEFAULT 0,status TEXT NOT NULL);CREATE TABLE IF NOT EXISTS defect_events(defect_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,class_id INTEGER NOT NULL,class_name TEXT NOT NULL,confidence REAL NOT NULL,distance_m REAL NOT NULL,distance_source TEXT NOT NULL,timestamp TEXT NOT NULL,image_path TEXT,bbox TEXT,status TEXT NOT NULL,FOREIGN KEY(session_id) REFERENCES inspection_sessions(session_id));CREATE TABLE IF NOT EXISTS system_events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,timestamp TEXT,event_type TEXT NOT NULL,payload TEXT NOT NULL)"""); self.connection.commit()
 def create_session(self,session_id,start_time,status='running'): self.connection.execute('INSERT INTO inspection_sessions(session_id,start_time,status) VALUES(?,?,?)',(session_id,start_time,status)); self.connection.commit()
 def finish_session(self,session_id,end_time,distance,status='completed'): self.connection.execute('UPDATE inspection_sessions SET end_time=?,distance=?,status=? WHERE session_id=?',(end_time,distance,status,session_id)); self.connection.commit()
 def add_defect(self,event,session_id,image_path=None): self.connection.execute('INSERT OR REPLACE INTO defect_events VALUES(?,?,?,?,?,?,?,?,?,?,?)',(event['defect_id'],session_id,event['class_id'],event['class_name'],event['confidence'],event['distance_m'],event['distance_source'],event['timestamp'],image_path or event.get('image_path'),json.dumps(event['bbox']),event['status'])); self.connection.commit()
 def defects(self,session_id=None):
  q='SELECT * FROM defect_events'; args=()
  if session_id: q+=' WHERE session_id=?'; args=(session_id,)
  return [dict(r) for r in self.connection.execute(q+' ORDER BY timestamp DESC',args)]
 def close(self): self.connection.close()
