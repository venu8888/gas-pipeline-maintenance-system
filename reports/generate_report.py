from __future__ import annotations
import argparse, html
from pathlib import Path
from storage.database import InspectionDatabase
def generate(db_path,session_id,output):
 db=InspectionDatabase(db_path); events=db.defects(session_id); session=db.connection.execute('SELECT * FROM inspection_sessions WHERE session_id=?',(session_id,)).fetchone(); rows=''.join(f"<tr><td>{html.escape(str(e['defect_id']))}</td><td>{html.escape(str(e['class_name']))}</td><td>{e['distance_m']:.2f}</td><td>{e['confidence']:.3f}</td><td>{html.escape(str(e.get('image_path') or ''))}</td></tr>" for e in events); text=f"<html><body><h1>Gas Pipeline Inspection</h1><p>Inspection ID: {html.escape(session_id)}</p><p>Confirmed defects: {len(events)}</p><p>Distance: {session['distance'] if session else 'unknown'} m</p><p>Start: {session['start_time'] if session else 'unknown'} | End: {session['end_time'] if session else 'unknown'}</p><table><tr><th>ID</th><th>Class</th><th>Distance (m)</th><th>Confidence</th><th>Image</th></tr>{rows}</table></body></html>"; Path(output).parent.mkdir(exist_ok=True); Path(output).write_text(text,encoding='utf-8'); db.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--database',default='inspection/events.db'); p.add_argument('--session',required=True); p.add_argument('--output',default='reports/inspection_report.html'); a=p.parse_args(); generate(a.database,a.session,a.output)
