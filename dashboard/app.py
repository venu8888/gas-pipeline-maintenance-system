from pathlib import Path
import streamlit as st
from storage.database import InspectionDatabase
st.set_page_config(page_title='Gas Pipeline Inspection',layout='wide'); st.title('GAS PIPELINE INSPECTION')
db_path=st.sidebar.text_input('Database','inspection/events.db'); session=st.sidebar.text_input('Session ID','')
try:
 db=InspectionDatabase(db_path); events=db.defects(session or None); st.metric('Confirmed defects',len(events)); st.caption('Distance source: simulation')
 if events:
  st.subheader('Live event feed')
  for e in events:
   st.write(f"{e['defect_id']} | {e['class_name']} | {e['distance_m']:.2f} m | {e['confidence']:.0%}")
   if e.get('image_path') and Path(e['image_path']).exists(): st.image(e['image_path'],caption=e['class_name'])
 else: st.info('No confirmed defects recorded.')
 db.close()
except Exception as exc: st.error(str(exc))
