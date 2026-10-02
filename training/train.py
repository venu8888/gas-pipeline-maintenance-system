"""Train YOLO with explicit dependency and dataset errors."""
from __future__ import annotations
import argparse, datetime, json, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import load_config
from training.dataset_audit import audit, prepare_dataset
NAMES=['Deformation','Obstacle','Rupture','Disconnect','Misalignment','Deposition']
def require_ultralytics():
    try: from ultralytics import YOLO; return YOLO
    except ImportError as e: raise RuntimeError('Training requires ultralytics. Install requirements.txt first.') from e
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--epochs',type=int); ap.add_argument('--imgsz',type=int); ap.add_argument('--data',default=None); ap.add_argument('--batch',default=None); ap.add_argument('--config',default='config/config.yaml'); a=ap.parse_args(); c=load_config(a.config)
    weights=a.model or c['model']['weights']; epochs=a.epochs or c['training']['epochs']; imgsz=a.imgsz or c['training']['image_size']; data=Path(a.data or c['dataset']['yaml'])
    if not data.exists(): raise FileNotFoundError(f'Dataset YAML not found: {data}. Audit/prepare dataset first.')
    dataset_root=prepare_dataset(Path(c['dataset']['root']))
    if not dataset_root.exists(): raise FileNotFoundError(f'Dataset root not found: {dataset_root}. Run the download utility or update config/config.yaml.')
    audit_result=audit(dataset_root)
    if not audit_result['mapping_matches_expected'] or audit_result['total_annotations'] == 0 or audit_result['invalid_class_ids'] or audit_result['malformed_labels'] or audit_result['invalid_bounding_boxes']:
        raise RuntimeError('Dataset annotation/class mapping verification failed. Review reports/dataset_audit.json; training was not started.')
    YOLO=require_ultralytics(); model=YOLO(weights)
    kwargs={'data':str(data),'epochs':epochs,'imgsz':imgsz,'batch':a.batch or c['training']['batch_size'],'patience':c['training']['patience'],'pretrained':c['training']['pretrained'],'seed':c['training']['seed'],'project':'training_outputs','name':f'{Path(weights).stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}','exist_ok':False}
    if kwargs['batch']=='auto': kwargs['batch']=-1
    results=model.train(**kwargs)
    save_dir=Path(getattr(results,'save_dir',kwargs['project']) )
    weights_dir=save_dir/'weights'
    models_dir=Path('models'); models_dir.mkdir(exist_ok=True)
    for source, target in ((weights_dir/'best.pt', models_dir/'best.pt'), (weights_dir/'last.pt', models_dir/'last.pt')):
        if source.exists(): shutil.copy2(source,target)
    out=Path('reports'); out.mkdir(exist_ok=True); record={'experiment_id':kwargs['name'],'model':weights,'dataset':str(data),'epochs':epochs,'image_size':imgsz,'batch_size':kwargs['batch'],'training_date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'classes':NAMES}
    try:
        import yaml
        (out/'training_config.yaml').write_text(yaml.safe_dump(record,sort_keys=False),encoding='utf-8')
    except ImportError:
        (out/'training_config.yaml').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(results)
if __name__=='__main__': main()
