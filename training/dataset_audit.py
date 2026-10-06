"""Read-only YOLO dataset audit; never changes the source dataset."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys
EXPECTED_NAMES={0:'Deformation',1:'Obstacle',2:'Rupture',3:'Disconnect',4:'Misalignment',5:'Deposition'}
IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def image_info(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im: return {'width':im.width,'height':im.height,'format':im.format}
    except Exception:
        try:
            import cv2
            im=cv2.imread(str(path)); return {'width':int(im.shape[1]),'height':int(im.shape[0]),'format':path.suffix.lower()[1:]} if im is not None else None
        except Exception: return None

def audit(root: Path, fast: bool = True) -> dict[str,Any]:
    root=Path(root); images=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels=sorted(p for p in root.rglob('*.txt') if p.is_file())
    image_names={p.stem for p in images}; label_by_stem={p.stem:p for p in labels}
    ann_total=0; ann_per=Counter(); img_per=Counter(); malformed=[]; invalid_ids=[]; invalid_boxes=[]; empty=[]; missing=[]; dims=Counter(); formats=Counter(); corrupted=[]; hashes=defaultdict(list)
    for im in images:
        if not fast:
            info=image_info(im)
            if not info: corrupted.append(str(im)); continue
            dims[f"{info['width']}x{info['height']}"]+=1; formats[str(info['format']).lower()]+=1
            try: hashes[hashlib.sha256(im.read_bytes()).hexdigest()].append(str(im))
            except OSError: pass
        lp=label_by_stem.get(im.stem)
        if not lp: missing.append(str(im)); continue
        lines=[x.strip() for x in lp.read_text(errors='replace').splitlines() if x.strip()]
        if not lines: empty.append(str(lp)); continue
        image_classes=set()
        for no,line in enumerate(lines,1):
            parts=line.split()
            if len(parts)!=5:
                malformed.append({'file':str(lp),'line':no,'value':line}); continue
            try: cid=int(parts[0]); vals=[float(x) for x in parts[1:]]
            except ValueError: malformed.append({'file':str(lp),'line':no,'value':line}); continue
            ann_total+=1; ann_per[cid]+=1; image_classes.add(cid)
            if cid not in EXPECTED_NAMES: invalid_ids.append({'file':str(lp),'line':no,'class_id':cid})
            if not all(0<=v<=1 for v in vals) or vals[2]<=0 or vals[3]<=0: invalid_boxes.append({'file':str(lp),'line':no,'values':vals})
        for cid in image_classes:
            img_per[cid]+=1
    dup=[v for v in hashes.values() if len(v)>1]
    split=Counter()
    for im in images:
        rel=im.relative_to(root).parts
        split[next((x for x in rel if x.lower() in ('train','val','valid','test')), 'unspecified')]+=1
    return {'root':str(root),'expected_class_mapping':EXPECTED_NAMES,'actual_class_ids':sorted(ann_per),'actual_class_mapping':{str(k):EXPECTED_NAMES.get(k,'UNKNOWN') for k in sorted(ann_per)},'mapping_matches_expected':sorted(ann_per)==list(EXPECTED_NAMES),'total_images':len(images),'total_annotation_files':len(labels),'total_annotations':ann_total,'images_per_class':{str(k):v for k,v in sorted(img_per.items())},'annotations_per_class':{str(k):v for k,v in sorted(ann_per.items())},'image_dimensions':dict(dims),'image_formats':dict(formats),'split':dict(split),'missing_labels':missing,'empty_labels':empty,'malformed_labels':malformed,'invalid_class_ids':invalid_ids,'invalid_bounding_boxes':invalid_boxes,'duplicate_images':dup,'corrupted_images':corrupted,'class_imbalance':{'max_annotations':max(ann_per.values(),default=0),'min_annotations':min(ann_per.values(),default=0),'classes_with_annotations':len(ann_per)}}

def write_reports(result:dict[str,Any], json_path=Path('reports/dataset_audit.json'), csv_path=Path('reports/dataset_audit.csv')):
    json_path.parent.mkdir(parents=True,exist_ok=True); json_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['metric','value'])
        for k,v in result.items(): w.writerow([k,json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path); ap.add_argument('--json',type=Path,default=Path('reports/dataset_audit.json')); ap.add_argument('--csv',type=Path,default=Path('reports/dataset_audit.csv')); a=ap.parse_args()
    root=a.root
    if root is None:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from config import load_config
        root=Path(load_config()['dataset']['root'])
    if not root.exists():
        raise FileNotFoundError(f'Dataset root not found: {root}. Pass --root or update config/config.yaml.')
    result=audit(root); write_reports(result,a.json,a.csv); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
