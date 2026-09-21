#!/usr/bin/env python3
"""Evaluate YOLO and save green-GT/red-prediction overlay samples."""
import argparse, json, random
from datetime import datetime
from pathlib import Path
import cv2, yaml
from experiment_paths import timestamped_experiment_name
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def cli():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--split',default='test'); p.add_argument('--experiment',default='yolo_test'); p.add_argument('--samples',type=int,default=100,help='Overlays to save; 0 saves all'); p.add_argument('--seed',type=int,default=0); p.add_argument('--conf',type=float,default=.25); p.add_argument('--iou',type=float,default=.7); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--batch',type=int,default=16); p.add_argument('--device',default='0'); p.add_argument('--workers',type=int,default=8)
 return p.parse_args()
def dataset(data,split):
 c=yaml.safe_load(data.read_text()); root=Path(c.get('path',data.parent)); root=root if root.is_absolute() else data.parent/root; images=Path(c[split]); images=images if images.is_absolute() else root/images; images=images.resolve(); files=sorted(x for x in images.rglob('*') if x.suffix.lower() in EXT)
 if not files: raise SystemExit(f'No images found in {images}')
 return images,files
def label(image,root):
 rel=image.relative_to(root)
 if root.parent.name=='images': labels=root.parent.parent/'labels'/root.name
 elif root.name=='images': labels=root.parent/'labels'
 else: raise SystemExit(f'Cannot infer labels directory from {root}')
 return (labels/rel).with_suffix('.txt')
def draw(im,coords,color,text):
 x1,y1,x2,y2=map(round,coords); t=max(2,round(min(im.shape[:2])/300)); cv2.rectangle(im,(x1,y1),(x2,y2),color,t); s=max(.45,min(im.shape[:2])/1000); (w,h),b=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,s,t); top=max(0,y1-h-b-4); cv2.rectangle(im,(x1,top),(x1+w+4,y1),color,-1); cv2.putText(im,text,(x1+2,max(h+1,y1-b-2)),cv2.FONT_HERSHEY_SIMPLEX,s,(0,0,0),t,cv2.LINE_AA)
def draw_gt(im,path):
 if not path.is_file(): return 0
 h,w=im.shape[:2]; n=0
 for line in path.read_text().splitlines():
  v=line.split()
  if len(v)<5: continue
  cls=int(float(v[0])); x,y,bw,bh=map(float,v[1:5]); draw(im,((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),(0,255,0),f'GT {cls}'); n+=1
 return n
def scalar(v):
 try:return v.item()
 except AttributeError:return v if isinstance(v,(int,float,str)) else float(v)
def main():
 a=cli()
 if not a.model.is_file() or not a.data.is_file(): raise SystemExit('Model or dataset YAML not found')
 if a.samples<0: raise SystemExit('--samples must be >= 0')
 from ultralytics import YOLO
 data=a.data.resolve(); root,files=dataset(data,a.split); chosen=files if not a.samples or a.samples>=len(files) else sorted(random.Random(a.seed).sample(files,a.samples)); out=Path('artifacts').resolve()/timestamped_experiment_name(a.experiment,datetime.now()); overlays=out/'overlays'; overlays.mkdir(parents=True); model=YOLO(str(a.model)); metrics=model.val(data=str(data),split=a.split,imgsz=a.imgsz,batch=a.batch,device=a.device,workers=a.workers,iou=a.iou,plots=True,save_json=True,project=str(out.parent),name=out.name,exist_ok=True); manifest=[]
 for start in range(0,len(chosen),a.batch):
  batch_paths=chosen[start:start+a.batch]
  results=model.predict(source=[str(x) for x in batch_paths],stream=True,conf=a.conf,iou=a.iou,imgsz=a.imgsz,device=a.device,verbose=False)
  for offset,(path,result) in enumerate(zip(batch_paths,results),1):
   i=start+offset; im=cv2.imread(str(path)); gt=draw_gt(im,label(path,root)); pred=0
   for detection in ([] if result.boxes is None else result.boxes):
    cls,conf=int(detection.cls[0]),float(detection.conf[0]); draw(im,detection.xyxy[0].tolist(),(0,0,255),f'PRED {cls} {conf:.2f}'); pred+=1
   rel=path.relative_to(root); dst=(overlays/rel).with_suffix('.jpg'); dst.parent.mkdir(parents=True,exist_ok=True)
   if not cv2.imwrite(str(dst),im): raise SystemExit(f'Could not save {dst}')
   manifest.append({'image':str(rel),'overlay':str(dst.relative_to(out)),'ground_truth_boxes':gt,'prediction_boxes':pred})
   if i%25==0 or i==len(chosen): print(f'Saved overlays: {i}/{len(chosen)}',flush=True)
 summary={'model':str(a.model.resolve()),'data':str(data),'split':a.split,'evaluated_images':len(files),'saved_overlays':len(manifest),'overlay_confidence':a.conf,'colors':{'ground_truth':'green','predictions':'red'},'metrics':{k:scalar(v) for k,v in metrics.results_dict.items()},'samples':manifest}; (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(f'Complete. Metrics and overlays are in {out}')
if __name__=='__main__':main()
