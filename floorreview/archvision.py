"""ArchVision pretrained YOLO integration with overlapping tiles for dense sheets.

Reference: Parth-Talsania/Archvision, commit recorded in THIRD_PARTY.md.
This adapter is original; upstream trained weights retain their own provenance.
"""
import threading
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO
import torch

MODELS={}
LOCK=threading.Lock()


def detect(image,path):
    # Ultralytics predictor is mutable, so each inference sequence owns the lock.
    with LOCK:
        torch.set_num_threads(2)
        path=str(Path(path).resolve())
        if path not in MODELS:MODELS[path]=YOLO(path)
        model=MODELS[path];h,w=image.shape[:2]
        crops=[(0,0,image)]
        # Global context plus local crops, without forcing every small room into 640px.
        if max(h,w)>1000:
            crops.extend((x,y,image[y:y+800,x:x+800]) for y in range(0,h,640) for x in range(0,w,640) if min(image[y:y+800,x:x+800].shape[:2])>=200)
        candidates=[]
        for x,y,crop in crops:
            results=model.predict(source=np.ascontiguousarray(crop[:,:,::-1]),imgsz=800,conf=.3,iou=.5,device='cpu',verbose=False,retina_masks=True,max_det=1000)
            result=results[0]
            if result.masks is None:continue
            for points,score in zip(result.masks.xy,result.boxes.conf.cpu().tolist()):
                local=points.astype(float)
                ch,cw=crop.shape[:2]
                if local[:,0].min()<=2 or local[:,1].min()<=2 or local[:,0].max()>=cw-3 or local[:,1].max()>=ch-3:continue
                points=local+np.array([x,y])
                shape=Polygon(points)
                if not shape.is_valid:shape=shape.buffer(0)
                if shape.geom_type!='Polygon' or shape.area<60 or shape.area>crop.shape[0]*crop.shape[1]*.6:continue
                shape=shape.simplify(1,preserve_topology=True)
                candidates.append((score,shape))
        keep=[]
        for score,shape in sorted(candidates,key=lambda r:r[0],reverse=True):
            if any(shape.intersection(old).area/max(1,min(shape.area,old.area))>.65 for _,old in keep):continue
            keep.append((score,shape))
    return [{'id':i,'name':f'Space {i}','kind':'room','occupancy':'unclassified','unit':'','polygon':[[round(x,2),round(y,2)] for x,y in shape.exterior.coords[:-1]],'confidence':round(score,4),'geometry_source':'archvision_yolo'} for i,(score,shape) in enumerate(keep,1)],'archvision_pretrained'
