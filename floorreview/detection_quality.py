"""Conservative evidence gate for unsupported room instance predictions.

This rejects only sparse, weakly bounded predictions. It does not prove a room
is correct, and faint real rooms can be flagged. Rejected polygons are retained.
"""
import cv2
import numpy as np


def screen_predictions(image, regions):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    radius=max(3,round(max(h,w)/1000))
    kept,rejected=[],[]
    for original in regions:
        region=dict(original)
        if region.get('geometry_source')!='archvision_yolo' or region.get('reviewed') or region.get('semantic_source'):
            kept.append(region);continue
        points=np.rint(region['polygon']).astype(np.int32)
        x,y,bw,bh=cv2.boundingRect(points)
        x0,y0=max(0,x-radius-1),max(0,y-radius-1)
        x1,y1=min(w,x+bw+radius+1),min(h,y+bh+radius+1)
        crop=gray[y0:y1,x0:x1]
        if crop.size==0:
            kept.append(region);continue
        local=points-np.array([x0,y0])
        inside=np.zeros(crop.shape,np.uint8);cv2.fillPoly(inside,[local],1)
        boundary=np.zeros(crop.shape,np.uint8);cv2.polylines(boundary,[local],True,1,1)
        dark=(crop<155).astype(np.uint8)
        nearby=cv2.dilate(dark,np.ones((2*radius+1,2*radius+1),np.uint8))
        support=float(nearby[boundary>0].mean()) if boundary.any() else 0.
        core=cv2.erode(inside,np.ones((2*radius+1,2*radius+1),np.uint8))
        density=float(dark[core>0].mean()) if core.any() else 0.
        region['boundary_evidence']={'wall_support_fraction':round(support,4),'interior_ink_fraction':round(density,4),'method':'sparse_boundary_gate_v1'}
        if support<.18 and density<.012:
            region['rejection_reason']='Sparse interior and little dark-line support around the proposed boundary; possible grid, road or page graphic.'
            rejected.append(region)
        else:kept.append(region)
    return kept,rejected
