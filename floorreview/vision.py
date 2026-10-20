"""CPU pixel segmentation + enclosed-region proposals. No text-based room inference."""
from pathlib import Path
import cv2
import joblib
import numpy as np

CLASSES = ['background', 'wall', 'room', 'stairs', 'corridor', 'mep']

def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return np.stack([gray, cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(gray, (15, 15), 0), np.abs(gx), np.abs(gy)], axis=-1)

def proposals(image, model_path=None):
    if model_path and Path(model_path).exists() and str(model_path).endswith('.pt'):
        if 'archvision' in Path(model_path).name:
            from .archvision import detect
            return detect(image, model_path)
        from .neural import infer
        scores, source = infer(image, model_path)
        return regions_from_semantic(scores), source
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    source = 'geometry'
    scores = None
    if model_path and Path(model_path).exists():
        # Only server-controlled, locally trained artifacts are loaded. Never deserialize uploads.
        artifact = joblib.load(model_path)
        x = features(image).reshape(-1, 5)
        prediction = np.concatenate([artifact['model'].predict(x[i:i+100000]) for i in range(0, len(x), 100000)]).reshape(gray.shape)
        source = artifact['source']
        scores = prediction
    # Dark drawing strokes form barriers; semantic wall predictions can add barriers.
    barriers = (gray < 145).astype(np.uint8)
    if scores is not None:
        barriers |= (scores == 1).astype(np.uint8)
    barriers = cv2.morphologyEx(barriers, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-barriers, 4)
    regions = []
    h, w = gray.shape
    for label in range(1, count):
        x, y, bw, bh, size = stats[label]
        if size < max(100, h*w*0.0003) or x == 0 or y == 0 or x+bw == w or y+bh == h:
            continue
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        points = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2).tolist()
        if len(points) < 3:
            continue
        points = clean_outline(points)
        if points is None:
            continue
        kind, occupancy = 'room', 'unclassified'
        if scores is not None:
            votes = np.bincount(scores[labels == label], minlength=6)
            semantic = int(votes[2:].argmax()+2)
            if votes[semantic] > size * 0.6:
                kind = {3: 'stairs', 4: 'corridor'}.get(semantic, 'room')
                occupancy = 'mep' if semantic == 5 else 'unclassified'
        regions.append({'id': len(regions)+1, 'name': f'Space {len(regions)+1}', 'kind': kind, 'occupancy': occupancy, 'unit': '', 'polygon': points})
    return regions, source


def footprint_proposal(image):
    """Propose the largest connected drawing outline; explicit review is required."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ink = (gray < 90).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h,w=gray.shape
    candidates=[]
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour)
        area=cv2.contourArea(contour)
        if area > w*h*.025 and not ((bw>w*.65 and bh>h*.85) or (bw>w*.85 and bh>h*.65)) and x>2 and y>2 and x+bw<w-2 and y+bh<h-2:
            candidates.append(contour)
    if not candidates:return None
    contour=max(candidates,key=cv2.contourArea)
    from shapely.geometry import Polygon
    shape=Polygon(cv2.approxPolyDP(contour,2,True).reshape(-1,2))
    if not shape.is_valid:shape=shape.buffer(0)
    if shape.geom_type=='MultiPolygon':shape=max(shape.geoms,key=lambda p:p.area)
    if shape.is_empty or shape.geom_type!='Polygon':return None
    return [[float(x),float(y)] for x,y in shape.exterior.coords[:-1]]


def regions_from_semantic(scores):
    regions=[]
    for c in (2,3,4,5):
        mask=(scores==c).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,4)
        for ident in range(1,count):
            if stats[ident,cv2.CC_STAT_AREA]<100:continue
            contours,_=cv2.findContours((labels==ident).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=max(contours,key=cv2.contourArea)
            points=cv2.approxPolyDP(contour,1.2,True).reshape(-1,2).tolist()
            if len(points)<3:continue
            points=clean_outline(points)
            if points is None:continue
            i=len(regions)+1
            regions.append({'id':i,'name':f'Space {i}','kind':{3:'stairs',4:'corridor'}.get(c,'room'),'occupancy':'mep' if c==5 else 'unclassified','unit':'','polygon':points})
    return regions


def clean_outline(points):
    from shapely.geometry import Polygon
    shape=Polygon(points)
    if not shape.is_valid:shape=shape.buffer(0)
    if shape.geom_type=='MultiPolygon':shape=max(shape.geoms,key=lambda p:p.area)
    if shape.is_empty or shape.geom_type!='Polygon' or shape.area<1:return None
    shape=shape.simplify(1,preserve_topology=True)
    points=[[float(x),float(y)] for x,y in shape.exterior.coords[:-1]]
    if len(points)>5000 or not Polygon(points).is_valid:return None
    return points


def room_envelope(regions):
    """Provisional envelope of the largest room cluster; requires exterior review."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import concave_hull
    shapes=[Polygon(r['polygon']) for r in regions]
    shapes=[s for s in shapes if s.is_valid and s.area>100]
    if len(shapes)<3:return None
    gap=float(np.median([s.area**.5 for s in shapes]))*.8
    clusters=unary_union([s.buffer(gap) for s in shapes])
    if clusters.geom_type=='MultiPolygon':clusters=max(clusters.geoms,key=lambda s:s.area)
    members=[s for s in shapes if clusters.covers(s.representative_point())]
    envelope=concave_hull(unary_union(members),ratio=.2,allow_holes=False)
    if envelope.geom_type!='Polygon':return None
    return clean_outline(list(envelope.exterior.coords)[:-1])
