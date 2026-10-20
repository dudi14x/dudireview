"""Use an explicit PDF color legend as evidence, not a universal color assumption."""
import math
import re
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import scale as scale_shape


def filled_polygons(drawing):
    """Flatten closed subpaths and retain nested holes in the support geometry."""
    rings=[];points=[]
    def flush():
        nonlocal points
        if len(points)>=3:
            shape=Polygon(points)
            if shape.is_valid and shape.area>1:rings.append(shape)
        points=[]
    for item in drawing['items']:
        if item[0] in ('re','qu'):
            flush()
            q=item[1]
            ring=[tuple(q.tl),tuple(q.tr),tuple(q.br),tuple(q.bl)] if item[0]=='re' else [tuple(q.ul),tuple(q.ur),tuple(q.lr),tuple(q.ll)]
            points=ring;flush();continue
        if item[0]=='l':segment=[tuple(item[1]),tuple(item[2])]
        elif item[0]=='c':
            p=np.array([tuple(x) for x in item[1:]],dtype=float)
            segment=[tuple((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]) for t in np.linspace(0,1,17)]
        else:
            flush();continue
        if points and math.dist(points[-1],segment[0])>.05:flush()
        if not points:points.append(segment[0])
        points.extend(segment[1:])
        if len(points)>3 and math.dist(points[-1],points[0])<.01:flush()
    flush()
    depths=[sum(other is not ring and other.contains(ring) for other in rings) for ring in rings]
    result=[]
    for i,ring in enumerate(rings):
        if depths[i]%2:continue
        holes=[other for j,other in enumerate(rings) if depths[j]==depths[i]+1 and ring.contains(other)]
        shape=ring.difference(unary_union(holes)) if holes else ring
        if shape.geom_type=='Polygon':result.append(shape)
        elif shape.geom_type=='MultiPolygon':result.extend(shape.geoms)
    return result


def category(text):
    text=text.upper().strip()
    if re.search(r'\b(MEP|ELECTRIC(?:AL)?|MECHANICAL|PLANT|PUMP|AHU)\b',text):return 'room','mep'
    if re.search(r'\b(CIRCULATION|CORRIDOR|PASSAGE|LOBBY)\b',text):return 'corridor','common'
    if re.search(r'\b(STAIR|STAIRS|VERTICAL TRANSPORT|LIFT)\b',text):return 'stairs','common'
    if re.search(r'\b(TERRACE|BALCONY|GARDEN|LANDSCAPE|PARKING|OUTDOOR)\b',text):return 'void','common'
    if re.search(r'\b(\d\s*(?:BR|BD|BED)|APARTMENT|RESIDENTIAL|GUESTROOM)\b',text):return 'room','residential'
    return None


def legend_geometry(page,width,height):
    drawings=page.get_drawings()
    if not any('LEGEND' in w[4].upper() for w in page.get_text('words')):return [],[]
    lines=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            text=' '.join(s['text'] for s in line['spans']).strip()
            lines.append((text,line['bbox']))
    entries=[];swatches=set()
    headings=[box for text,box in lines if 'LEGEND' in text.upper()]
    # Match small filled boxes directly to a nearby text line on their right.
    for text,box in lines:
        kind=category(text)
        if not kind or not any(box[1]>heading[1] and box[0]>=heading[0]-20 for heading in headings):continue
        matches=[]
        for i,d in enumerate(drawings):
            color=d.get('fill');r=d['rect']
            if color is None or len(color)!=3 or not .05<sum(color)/3<.97:continue
            if not 3<=r.width<=40 or not 3<=r.height<=25 or len(d['items'])>5:continue
            dy=abs((r.y0+r.y1)/2-(box[1]+box[3])/2)
            if 0<=box[0]-r.x1<=12 and dy<max(4,r.height*.5):matches.append((box[0]-r.x1+dy,i,d))
        if matches:
            _,i,d=min(matches,key=lambda m:m[0]);swatches.add(i)
            entries.append((tuple(d['fill']),text,kind,d['rect']))
    if not entries:return [],[]
    sx,sy=width/page.rect.width,height/page.rect.height
    supports=[];semantic=[]
    for i,d in enumerate(drawings):
        color=d.get('fill')
        if i in swatches or color is None:continue
        matching=[e for e in entries if len(color)==3 and max(abs(a-b) for a,b in zip(color,e[0]))<.008]
        if not matching:continue
        # A shared swatch used for conflicting meanings cannot establish an occupancy.
        meanings={e[2] for e in matching}
        label=matching[0][1];kind=matching[0][2] if len(meanings)==1 else None
        minimum=max(e[3].get_area() for e in matching)*1.5
        for shape in filled_polygons(d):
            if shape.area<minimum:continue
            shape=scale_shape(shape,xfact=sx,yfact=sy,origin=(0,0))
            supports.append(shape)
            if kind and (kind[0]!='room' or kind[1]=='mep') and not shape.interiors:
                semantic.append({'name':label,'kind':kind[0],'occupancy':kind[1],'unit':'','polygon':[list(p) for p in shape.exterior.coords[:-1]],'geometry_source':'pdf_legend_vector','semantic_source':'pdf_legend_color','reviewed':False})
    return supports,semantic


def apply_legend(regions,supports,semantic):
    if not supports:return regions,0
    support=unary_union(supports);kept=[]
    for region in regions:
        shape=Polygon(region['polygon'])
        if shape.intersection(support).area/max(shape.area,1)<.15:continue
        # Exact vector service/exclusion zones supersede overlapping room guesses.
        if any(shape.intersection(Polygon(s['polygon'])).area/max(shape.area,1)>.65 for s in semantic):continue
        kept.append(region)
    removed=len(regions)-len(kept)
    for item in semantic:
        region=dict(item);region['id']=max([r['id'] for r in kept],default=0)+1;kept.append(region)
    return kept,removed
