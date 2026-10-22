"""Derive physical scale from PDF geometry and independently check its references.

No area schedule, assumed room size, or neural guess supplies a physical length.
All lengths are extracted in unrotated PDF points; pixel conversion uses page.rect.
"""
import math
import re
import numpy as np


def line_segments(page):
    rows=[]
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0]=='l':rows.append([item[1].x,item[1].y,item[2].x,item[2].y])
    return np.asarray(rows,dtype=float).reshape(-1,4)


def declared_ratios(page):
    return sorted({int(v) for v in re.findall(r'(?<!\d)1\s*:\s*(\d{2,5})(?!\d)',page.get_text()) if 10<=int(v)<=10000})


def scale_bars(page,segments):
    words=page.get_text('words');found=[]
    units=[w for w in words if w[4].lower() in ('m','metres','meters','mm','cm','ft')]
    if not len(segments):return []
    vertical=segments[(np.abs(segments[:,0]-segments[:,2])<.05)&(np.abs(segments[:,1]-segments[:,3])>=3)&(np.abs(segments[:,1]-segments[:,3])<=18)]
    for zero in [w for w in words if w[4] in ('0','0.0')]:
        cy=(zero[1]+zero[3])/2;labels=[]
        for w in words:
            if not re.fullmatch(r'\d+(?:\.\d+)?',w[4]):continue
            if abs((w[1]+w[3])/2-cy)>2 or w[0]<zero[0]-2:continue
            cx=(w[0]+w[2])/2
            ticks=[s for s in vertical if abs(s[0]-cx)<2.5 and w[3]-3<=min(s[1],s[3])<=w[3]+15]
            if ticks:
                tick=min(ticks,key=lambda s:abs(s[0]-cx))
                labels.append((float(w[4]),float(tick[0]),float(min(tick[1],tick[3]))))
        labels=sorted(set(labels),key=lambda p:p[1])
        # Keep a single compact scale-bar row, not a full page of numeric grids.
        labels=[p for p in labels if p[1]-labels[0][1]<=250 and abs(p[2]-labels[0][2])<1] if labels else []
        if len(labels)<3 or len(labels)>10 or labels[0][0]!=0 or any(b[0]<=a[0] for a,b in zip(labels,labels[1:])):continue
        nearby=[w for w in units if abs((w[0]+w[2])/2-labels[-1][1])<35 and abs((w[1]+w[3])/2-cy)<30]
        if not nearby:continue
        unit=min(nearby,key=lambda w:abs((w[0]+w[2])/2-labels[-1][1])+abs((w[1]+w[3])/2-cy))[4].lower()
        factor={'m':1,'metres':1,'meters':1,'mm':.001,'cm':.01,'ft':.3048}[unit]
        x=np.asarray([p[1] for p in labels]);v=np.asarray([p[0]*factor for p in labels])
        slope=float(np.dot(x-x.mean(),v-v.mean())/np.dot(x-x.mean(),x-x.mean()))
        residual=float(np.max(np.abs((x-x.mean())*slope+v.mean()-v)))
        if slope<=0 or residual>max(.002,(v.max()-v.min())*.005):continue
        found.append({'metres_per_point':slope,'method':'graphic_scale_bar','unit':unit,'references':[{'label':p[0],'tick_pdf':[p[1],p[2]]} for p in labels],'max_residual_metres':residual})
    return found


def dimension_references(page,segments):
    if not len(segments):return []
    horizontal=np.abs(segments[:,1]-segments[:,3])<.05
    vertical=np.abs(segments[:,0]-segments[:,2])<.05
    lengths=np.hypot(segments[:,2]-segments[:,0],segments[:,3]-segments[:,1])
    found=[];seen=set()
    for w in page.get_text('words'):
        text=w[4].replace(',','')
        match=re.fullmatch(r'(\d+(?:\.\d+)?)(mm|cm|m|ft)?',text,re.I)
        if not match:continue
        number=float(match[1]);unit=match[2].lower() if match[2] else None
        if not unit and not 1000<=number<=500000:continue
        if number<=0:continue
        cx,cy=(w[0]+w[2])/2,(w[1]+w[3])/2
        orientation='vertical' if w[3]-w[1]>(w[2]-w[0])*1.2 else 'horizontal'
        if orientation=='horizontal':
            orthogonal=np.abs(segments[:,1]-cy);mid=np.abs((segments[:,0]+segments[:,2])/2-cx)
            mask=horizontal
        else:
            orthogonal=np.abs(segments[:,0]-cx);mid=np.abs((segments[:,1]+segments[:,3])/2-cy)
            mask=vertical
        mask=mask&(lengths>12)&(orthogonal<=12)&(mid<=np.maximum(4,lengths*.1))
        indices=np.flatnonzero(mask)
        if not len(indices):continue
        index=int(min(indices,key=lambda i:orthogonal[i]+mid[i]*.5))
        signature=(number,unit,orientation,tuple(round(v,1) for v in segments[index]))
        if signature in seen:continue
        seen.add(signature)
        found.append({'value':number,'unit':unit,'length_points':float(lengths[index]),'axis':orientation,'text_box':[float(v) for v in w[:4]],'endpoints':segments[index].tolist()})
    return found


def auto_scale(page,width):
    segments=line_segments(page);ratios=declared_ratios(page)
    bars=scale_bars(page,segments);dimensions=dimension_references(page,segments)
    factors={'mm':.001,'cm':.01,'m':1,'ft':.3048}
    text=page.get_text().upper()
    unit_note='mm' if re.search(r'(?:ALL\s+)?DIMENSIONS.{0,50}(?:MILLIMET(?:ER|RE)|\bMM\b)',text,re.S) else None
    evidence=[];candidates=[]
    for bar in bars:candidates.append(bar['metres_per_point']);evidence.append(bar)
    if len(ratios)>1:
        return {'status':'ambiguous','metres_per_pixel':None,'source':'multiple_pdf_scales','automatic':False,'evidence':evidence,'message':'Multiple declared scales/viewports were found. A single page-wide metric scale cannot be established safely.'}
    ratio_scale=ratios[0]*.0254/72 if len(ratios)==1 else None
    if ratio_scale:evidence.append({'method':'declared_ratio','denominator':ratios[0],'metres_per_point':ratio_scale})
    dimensional=[]
    for ref in dimensions:
        if ref['length_points']<70:continue
        unit=ref['unit'] or unit_note
        if unit:
            value=ref['value']*factors[unit]/ref['length_points']
        elif ratio_scale:
            options=[(abs(ref['value']*factor/ref['length_points']/ratio_scale-1),name,factor) for name,factor in factors.items()]
            error,unit,factor=min(options)
            if error>.015 or ref['length_points']<70:continue
            value=ref['value']*factor/ref['length_points']
        else:continue
        dimensional.append((value,{**ref,'unit':unit}))
    # Avoid 'confirming' a ratio by cherry-picking isolated accidental matches.
    if dimensional:
        clusters=[]
        for value,ref in dimensional:
            group=next((g for g in clusters if abs(value/np.median([r[0] for r in g])-1)<.03),None)
            if group is None:clusters.append([(value,ref)])
            else:group.append((value,ref))
        best=max(clusters,key=len);different_values={r[1]['value'] for r in best}
        if len(best)>=3 and len(different_values)>=2:
            median=float(np.median([r[0] for r in best]))
            competing=[g for g in clusters if g is not best and len(g)>=max(3,len(best)*.35) and len({r[1]['value'] for r in g})>=2]
            if competing:
                return {'status':'ambiguous','metres_per_pixel':None,'source':'conflicting_dimensions','automatic':False,'evidence':evidence,'message':'Dimension annotations imply different scales. The page may contain multiple viewports or distorted geometry.'}
            candidates.append(median)
            evidence.append({'method':'dimension_lines','metres_per_point':median,'references':[r[1] for r in best[:12]],'matched_count':len(best),'unit_source':'explicit_annotation_or_note' if unit_note or any(ref.get('unit') for ref in dimensions) else 'cross_checked_with_declared_ratio'})
    if candidates:
        if max(candidates)/min(candidates)>1.03:
            return {'status':'ambiguous','metres_per_pixel':None,'source':'conflicting_scale_references','automatic':False,'evidence':evidence,'message':'The scale bar and dimension lines disagree. No automatic measurements were accepted.'}
        points_scale=float(np.median(candidates))
        return {'status':'resolved','metres_per_pixel':points_scale*page.rect.width/width,'source':'pdf_auto_references','automatic':True,'evidence':evidence,'message':'Scale derived automatically from geometric references in the drawing.'}
    if ratio_scale:
        return {'status':'declared_only','metres_per_pixel':ratio_scale*page.rect.width/width,'source':'printed_ratio_unverified','automatic':False,'evidence':[{'method':'declared_ratio','denominator':ratios[0]}],'message':'A printed scale was found, but independent geometric references could not confirm it. Measurements can only be estimated.'}
    return {'status':'unavailable','metres_per_pixel':None,'source':'missing','automatic':False,'evidence':[],'message':'No reliable scale bar, matched dimensions or declared scale was found. Absolute distances cannot be recovered from pixels alone.'}
