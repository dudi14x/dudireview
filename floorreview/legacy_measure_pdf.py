"""Vector geometry measurement, not a trained architectural classifier.
Run: py measure_pdf.py drawing.pdf --out results
"""
import argparse, csv, hashlib, json, math, sqlite3, statistics
from pathlib import Path
import fitz

def distance(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
def midpoint(a,b): return [(a[0]+b[0])/2,(a[1]+b[1])/2]
def bezier_length(points,tolerance=.001,depth=0):
    """Adaptive de Casteljau; chord/control-polygon bounds bound length error."""
    a,b,c,d=points; lower=distance(a,d); upper=distance(a,b)+distance(b,c)+distance(c,d)
    if upper-lower<=2*tolerance or depth>=24:return (lower+upper)/2,(upper-lower)/2
    ab,bc,cd=midpoint(a,b),midpoint(b,c),midpoint(c,d)
    abc,bcd=midpoint(ab,bc),midpoint(bc,cd); m=midpoint(abc,bcd)
    l,e=bezier_length([a,ab,abc,m],tolerance/2,depth+1)
    r,f=bezier_length([m,bcd,cd,d],tolerance/2,depth+1)
    return l+r,e+f

def primitives(drawings):
    rows=[]
    for path_id,d in enumerate(drawings):
        pieces=[]
        for item in d['items']:
            kind=item[0]
            if kind=='l':pieces.append(('line',[list(item[1]),list(item[2])]))
            elif kind=='c':pieces.append(('cubic_bezier',[list(p) for p in item[1:5]]))
            elif kind=='re':
                r=item[1];pts=[list(r.tl),list(r.tr),list(r.br),list(r.bl)]
                pieces.extend(('line',[pts[i],pts[(i+1)%4]]) for i in range(4))
            elif kind=='qu':
                q=item[1];pts=[list(q.ul),list(q.ur),list(q.lr),list(q.ll)]
                pieces.extend(('line',[pts[i],pts[(i+1)%4]]) for i in range(4))
        # Keep disconnected subpaths separate; do not add arbitrary bridges.
        chain=0;previous=None;start=None
        for kind,pts in pieces:
            if previous is None or distance(previous,pts[0])>.001:
                chain+=1;start=pts[0]
            length,error=bezier_length(pts) if kind=='cubic_bezier' else (distance(*pts),0.)
            rows.append(dict(path_id=path_id,chain_id=chain,kind=kind,points=pts,
                length_pt=length,length_error_bound_pt=error,stroke=d.get('color'),
                fill=d.get('fill'),stroke_width_pt=d.get('width'),implicit_close=False))
            previous=pts[-1]
        if d.get('closePath') and previous is not None and distance(previous,start)>.001:
            rows.append(dict(path_id=path_id,chain_id=chain,kind='line',points=[previous,start],
                length_pt=distance(previous,start),length_error_bound_pt=0.,stroke=d.get('color'),
                fill=d.get('fill'),stroke_width_pt=d.get('width'),implicit_close=True))
    return rows

def scale_candidate(page,rows):
    """Conservative horizontal metric bar recognizer. No candidate means no metres.
    Requires >=3 increasing numeric labels, nearby vertical ticks, metre unit,
    a horizontal bar, and agreement of interval scales within 2 percent.
    Does not prove the entire page uses one scale.
    """
    words=page.get_text('words'); numbers=[]
    for w in words:
        try:v=float(w[4])
        except ValueError:continue
        if v>=0 and math.isfinite(v):numbers.append((w,v))
    units=[w for w in words if w[4].lower() in ('m','metres','meters')]
    ticks=[r['points'] for r in rows if r['kind']=='line' and abs(r['points'][0][0]-r['points'][1][0])<.1 and 2<r['length_pt']<18]
    candidates=[]
    for zero,v in numbers:
        if v!=0:continue
        nearby=sorted([(w,n) for w,n in numbers if abs(w[1]-zero[1])<3 and w[0]>=zero[0]-1 and w[0]-zero[0]<250],key=lambda z:z[0][0])
        matched=[]
        for w,n in nearby:
            cx=(w[0]+w[2])/2
            possible=[t for t in ticks if abs(t[0][0]-cx)<4 and abs(min(t[0][1],t[1][1])-w[3])<10]
            if possible:
                t=min(possible,key=lambda t:abs(t[0][0]-cx));matched.append((n,t[0][0],min(t[0][1],t[1][1])))
        if len(matched)<3 or matched[0][0]!=0:continue
        if any(b[0]<=a[0] or b[1]<=a[1] for a,b in zip(matched,matched[1:])):continue
        x0,x1=matched[0][1],matched[-1][1];y=statistics.mean(t[2] for t in matched)
        if not any(x0-20<u[0]<x1+30 and abs(u[1]-y)<25 for u in units):continue
        if not any(r['kind']=='line' and abs(r['points'][0][1]-r['points'][1][1])<.1 and abs(r['points'][0][1]-y)<12 and min(p[0] for p in r['points'])>=x0-2 and max(p[0] for p in r['points'])<=x1+2 and r['length_pt']>10 for r in rows):continue
        ratios=[(b[0]-a[0])/(b[1]-a[1]) for a,b in zip(matched,matched[1:])]
        ratio=statistics.median(ratios)
        if max(abs(x/ratio-1) for x in ratios)>.02:continue
        candidates.append({'metres_per_point':ratio,'ticks':matched,'status':'scale_bar_candidate_unverified','scope':'page-wide application is an assumption; check viewports'})
    unique={round(c['metres_per_point'],7):c for c in candidates}
    return next(iter(unique.values())) if len(unique)==1 else {'metres_per_point':None,'status':'ambiguous_or_missing_scale','candidates':list(unique.values())}

def analyse(pdf,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);doc=fitz.open(pdf);all_rows=[];pages=[];chains=[]
    for page_no,page in enumerate(doc,1):

        if page.rotation:
            raise ValueError('Rotated PDF pages require coordinate normalization before measurement; export an unrotated copy.')
        rows=primitives(page.get_drawings());scale=scale_candidate(page,rows) if rows else {'metres_per_point':None,'status':'no_vector_geometry'}
        seen={};groups={}
        for r in rows:
            key=(r['kind'],tuple(tuple(round(x,4) for x in p) for p in r['points']))
            reverse=(key[0],tuple(reversed(key[1])));duplicate=seen.get(key,seen.get(reverse))
            r.update(id=len(all_rows)+1,page=page_no,length_m_estimate=r['length_pt']*scale['metres_per_point'] if scale.get('metres_per_point') else None,duplicate_of=duplicate)
            if duplicate is None:seen[key]=r['id']
            groups.setdefault((r['path_id'],r['chain_id']),[]).append(r);all_rows.append(r)
        for (pid,cid),items in groups.items():
            chains.append({'page':page_no,'path_id':pid,'chain_id':cid,'segment_ids':[r['id'] for r in items],
              'segment_count':len(items),'length_pt':sum(r['length_pt'] for r in items),
              'length_m_estimate':sum(r['length_m_estimate'] for r in items) if scale.get('metres_per_point') else None,
              'closed':distance(items[0]['points'][0],items[-1]['points'][-1])<.001,
              'classification':'connected drawing path, not a room or wall'})
        pix=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False);pix.save(out/f'page_{page_no}.png')
        pages.append({'page':page_no,'width_pt':page.rect.width,'height_pt':page.rect.height,'scale':scale,'segments':len(rows)})
    report={'source_sha256':hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),'pages':pages,
      'segments':all_rows,'chains':chains,'warnings':['Vector drawing geometry includes furniture, grids, symbols, legends and scale bars, not just architecture.','No room or occupancy detector is included.','Metric lengths use an unverified scale-bar candidate; multiple view scales require regional calibration.','Raster-only PDFs require a separate detection engine.','Near-duplicate and partially overlapping segments are not deduplicated.','PDF curves already flattened into lines cannot recover their original analytic curve.']}
    (out/'geometry.json').write_text(json.dumps(report),encoding='utf-8')
    columns=['id','page','path_id','chain_id','kind','length_pt','length_error_bound_pt','length_m_estimate','duplicate_of']
    with (out/'segments.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    db=sqlite3.connect(out/'geometry.sqlite');db.execute('DROP TABLE IF EXISTS segments')
    db.execute('CREATE TABLE segments(id INTEGER PRIMARY KEY,page INTEGER,path_id INTEGER,chain_id INTEGER,kind TEXT,length_pt REAL,length_error_bound_pt REAL,length_m_estimate REAL,duplicate_of INTEGER,points_json TEXT)')
    db.executemany('INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?,?)',[[r[c] for c in columns]+[json.dumps(r['points'])] for r in all_rows]);db.commit();db.close()
    summary={'pages':pages,'segment_count':len(all_rows),'connected_paths':len(chains),'duplicate_segments':sum(r['duplicate_of'] is not None for r in all_rows),'trained_model':False,'architecture_accuracy_verified':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8');print(json.dumps(summary,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pdf');p.add_argument('--out',default='results');a=p.parse_args();analyse(a.pdf,a.out)
