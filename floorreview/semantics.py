"""Optional PDF evidence supplements geometric proposals; labels never supply measured areas."""
import re
import fitz
from shapely.geometry import Point
from .geometry import polygon

UNIT = re.compile(r'^(\d)\s*(?:BR|BED|BD)(?:[-\s]+([A-Z]))?$', re.I)

def pdf_evidence(page, width, height, regions):
    sx, sy = width/page.rect.width, height/page.rect.height
    words = page.get_text('words')
    # Text coordinates are unrotated; the raster follows the PDF page rotation.
    def image_point(x,y):
        point=fitz.Point(x,y)*page.rotation_matrix
        return [point.x*sx,point.y*sy]
    # Labels such as '2 BED' can be separate PDF words on one line.
    lines_by_id={}
    for word in words:lines_by_id.setdefault(tuple(word[5:7]),[]).append(word)
    combined=[]
    for line_words in lines_by_id.values():
        ordered=sorted(line_words,key=lambda w:w[0])
        text=' '.join(w[4] for w in ordered)
        if len(ordered)>1 and UNIT.fullmatch(text):
            combined.append((min(w[0] for w in ordered),min(w[1] for w in ordered),max(w[2] for w in ordered),max(w[3] for w in ordered),text))
    words=words+combined
    unit_candidates = []
    has_br_markers = any(re.fullmatch(r'\d\s*BR(?:[-\s]+[A-Z])?', w[4], re.I) for w in words)
    # A marker is evidence of a unit, not proof of its boundary. Legend rows are excluded
    # by identifying the legend block and all markers aligned beneath its heading.
    legends = [w for w in words if w[4].upper() == 'LEGEND']
    for word in words:
        x0,y0,x1,y1,text,*_ = word
        match = UNIT.fullmatch(text)
        if not match: continue
        if has_br_markers and not re.search(r'BR',text,re.I):continue
        if any(x0 >= l[0]-25 and y0 >= l[1] for l in legends): continue
        unit_candidates.append({'id':f'U{len(unit_candidates)+1:03}', 'bedrooms':int(match[1]), 'label':text, 'position':image_point((x0+x1)/2,(y0+y1)/2), 'source':'pdf_marker', 'reviewed':False})
    # Match full text lines to contained spaces. No nearest-neighbour unit assignment:
    # adjacency alone cannot establish apartment ownership.
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            text = ' '.join(s['text'] for s in line['spans']).strip()
            upper = text.upper()
            kind = None; occupancy = None
            if re.search(r'\b(STAIR|STAIRS)\b',upper): kind='stairs'
            elif re.search(r'\b(CORRIDOR|PASSAGE|LIFT LOBBY)\b',upper): kind='corridor'
            elif re.search(r'\b(ELECTRIC|ELECTRICAL|ELEC|MEP|WATER METER|PUMP|TRANSFORMER|PLANT|AHU|MECHANICAL|IDF|MDF|LV ROOM|HV ROOM|SWITCHGEAR|FCU)\b',upper): kind='room';occupancy='mep'
            elif re.search(r'\b(BEDROOM|KITCHEN|LIVING ROOM|BATHROOM|FOYER|STUDY)\b',upper): kind='room';occupancy='residential'
            if not kind:continue
            box=line['bbox']; point=Point(image_point((box[0]+box[2])/2,(box[1]+box[3])/2))
            matches=[r for r in regions if polygon(r['polygon']).covers(point)]
            if not matches:continue
            region=min(matches,key=lambda r:polygon(r['polygon']).area)
            region.update(name=text,kind=kind,occupancy=occupancy or 'common',semantic_source='contained_pdf_label')
    return unit_candidates


def scale_from_pdf(page,width):
    text=page.get_text()
    ratios=re.findall(r'(?<!\d)1\s*:\s*(\d{2,5})(?!\d)',text)
    ratios={int(v) for v in ratios if 10<=int(v)<=10000}
    if len(ratios)==1:
        # PDF user-space points are 1/72 inch. Assumes printed scale matches page size.
        return next(iter(ratios)) * .0254/72 * page.rect.width/width
    return None
