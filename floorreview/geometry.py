"""Areas are computed from reviewed polygons, never OCR text or a bounding box."""
import math
from shapely.geometry import Polygon
from shapely.ops import unary_union

KINDS = {'room', 'stairs', 'corridor', 'wall', 'void'}
OCCUPANCIES = {'unclassified', 'residential', 'mep', 'commercial', 'common'}

def polygon(points):
    if not isinstance(points, list) or not 3 <= len(points) <= 5000:
        raise ValueError('A polygon needs 3–5000 vertices.')
    if any(len(p) != 2 or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in p) for p in points):
        raise ValueError('Coordinates must be finite numbers.')
    shape = Polygon(points)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError('Polygon must be simple and have positive area.')
    return shape

def calculate(result):
    scale = result.get('metres_per_pixel')
    if scale is not None and (not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Scale must be a positive metres-per-pixel value.')
    footprint = polygon(result['footprint']) if result.get('footprint') else None
    shapes = []
    identifiers = set()
    for entity in result['regions']:
        if entity['id'] in identifiers:
            raise ValueError('Duplicate region ID.')
        identifiers.add(entity['id'])
        if entity['kind'] not in KINDS or entity['occupancy'] not in OCCUPANCIES:
            raise ValueError('Unknown region kind or occupancy.')
        shape = polygon(entity['polygon'])
        if footprint is not None:
            shape = shape.intersection(footprint)
        shapes.append((entity, shape))
    exclusions = unary_union([s for r, s in shapes if r['kind'] != 'room'])
    rooms = [(r, s.difference(exclusions)) for r, s in shapes if r['kind'] == 'room']
    def area(shape):
        return round(shape.area * scale * scale, 4) if scale else None
    total = unary_union([s for _, s in rooms])
    overlap = sum(s.area for _, s in rooms) - total.area
    warnings = []
    if scale is None:
        warnings.append('Set a measured reference distance to obtain square metres.')
    if result.get('scale_source')=='printed_ratio_unverified':
        warnings.append('Areas are estimates from the printed PDF ratio. Verify against a known dimension; resized sheets and multiple viewports can invalidate this scale.')
    if result.get('footprint_source') in ('automatic_outline_proposal','room_envelope_proposal') and not result.get('reviewed'):
        warnings.append('Gross area uses a proposed outline, which may omit exterior walls, balconies or disconnected wings. Confirm the entire exterior boundary before using the area.')
    if footprint is None:
        warnings.append('Trace the outer floor boundary to calculate gross floor area.')
    if overlap > 0.01:
        warnings.append('Room polygons overlap. Individual room and group totals may overlap; net total uses their union.')
    if result.get('model_source') == 'synthetic':
        warnings.append('Synthetic baseline: predictions are unvalidated on real drawings.')
    if not any(r['occupancy']=='mep' for r, _ in rooms):
        warnings.append('No MEP space has been identified. This does not establish that the drawing contains none.')
    if result.get('units'):
        warnings.append('Unit counts are marker candidates until reviewed; legends, duplicates and missing markers can affect counts.')
    if not result.get('reviewed'):
        warnings.append('Automatic suggestions require boundary and classification review.')
    groups = {}
    for field in ('occupancy', 'unit'):
        keys = sorted({str(r.get(field) or 'unassigned') for r, _ in rooms})
        groups[field] = {key: area(unary_union([s for r, s in rooms if str(r.get(field) or 'unassigned') == key])) for key in keys}
    return {
        'units': 'm²' if scale else 'uncalibrated',
        'unit_counts': {f'{n} bed': sum(u['bedrooms']==n for u in result.get('units', [])) for n in sorted({u['bedrooms'] for u in result.get('units', [])})},
        'unit_count_source': 'unverified_pdf_markers_or_user_review',
        'mep_space_count':sum(r['occupancy']=='mep' for r, _ in rooms),
        'gross_floor_area_m2': area(footprint) if footprint is not None else None,
        'net_room_area_m2': area(total),
        'excluded_area_m2': area(exclusions),
        'unallocated_area_m2': area(footprint.difference(unary_union([total, exclusions]))) if footprint is not None else None,
        'rooms': [{'id': r['id'], 'name': r['name'], 'occupancy': r['occupancy'], 'unit': r.get('unit', ''), 'area_m2': area(s), 'area_pixels': round(s.area, 2)} for r, s in rooms],
        'groups': groups, 'warnings': warnings,
    }
