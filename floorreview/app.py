import csv
import io
import json
import math
import os
from pathlib import Path
import secrets
import threading
import uuid
import zipfile
import cv2
import fitz
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, abort, jsonify, request, send_file
from dotenv import load_dotenv
from .gemini import GeminiReview, GeminiError, fingerprint
from .geometry import calculate, polygon
from .vision import proposals, footprint_proposal
from .semantics import pdf_evidence
from .auto_scale import auto_scale
from .pdf_geometry import legend_geometry, apply_legend

ROOT = Path(__file__).resolve().parent
LOCK = threading.RLock()

def create_app(data_dir=None):
    if data_dir is None:
        load_dotenv(ROOT.parent/'.env.local',override=False,encoding='utf-8')
    app = Flask(__name__, static_folder='static')
    data = Path(data_dir or os.environ.get('FLOORREVIEW_DATA', ROOT.parent/'data'))
    data.mkdir(parents=True, exist_ok=True)
    app.config.update(MAX_CONTENT_LENGTH=32*1024*1024)
    token = secrets.token_urlsafe(32)
    gemini = GeminiReview(data)
    model = Path(os.environ.get('FLOORREVIEW_MODEL', ROOT.parent/'models/archvision-rooms.pt'))

    @app.before_request
    def protect():
        if request.method in ('POST', 'PUT', 'DELETE') and request.headers.get('X-FloorReview-Token') != token:
            abort(403)

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(GeminiError)
    def gemini_error(exc):
        return jsonify(error=str(exc)),exc.status

    @app.errorhandler(413)
    def too_large(exc):
        return jsonify(error='Upload limit is 32 MiB.'), 413

    def directory(ident):
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            abort(404)
        folder = data/ident
        if not (folder/'result.json').exists():
            abort(404)
        return folder

    def read(ident):
        result=json.loads((directory(ident)/'result.json').read_text(encoding='utf-8'))
        if 'regions' in result:result['analysis']=calculate(result)
        return result

    def save(folder, result):
        temporary = folder/'result.tmp'
        temporary.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
        temporary.replace(folder/'result.json')

    @app.get('/')
    def index():
        return (ROOT/'static/index.html').read_text(encoding='utf-8').replace('__TOKEN__', token)

    @app.get('/api/health')
    def health():
        metrics=model.with_suffix('.metrics.json')
        return jsonify(status='ok', model_available=model.exists(), model_name=model.name, metrics=json.loads(metrics.read_text(encoding='utf-8')) if metrics.exists() else None, training_runs=[{'checkpoint':p.with_suffix('').with_suffix('.pt' if 'unet' in p.name else '.joblib').name,'report':json.loads(p.read_text(encoding='utf-8'))} for p in (ROOT.parent/'models').glob('*.metrics.json')])

    @app.get('/api/projects')
    def projects():
        return jsonify([{'id': r['id'], 'name': r['name']} for r in [json.loads(p.read_text(encoding='utf-8')) for p in sorted(data.glob('*/result.json'), key=lambda p:p.stat().st_mtime, reverse=True)]])

    @app.post('/api/projects')
    def upload():
        uploaded = request.files.get('file')
        if uploaded is None:
            raise ValueError('Choose a PDF, PNG, or JPEG drawing.')
        raw = uploaded.read()
        suffix = Path(uploaded.filename or '').suffix.lower()
        if suffix == '.pdf':
            try:
                with fitz.open(stream=raw, filetype='pdf') as doc:
                    if doc.needs_pass: raise ValueError('Encrypted PDFs are unsupported.')
                    page_number = int(request.form.get('page', 1))
                    if not 1 <= page_number <= len(doc): raise ValueError('Page number is outside this PDF.')
                    page = doc[page_number-1]
                    if page.rect.width <= 0 or page.rect.height <= 0: raise ValueError('Invalid PDF page size.')
                    pix = page.get_pixmap(matrix=fitz.Matrix( min(2,4096/max(page.rect.width,page.rect.height)), min(2,4096/max(page.rect.width,page.rect.height))), colorspace=fitz.csRGB, alpha=False)
                    image = Image.frombytes('RGB', (pix.width,pix.height), pix.samples)
            except (fitz.FileDataError, RuntimeError):
                raise ValueError('Could not decode this PDF.')
        elif suffix in ('.png', '.jpg', '.jpeg'):
            try:
                image = Image.open(io.BytesIO(raw))
                if image.width*image.height > 40_000_000: raise ValueError('Image exceeds 40 megapixels.')
                image = image.convert('RGB')
                page_number = 1
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                raise ValueError('Could not decode this image.')
        else:
            raise ValueError('Supported formats: PDF, PNG, JPEG. Export CAD drawings to one of these first.')
        image.thumbnail((4096,4096))
        regions, source = proposals(np.array(image), model)
        footprint = footprint_proposal(np.array(image))
        removed_regions = 0
        units = []
        inferred_scale = None
        scale_detection={'status':'unavailable','source':'missing','automatic':False,'metres_per_pixel':None,'evidence':[],'message':'This raster image has no embedded vector scale references. Automatic physical scale cannot be established from its pixels alone.'}
        if suffix == '.pdf':
            with fitz.open(stream=raw, filetype='pdf') as doc:
                supports, semantic = legend_geometry(doc[page_number-1],image.width,image.height)
                regions, removed_regions = apply_legend(regions,supports,semantic)
                units = pdf_evidence(doc[page_number-1], image.width, image.height, regions)
                # Recover enclosed service-space geometry missed by the room instance model.
                candidates, _ = proposals(np.array(image), None)
                pdf_evidence(doc[page_number-1], image.width, image.height, candidates)
                from shapely.geometry import Polygon
                for candidate in candidates:
                    if candidate['occupancy']!='mep':continue
                    shape=Polygon(candidate['polygon'])
                    if any(shape.intersection(Polygon(r['polygon'])).area/max(1,shape.area)>.65 for r in regions):continue
                    candidate['id']=max([r['id'] for r in regions],default=0)+1
                    candidate['geometry_source']='enclosed_lines_with_pdf_label'
                    regions.append(candidate)
                scale_detection=auto_scale(doc[page_number-1],image.width)
                inferred_scale=scale_detection['metres_per_pixel']
        result = {'id': uuid.uuid4().hex, 'name': Path(uploaded.filename.replace('\\','/')).name, 'page':page_number, 'width':image.width, 'height':image.height, 'metres_per_pixel':inferred_scale, 'scale_source':scale_detection['source'], 'scale_detection':scale_detection, 'footprint':footprint, 'footprint_source':'ink_outline_proposal', 'scale_confirmed':scale_detection['automatic'], 'footprint_confirmed':False, 'filtered_regions':removed_regions, 'units':units, 'regions':regions, 'model_source':source, 'reviewed':False, 'revision':1}
        folder = data/result['id']; folder.mkdir()
        image.save(folder/'drawing.png')
        if suffix=='.pdf':(folder/'source.pdf').write_bytes(raw)
        result['analysis'] = calculate(result)
        save(folder, result)
        return jsonify(result), 201

    @app.get('/api/projects/<ident>')
    def detail(ident): return jsonify(read(ident))

    @app.get('/api/projects/<ident>/image')
    def drawing(ident): return send_file(directory(ident)/'drawing.png')

    @app.put('/api/projects/<ident>')
    def update(ident):
        body = request.get_json(silent=True)
        if not isinstance(body, dict): raise ValueError('Expected a JSON object.')
        with LOCK:
            result = read(ident)
            if body.get('revision') != result['revision']:
                return jsonify(error='This drawing changed. Reload before saving.'), 409
            previous_scale=result.get('metres_per_pixel')
            previous_footprint=result.get('footprint')
            for field in ('metres_per_pixel', 'footprint', 'regions', 'reviewed', 'units', 'scale_source', 'scale_confirmed', 'footprint_confirmed', 'footprint_source'):
                if field in body: result[field] = body[field]
            if result.get('metres_per_pixel') != previous_scale:
                result['scale_source']='user_calibration'
                result['scale_confirmed']=True
            if result.get('footprint') != previous_footprint:
                result['footprint_source']='user_trace'
                result['footprint_confirmed']=True
            if any(not isinstance(result.get(key,False),bool) for key in ('scale_confirmed','footprint_confirmed')):raise ValueError('Confirmation flags must be true or false.')
            if not isinstance(result['reviewed'], bool): raise ValueError('Review flag must be true or false.')
            if not isinstance(result['regions'], list) or len(result['regions']) > 1000: raise ValueError('Maximum 1000 regions.')
            for region in result['regions']:
                if not isinstance(region, dict) or not all(k in region for k in ('id','name','kind','occupancy','polygon')): raise ValueError('Incomplete region.')
                if not isinstance(region['id'], int) or not isinstance(region['name'],str) or len(region['name'])>200 or not isinstance(region.get('unit',''),str) or len(region.get('unit',''))>100: raise ValueError('Invalid region identifier or name.')
            for points in [r['polygon'] for r in result['regions']] + ([result['footprint']] if result['footprint'] else []):
                polygon(points)
                if any(x<0 or y<0 or x>result['width'] or y>result['height'] for x,y in points): raise ValueError('Vertices must be within the drawing.')
            if not isinstance(result.get('units', []), list) or len(result.get('units', []))>1000: raise ValueError('Invalid units.')
            for unit in result.get('units', []):
                if not isinstance(unit, dict) or not isinstance(unit.get('id'), str) or len(unit['id'])>100 or not isinstance(unit.get('bedrooms'), int) or not 0 <= unit['bedrooms'] <= 20: raise ValueError('Invalid unit record.')
            result['analysis'] = calculate(result)
            result['revision'] += 1
            save(directory(ident), result)
        return jsonify(result)

    @app.get('/api/projects/<ident>/export')
    def export(ident):
        result = read(ident)
        out = io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('analysis.json', json.dumps(result,indent=2))
            csvout = io.StringIO()
            writer = csv.DictWriter(csvout,fieldnames=['id','name','occupancy','unit','area_m2','area_estimate_m2','area_pixels'])
            writer.writeheader(); writer.writerows(result['analysis']['rooms'])
            archive.writestr('rooms.csv', csvout.getvalue())
            archive.write(directory(ident)/'drawing.png','drawing.png')
            archive.writestr('README.txt','Areas use the recorded automatic scale references or a manual override. Estimates are separate from checked measurements. Gross area is the traced outer boundary; net area is room union minus exclusions. Unknown residual is not automatically walls. Predictions and classifications require review. See analysis.json for provenance and warnings.')
        out.seek(0)
        return send_file(out,mimetype='application/zip',as_attachment=True,download_name='floorreview-results.zip')

    @app.get('/api/projects/<ident>/labels')
    def labels(ident):
        result = read(ident)
        if not result['reviewed'] or not result['footprint'] or not result['analysis']['measurement_status']['boundary_confirmed']:
            raise ValueError('Trace the footprint and mark the drawing reviewed before exporting training labels.')
        mask = np.zeros((result['height'],result['width']),np.uint8)
        cv2.fillPoly(mask,[np.rint(result['footprint']).astype(np.int32)],255)
        # 255 is ignored/unlabeled until fully annotated; never silently label unknown pixels as walls.
        classes = {'wall':1,'room':2,'stairs':3,'corridor':4,'void':0}
        for region in sorted(result['regions'], key=lambda r: r['kind']=='room', reverse=True):
            class_id = 5 if region['kind']=='room' and region['occupancy']=='mep' else classes[region['kind']]
            cv2.fillPoly(mask,[np.rint(region['polygon']).astype(np.int32)],class_id)
        out = io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(directory(ident)/'drawing.png',f'images/{ident}.png')
            archive.writestr(f'masks/{ident}.png', cv2.imencode('.png',mask)[1].tobytes())
            archive.writestr('annotation.json',json.dumps(result))
        out.seek(0)
        return send_file(out,mimetype='application/zip',as_attachment=True,download_name='floorreview-labels.zip')
    @app.post('/api/projects/<ident>/auto-scale')
    def automatic_scale(ident):
        body=request.get_json(silent=True) or {}
        if not isinstance(body,dict):raise ValueError('Expected a JSON object.')
        with LOCK:
            result=read(ident)
            if body.get('revision')!=result['revision']:
                return jsonify(error='Reload this drawing before automatic scale detection.'),409
            path=directory(ident)/'source.pdf'
            if not path.exists():raise ValueError('Re-upload the original dimensioned PDF to run automatic scale detection. This older result only retained a rendered image.')
            original=path.read_bytes()
        with fitz.open(stream=original,filetype='pdf') as doc:
            detection=auto_scale(doc[result['page']-1],result['width'])
        with LOCK:
            current=read(ident)
            if current['revision']!=result['revision']:
                return jsonify(error='The drawing changed during scale detection. Reload and try again.'),409
            if current.get('scale_source')=='user_calibration':
                return jsonify(error='A manual override is active. Re-upload to analyze the PDF without replacing your saved manual measurement.'),409
            current.update(metres_per_pixel=detection['metres_per_pixel'],scale_source=detection['source'],scale_detection=detection,scale_confirmed=detection['automatic'])
            current['analysis']=calculate(current);current['revision']+=1
            save(directory(ident),current)
        return jsonify(current)

    @app.get('/api/gemini/status')
    def gemini_status():
        return jsonify(gemini.status())

    @app.post('/api/projects/<ident>/gemini')
    def gemini_review(ident):
        body=request.get_json(silent=True) or {}
        if not isinstance(body,dict) or body.get('consent') is not True:
            raise ValueError('Confirm that you want to send this drawing image to Google Gemini.')
        with LOCK:
            result=read(ident)
            if body.get('revision')!=result['revision']:
                return jsonify(error='Save or reload the drawing before requesting Gemini review.'),409
            image=(directory(ident)/'drawing.png').read_bytes()
        review=gemini.review(result,image,body.get('phase','occupancy'))
        return jsonify(review)

    @app.post('/api/projects/<ident>/gemini/apply')
    def gemini_apply(ident):
        body=request.get_json(silent=True) or {}
        if not isinstance(body,dict):raise ValueError('Expected a JSON object.')
        key=body.get('review_id')
        if not isinstance(key,str) or len(key)!=64:raise ValueError('Invalid review identifier.')
        review=gemini.cached(key)
        if not review:raise ValueError('Review was not found. Run a Gemini review first.')
        selected=body.get('region_ids')
        if not isinstance(selected,list) or not selected or any(type(i) is not int for i in selected) or len(set(selected))!=len(selected):
            raise ValueError('Select one or more distinct suggestions to apply.')
        suggestions={r['id']:r for r in review['suggestions']}
        if any(i not in suggestions for i in selected):raise ValueError('Selected suggestion was not found.')
        with LOCK:
            result=read(ident)
            if body.get('revision')!=result['revision']:
                return jsonify(error='The drawing changed. Reload before applying suggestions.'),409
            image=(directory(ident)/'drawing.png').read_bytes()
            if fingerprint(result,image,review['phase'],review['model'])!=key:
                return jsonify(error='These suggestions describe an older drawing state. Request a new review.'),409
            for region in result['regions']:
                if region['id'] not in selected:continue
                suggestion=suggestions[region['id']]
                for field in ('name','kind','occupancy'):region[field]=suggestion[field]
                region['semantic_source']='gemini_suggestion'
                region['gemini_confidence']=suggestion['confidence']
                region['reviewed']=False
            result['reviewed']=False
            result['gemini_last_review']={'review_id':key,'model':review['model'],'phase':review['phase'],'applied_ids':selected}
            result['analysis']=calculate(result)
            result['revision']+=1
            save(directory(ident),result)
        return jsonify(result)

    from .batch import register_queue
    register_queue(app, data, upload)
    return app

if __name__ == '__main__':
    from waitress import serve
    serve(create_app(), host=os.environ.get('FLOORREVIEW_HOST','127.0.0.1'), port=int(os.environ.get('PORT','8765')), threads=4)
