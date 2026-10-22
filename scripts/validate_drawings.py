"""Validate actual uploads without claiming ground-truth accuracy."""
import io,json,re,time
from pathlib import Path
from floorreview.app import create_app
app=create_app();client=app.test_client()
token=re.search("window.FR_TOKEN='([^']+)'",client.get('/').text)[1]
reports=[]
for path in Path('/workspace/attachments').glob('*/*.pdf'):
    start=time.monotonic()
    response=client.post('/api/projects',data={'file':(io.BytesIO(path.read_bytes()),path.name)},headers={'X-FloorReview-Token':token})
    if response.status_code!=201:raise RuntimeError(response.text)
    r=response.json
    reports.append({'name':path.name,'id':r['id'],'seconds':round(time.monotonic()-start,2),'model':r['model_source'],'region_candidates':len(r['regions']),'unit_candidates':r['analysis']['unit_counts'],'mep_candidates':r['analysis']['mep_space_count'],'footprint_vertices':len(r['footprint'] or []),'scale':r['metres_per_pixel'],'accuracy_status':'No reviewed ground truth; detection counts are not accuracy scores.'})
    print(json.dumps(reports[-1]),flush=True)
Path('data/real-drawing-validation.json').write_text(json.dumps(reports,indent=2), encoding='utf-8')
