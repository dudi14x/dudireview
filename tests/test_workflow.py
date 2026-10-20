import io
import json
import re
import zipfile
import cv2
import fitz
import numpy as np
import pytest
from floorreview.app import create_app
from floorreview.geometry import calculate
from floorreview.vision import proposals
from floorreview.training import synthetic, train


def rect(x,y,w,h):return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
def region(i,points,kind='room',occupancy='unclassified'):
    return dict(id=i,name=f'Space {i}',polygon=points,kind=kind,occupancy=occupancy,unit='')


def test_area_exclusions_and_overlap():
    r={'metres_per_pixel':.1,'scale_confirmed':True,'footprint_confirmed':True,'footprint':rect(0,0,100,100),'regions':[region(1,rect(0,0,60,100)),region(2,rect(40,0,60,100)),region(3,rect(0,0,100,10),'corridor')]}
    a=calculate(r)
    assert a['gross_floor_area_m2']==100
    assert a['net_room_area_m2']==90
    assert a['excluded_area_m2']==10
    assert sum(x['area_m2'] for x in a['rooms'])==108
    assert any('overlap' in w for w in a['warnings'])


def test_no_scale_no_fabricated_metric_area():
    assert calculate({'regions':[region(1,rect(0,0,10,10))]})['net_room_area_m2'] is None


def test_invalid_geometry():
    with pytest.raises(ValueError):calculate({'regions':[region(1,[[0,0],[10,10],[0,10],[10,0]])]})
    with pytest.raises(ValueError):calculate({'metres_per_pixel':float('nan'),'regions':[]})


def test_closed_room_detection():
    im=np.full((256,256,3),255,np.uint8)
    cv2.rectangle(im,(30,30),(220,220),(0,0,0),5)
    rooms,source=proposals(im)
    assert source=='geometry'
    assert len(rooms)==1


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv("FLOORREVIEW_MODEL",str(tmp_path/"missing.joblib"))
    app=create_app(tmp_path);app.testing=True
    c=app.test_client()
    token=re.search("window.FR_TOKEN='([^']+)'",c.get('/').text)[1]
    return c,{'X-FloorReview-Token':token}


def test_upload_review_export_and_reload(client):
    c,headers=client
    doc=fitz.open();page=doc.new_page(width=300,height=300)
    page.draw_rect(fitz.Rect(30,30,270,270),width=4)
    page.insert_text((70,150),'2BR-A')
    raw=doc.tobytes()
    assert c.post('/api/projects',data={}).status_code==403
    response=c.post('/api/projects',data={'file':(io.BytesIO(raw),'plan.pdf')},headers=headers)
    assert response.status_code==201,response.text
    r=response.json; assert r['regions']; assert r['analysis']['unit_counts']=={'2 bed':1}
    r.update(metres_per_pixel=.1,footprint=rect(0,0,300,300),regions=[region(1,rect(20,20,100,100),'room','residential'),region(2,rect(130,20,100,100),'room','mep')],reviewed=True)
    saved=c.put('/api/projects/'+r['id'],json=r,headers=headers)
    assert saved.status_code==200,saved.text
    assert saved.json['analysis']['net_room_area_m2']==200
    assert saved.json['analysis']['mep_space_count']==1
    assert c.put('/api/projects/'+r['id'],json=r,headers=headers).status_code==409
    assert c.get('/api/projects/'+r['id']).json['reviewed']
    export=c.get('/api/projects/'+r['id']+'/export')
    with zipfile.ZipFile(io.BytesIO(export.data)) as z:
        assert 'rooms.csv' in z.namelist()
        assert json.loads(z.read('analysis.json'))['analysis']['net_room_area_m2']==200
    labels=c.get('/api/projects/'+r['id']+'/labels');assert labels.status_code==200
    with zipfile.ZipFile(io.BytesIO(labels.data)) as z:
        mask=cv2.imdecode(np.frombuffer(z.read('masks/'+r['id']+'.png'),np.uint8),0)
        assert mask[30,30]==2 and mask[30,140]==5 and mask[290,290]==255


def test_reject_bad_files_and_coordinates(client):
    c,h=client
    assert c.post('/api/projects',data={'file':(io.BytesIO(b'bad'),'bad.pdf')},headers=h).status_code==400
    assert c.post('/api/projects',data={'file':(io.BytesIO(b'bad'),'bad.png')},headers=h).status_code==400
    assert c.get('/api/projects/not-an-id').status_code==404


def test_training_heldout_and_inference(tmp_path):
    synthetic(tmp_path/'dataset',n=8)
    report=train(tmp_path/'dataset',tmp_path/'model.joblib','synthetic')
    assert report['validation_drawings']==12
    assert report['class_iou']['wall']>.5
    assert report['class_iou']['stairs'] is None
    image=cv2.imread(str(tmp_path/'dataset/val/images/000.png'))
    regions,source=proposals(image,tmp_path/'model.joblib')
    assert source=='synthetic' and len(regions)==4


def test_batch_queue_processing_and_failure(client):
    import time
    c,h=client
    im=np.full((150,150,3),255,np.uint8);cv2.rectangle(im,(20,20),(130,130),(0,0,0),4)
    blob=cv2.imencode('.png',im)[1].tobytes()
    response=c.post('/api/queue',data={'file':(io.BytesIO(blob),'batch.png')},headers=h)
    assert response.status_code==202
    assert c.post('/api/queue',data={'file':(io.BytesIO(b'not pdf'),'broken.pdf')},headers=h).status_code==202
    for _ in range(100):
        q=c.get('/api/queue').json
        if q['counts'].get('ready')==1 and q['counts'].get('failed')==1:break
        time.sleep(.03)
    assert q['capacity']==1000 and q['workers']==2
    assert q['counts']=={'ready':1,'failed':1}
    assert next(j for j in q['jobs'] if j['status']=='ready')['project_id']


def test_queue_capacity(client,tmp_path):
    import sqlite3
    c,h=client
    with sqlite3.connect(tmp_path/'queue.sqlite') as conn:
        conn.executemany('INSERT INTO queue(id,name,page,status) VALUES(?,?,1,?)',[(f'{i:032x}','sample.png','queued') for i in range(1000)])
    response=c.post('/api/queue',data={'file':(io.BytesIO(b'png'),'over-limit.png')},headers=h)
    assert response.status_code==429
    assert c.get('/api/queue').json['counts']['queued']==1000


def test_neural_training_and_checkpoint(tmp_path):
    from floorreview.neural import train_unet,load_model,infer
    synthetic(tmp_path/'tiny',n=2)
    report=train_unet(tmp_path/'tiny',tmp_path/'plan.pt',epochs=1,source='synthetic')
    assert report['train_drawings']==2 and report['validation_drawings']==12
    assert report['source']=='synthetic' and report['class_iou']['mep'] is None
    model,state=load_model(tmp_path/'plan.pt')
    image=cv2.imread(str(tmp_path/'tiny/val/images/000.png'))
    mask,source=infer(image,tmp_path/'plan.pt')
    assert mask.shape==image.shape[:2] and source=='synthetic'


def test_page_frame_is_not_a_floor():
    from floorreview.vision import footprint_proposal
    from shapely.geometry import Polygon
    image=np.full((400,500,3),255,np.uint8)
    cv2.rectangle(image,(8,8),(490,390),(0,0,0),3)
    cv2.rectangle(image,(100,100),(350,280),(0,0,0),4)
    proposal=footprint_proposal(image)
    assert proposal is not None
    bounds=Polygon(proposal).bounds
    assert bounds[0]>80 and bounds[1]>80 and bounds[2]<370 and bounds[3]<300


def test_pdf_unit_marker_conventions_and_legend():
    from floorreview.semantics import pdf_evidence
    doc=fitz.open();page=doc.new_page(width=500,height=500)
    for x,y,text in [(50,50,'1BR'),(50,80,'1BD'),(100,50,'2BR-A'),(380,150,'LEGEND'),(400,180,'1BR')]:page.insert_text((x,y),text)
    units=pdf_evidence(page,500,500,[])
    assert sorted(u['bedrooms'] for u in units)==[1,2]


def test_windows_default_encoding_dashboard_and_unicode_projects(client,tmp_path,monkeypatch):
    """Reproduce a Windows CP1252 default without needing a Windows host."""
    from pathlib import Path
    original_read=Path.read_text
    original_write=Path.write_text
    def windows_read(self,encoding=None,errors=None):
        return original_read(self,encoding=encoding or 'cp1252',errors=errors)
    def windows_write(self,data,encoding=None,errors=None,newline=None):
        return original_write(self,data,encoding=encoding or 'cp1252',errors=errors,newline=newline)
    monkeypatch.setattr(Path,'read_text',windows_read)
    monkeypatch.setattr(Path,'write_text',windows_write)
    c,h=client
    response=c.get('/')
    assert response.status_code==200
    assert 'FloorReview — Drawing intelligence' in response.text
    assert 'm²' in response.text
    ident='a'*32
    folder=tmp_path/ident;folder.mkdir()
    name='Étage – مخطط الطابق'
    (folder/'result.json').write_text(json.dumps({'id':ident,'name':name},ensure_ascii=False),encoding='utf-8')
    assert c.get('/api/projects/'+ident).json['name']==name
    assert c.get('/api/projects').json[0]['name']==name


def test_unverified_scale_and_outline_do_not_create_measurements_or_clip_rooms():
    r={'metres_per_pixel':.1,'scale_source':'printed_ratio_unverified','footprint_source':'ink_outline_proposal','footprint':rect(0,0,10,10),'regions':[region(1,rect(0,0,20,10))]}
    a=calculate(r)
    assert a['gross_floor_area_m2'] is None and a['net_room_area_m2'] is None
    assert a['estimates']['net_detected_room_area_m2']==2
    r['scale_confirmed']=True
    a=calculate(r)
    assert a['net_room_area_m2']==2 and a['gross_floor_area_m2'] is None
    r['footprint_confirmed']=True
    a=calculate(r)
    assert a['gross_floor_area_m2']==1 and a['net_room_area_m2']==1
    assert not a['measurement_status']['complete_floor_net_area']


def test_concave_floor_area_and_render_resolution_invariance():
    points=[[0,0],[100,0],[100,50],[50,50],[50,100],[0,100]]
    for factor in (.5,1,2):
        scaled=[[x*factor,y*factor] for x,y in points]
        a=calculate({'metres_per_pixel':.1/factor,'scale_confirmed':True,'footprint_confirmed':True,'footprint':scaled,'regions':[region(1,scaled)]})
        assert a['gross_floor_area_m2']==75 and a['net_room_area_m2']==75


def test_explicit_pdf_legend_recovers_mep_and_rejects_false_room():
    from floorreview.pdf_geometry import legend_geometry,apply_legend
    doc=fitz.open();page=doc.new_page(width=400,height=300)
    page.insert_text((300,40),'LEGEND')
    page.draw_rect(fitz.Rect(300,50,310,60),fill=(.7,.7,.7))
    page.insert_text((315,59),'MEP')
    page.draw_rect(fitz.Rect(40,80,80,140),fill=(.7,.7,.7))
    support,semantic=legend_geometry(page,400,300)
    assert len(semantic)==1 and semantic[0]['occupancy']=='mep'
    assert semantic[0]['polygon'] and len(support)==1
    regions,removed=apply_legend([region(1,rect(150,10,30,30))],support,semantic)
    assert removed==1 and len(regions)==1 and regions[0]['occupancy']=='mep'
    assert regions[0]['geometry_source']=='pdf_legend_vector'


def test_gray_without_legend_is_not_automatically_mep():
    from floorreview.pdf_geometry import legend_geometry
    doc=fitz.open();page=doc.new_page()
    page.draw_rect(fitz.Rect(40,80,80,140),fill=(.7,.7,.7))
    assert legend_geometry(page,400,300)==([],[])
