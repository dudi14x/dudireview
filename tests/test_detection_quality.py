import cv2
import numpy as np
from floorreview.detection_quality import screen_predictions


def test_grid_box_flagged_wall_supported_room_kept():
    image=np.full((300,500,3),255,np.uint8)
    cv2.rectangle(image,(30,30),(180,180),(210,210,210),1)
    cv2.putText(image,'11050',(65,100),cv2.FONT_HERSHEY_SIMPLEX,.3,(70,70,70),1)
    cv2.rectangle(image,(260,30),(460,240),(30,30,30),4)
    def region(i,points):return {'id':i,'name':f'Space {i}','kind':'room','occupancy':'unclassified','geometry_source':'archvision_yolo','polygon':points}
    false=region(1,[[30,30],[180,30],[180,180],[30,180]])
    true=region(2,[[264,34],[456,34],[456,236],[264,236]])
    kept,rejected=screen_predictions(image,[false,true])
    assert [r['id'] for r in kept]==[2]
    assert [r['id'] for r in rejected]==[1]
    assert 'boundary_evidence' not in false
    false['semantic_source']='contained_pdf_label'
    kept,rejected=screen_predictions(image,[false])
    assert len(kept)==1 and not rejected


def test_manual_regions_are_not_automatically_removed():
    region={'id':1,'polygon':[[0,0],[20,0],[20,20],[0,20]],'kind':'room','occupancy':'unclassified'}
    kept,rejected=screen_predictions(np.full((100,100,3),255,np.uint8),[region])
    assert kept==[region] and not rejected


def test_flagged_prediction_restore_and_revision_checks(tmp_path,monkeypatch):
    import io,re
    from PIL import Image
    from floorreview.app import create_app
    candidate={'id':1,'name':'Space 1','kind':'room','occupancy':'unclassified','unit':'','geometry_source':'archvision_yolo','polygon':[[20,20],[180,20],[180,180],[20,180]]}
    monkeypatch.setattr('floorreview.app.proposals',lambda *args:([dict(candidate)],'archvision_pretrained'))
    app=create_app(tmp_path);client=app.test_client()
    token=re.search("window.FR_TOKEN='([^']+)'",client.get('/').text)[1]
    headers={'X-FloorReview-Token':token}
    raw=io.BytesIO();Image.new('RGB',(240,240),'white').save(raw,format='PNG');raw.seek(0)
    response=client.post('/api/projects',data={'file':(raw,'blank.png')},headers=headers)
    assert response.status_code==201,response.text
    result=response.json
    assert not result['regions'] and len(result['rejected_predictions'])==1
    url=f"/api/projects/{result['id']}/restore-prediction"
    assert client.post(url,json={'revision':0,'region_id':1},headers=headers).status_code==409
    restored=client.post(url,json={'revision':1,'region_id':1},headers=headers)
    assert restored.status_code==200,restored.text
    result=restored.json
    assert len(result['regions'])==1 and not result['rejected_predictions']
    checked=client.post(f"/api/projects/{result['id']}/screen-predictions",json={'revision':result['revision']},headers=headers)
    assert checked.status_code==200 and len(checked.json['regions'])==1
