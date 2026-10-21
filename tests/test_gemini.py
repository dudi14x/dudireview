import copy
import io
import json
import re
import urllib.error
import cv2
import numpy as np
import pytest
from PIL import Image
from floorreview.app import create_app
from floorreview.gemini import GeminiReview,GeminiError,validate_report


def result():
    return {'revision':1,'regions':[{'id':1,'name':'Space 1','kind':'room','occupancy':'unclassified','unit':'','polygon':[[20,20],[100,20],[100,100],[20,100]]}],'footprint':None,'metres_per_pixel':None}

def image():
    stream=io.BytesIO();Image.new('RGB',(150,150),'white').save(stream,format='PNG');return stream.getvalue()

def answer():
    return {'summary':'Possible service space.','issues':['Boundary needs checking.'],'suggestions':[{'id':1,'name':'Service room','kind':'room','occupancy':'mep','confidence':.8,'evidence':'Visible service equipment layout.'}]}

class Response:
    def __init__(self,value):self.value=value
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,*args):return json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(self.value)}]}}],'usageMetadata':{'totalTokenCount':40}}).encode()


def test_missing_key_and_daily_budget(monkeypatch,tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False);monkeypatch.delenv('GOOGLE_API_KEY',raising=False)
    service=GeminiReview(tmp_path)
    assert not service.status()['configured']
    with pytest.raises(GeminiError,match='not configured'):service.review(result(),image(),'occupancy')
    monkeypatch.setenv('GEMINI_API_KEY','test-only-key');monkeypatch.setenv('GEMINI_DAILY_REQUEST_LIMIT','0')
    with pytest.raises(GeminiError,match='daily request limit'):service.review(result(),image(),'occupancy')


def test_request_cache_and_no_mutation(monkeypatch,tmp_path):
    monkeypatch.setenv('GEMINI_API_KEY','test-only-key');monkeypatch.setenv('GEMINI_DAILY_REQUEST_LIMIT','1')
    calls=[]
    def request(req,timeout):
        body=json.loads(req.data);assert req.full_url.startswith('https://generativelanguage.googleapis.com/')
        assert 'test-only-key' not in req.full_url
        assert body['generationConfig']['responseMimeType']=='application/json'
        assert any('inlineData' in p for p in body['contents'][0]['parts'])
        calls.append(req);return Response(answer())
    monkeypatch.setattr('floorreview.gemini.urllib.request.urlopen',request)
    service=GeminiReview(tmp_path);before=result();saved=copy.deepcopy(before)
    first=service.review(before,image(),'occupancy');second=service.review(before,image(),'occupancy')
    assert before==saved and not first['cached'] and second['cached'] and len(calls)==1
    assert service.status()['used_today']==1
    with pytest.raises(GeminiError,match='daily request limit'):service.review(before,image(),'rooms')


@pytest.mark.parametrize('change',[lambda a:a['suggestions'][0].update(id=99),lambda a:a['suggestions'][0].update(area_m2=250),lambda a:a['suggestions'][0].update(confidence=float('nan')),lambda a:a['suggestions'].append(a['suggestions'][0]),lambda a:a.update(summary=123)])
def test_reject_hallucinated_or_invalid_fields(change):
    value=answer();change(value)
    with pytest.raises(GeminiError):validate_report(value,{1})


def test_quota_error_is_sanitized_without_retry(monkeypatch,tmp_path):
    monkeypatch.setenv('GEMINI_API_KEY','test-only-secret');monkeypatch.setenv('GEMINI_DAILY_REQUEST_LIMIT','5')
    calls=[]
    def failed(req,timeout):
        calls.append(req);raise urllib.error.HTTPError(req.full_url,429,'raw secret provider error',{},None)
    monkeypatch.setattr('floorreview.gemini.urllib.request.urlopen',failed)
    service=GeminiReview(tmp_path)
    with pytest.raises(GeminiError) as exc:service.review(result(),image(),'floor')
    assert exc.value.status==429 and 'secret' not in str(exc.value) and len(calls)==1


def test_api_review_consent_apply_and_stale_state(monkeypatch,tmp_path):
    monkeypatch.setenv('FLOORREVIEW_MODEL',str(tmp_path/'missing'))
    monkeypatch.setenv('GEMINI_API_KEY','test-only-secret');monkeypatch.setenv('GEMINI_DAILY_REQUEST_LIMIT','5')
    monkeypatch.setattr('floorreview.gemini.urllib.request.urlopen',lambda req,timeout:Response(answer()))
    app=create_app(tmp_path);app.testing=True;client=app.test_client()
    token=re.search("window.FR_TOKEN='([^']+)'",client.get('/').text)[1];headers={'X-FloorReview-Token':token}
    im=np.full((150,150,3),255,np.uint8);cv2.rectangle(im,(20,20),(130,130),(0,0,0),4)
    r=client.post('/api/projects',data={'file':(io.BytesIO(cv2.imencode('.png',im)[1].tobytes()),'plan.png')},headers=headers).json
    base='/api/projects/'+r['id']
    assert client.post(base+'/gemini',json={'revision':1},headers=headers).status_code==400
    assert client.post(base+'/gemini',json={'revision':1,'consent':True}).status_code==403
    report=client.post(base+'/gemini',json={'revision':1,'consent':True,'phase':'occupancy'},headers=headers)
    assert report.status_code==200,report.text
    assert 'test-only-secret' not in client.get('/api/gemini/status').text
    original=client.get(base).json
    applied=client.post(base+'/gemini/apply',json={'review_id':report.json['review_id'],'region_ids':[1],'revision':1},headers=headers)
    assert applied.status_code==200,applied.text
    updated=applied.json
    assert updated['regions'][0]['occupancy']=='mep'
    assert updated['regions'][0]['polygon']==original['regions'][0]['polygon']
    for field in ('footprint','metres_per_pixel','units','scale_confirmed','footprint_confirmed'):assert updated[field]==original[field]
    assert updated['reviewed'] is False
    assert client.post(base+'/gemini/apply',json={'review_id':report.json['review_id'],'region_ids':[1],'revision':updated['revision']},headers=headers).status_code==409
