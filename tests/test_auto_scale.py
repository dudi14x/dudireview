import io
import math
import fitz
import pytest
from floorreview.auto_scale import auto_scale


def bar_page():
    document=fitz.open();page=document.new_page(width=400,height=300)
    page.draw_line((50,120),(130,120))
    for x,label in [(50,'0'),(90,'2'),(130,'4')]:
        page.draw_line((x,117),(x,125))
        page.insert_text((x-3,110),label,fontsize=10)
    page.insert_text((140,130),'m',fontsize=10)
    return document,page


def test_scale_bar_automatically_supplies_scale_and_survives_resizing_rotation():
    doc,page=bar_page()
    scale=auto_scale(page,800)
    assert scale['status']=='resolved' and scale['automatic']
    assert scale['metres_per_pixel']==pytest.approx(.025,rel=.005)
    assert auto_scale(page,400)['metres_per_pixel']==pytest.approx(scale['metres_per_pixel']*2)
    page.set_rotation(90)
    assert auto_scale(page,600)['metres_per_pixel']==pytest.approx(.025,rel=.005)


def test_numeric_grid_without_unit_or_ticks_is_not_a_scale_bar():
    doc=fitz.open();p=doc.new_page()
    for x,text in [(50,'0'),(90,'2'),(130,'4')]:p.insert_text((x,100),text)
    assert auto_scale(p,800)['status']=='unavailable'


def test_conflicting_declared_scales_are_not_used_globally():
    doc,p=bar_page();p.insert_text((30,40),'Scale 1:100');p.insert_text((150,40),'Scale 1:200')
    r=auto_scale(p,800)
    assert r['status']=='ambiguous' and r['metres_per_pixel'] is None


def test_declared_ratio_without_dimensions_is_an_estimate():
    doc=fitz.open();p=doc.new_page();p.insert_text((30,40),'Scale 1:100')
    r=auto_scale(p,1000)
    assert r['status']=='declared_only' and not r['automatic']
    assert r['metres_per_pixel']==pytest.approx(100*.0254/72*p.rect.width/1000)


def test_dimension_annotations_with_units_supply_automatic_scale():
    doc=fitz.open();p=doc.new_page(width=600,height=400)
    p.insert_text((20,30),'ALL DIMENSIONS ARE IN MILLIMETERS')
    for y,length,label in [(100,100,'5000'),(180,200,'10000'),(260,300,'15000')]:
        p.draw_line((40,y),(40+length,y))
        p.insert_text((40+length/2-10,y-3),label,fontsize=9)
    r=auto_scale(p,1200)
    assert r['status']=='resolved' and r['automatic']
    assert r['metres_per_pixel']==pytest.approx(.025)
    refs=next(e for e in r['evidence'] if e['method']=='dimension_lines')
    assert refs['matched_count']==3


def test_conflicting_graphic_bars_do_not_guess_a_scale():
    doc,p=bar_page()
    p.draw_line((200,220),(280,220))
    for x,label in [(200,'0'),(240,'4'),(280,'8')]:
        p.draw_line((x,217),(x,225));p.insert_text((x-3,210),label,fontsize=10)
    p.insert_text((290,230),'m',fontsize=10)
    assert auto_scale(p,800)['status']=='ambiguous'


def test_upload_pdf_resolves_scale_without_user_input(client):
    c,headers=client
    doc,page=bar_page();page.draw_rect(fitz.Rect(20,150,160,280),width=3)
    response=c.post('/api/projects',data={'file':(io.BytesIO(doc.tobytes()),'dimensioned.pdf')},headers=headers)
    assert response.status_code==201,response.text
    r=response.json
    assert r['scale_confirmed'] is True and r['scale_source']=='pdf_auto_references'
    assert r['analysis']['net_room_area_m2'] is not None
    again=c.post('/api/projects/'+r['id']+'/auto-scale',json={'revision':r['revision']},headers=headers)
    assert again.status_code==200,again.text
    assert again.json['metres_per_pixel']==r['metres_per_pixel']
