import fitz
import pytest
from floorreview.semantics import pdf_evidence


@pytest.mark.parametrize('rotation',[0,90,180,270])
def test_labels_align_with_rotated_raster(rotation):
    doc=fitz.open();page=doc.new_page(width=400,height=300)
    page.insert_text((80,100),'ELEC ROOM')
    page.insert_text((80,180),'2 BED')
    page.set_rotation(rotation)
    bbox=page.search_for('ELEC ROOM')[0]*page.rotation_matrix
    region={'id':1,'name':'Space','kind':'room','occupancy':'unclassified','polygon':[[bbox.x0-2,bbox.y0-2],[bbox.x1+2,bbox.y0-2],[bbox.x1+2,bbox.y1+2],[bbox.x0-2,bbox.y1+2]]}
    units=pdf_evidence(page,page.rect.width,page.rect.height,[region])
    assert region['occupancy']=='mep'
    assert len(units)==1 and units[0]['bedrooms']==2
    expected=page.search_for('2 BED')[0]*page.rotation_matrix
    assert expected.contains(fitz.Point(*units[0]['position']))
