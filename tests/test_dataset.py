import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from floorreview.dataset import prepare


def export(path, value):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('annotation.json',json.dumps({'reviewed':True,'footprint_confirmed':True}))
        for folder, array in [('images',np.full((16,16,3),value,np.uint8)),('masks',np.full((16,16),2,np.uint8))]:
            output=io.BytesIO();Image.fromarray(array).save(output,format='PNG')
            archive.writestr(f'{folder}/drawing.png',output.getvalue())


def test_dataset_building_split_and_duplicate_protection(tmp_path):
    export(tmp_path/'a.zip',100);export(tmp_path/'b.zip',200)
    rows=[{'archive':'a.zip','building':'A','split':'train'}, {'archive':'b.zip','building':'B','split':'val'}]
    manifest=tmp_path/'labels.json';manifest.write_text(json.dumps(rows))
    result=prepare(manifest,tmp_path/'dataset')
    assert len(result)==2
    assert len(list((tmp_path/'dataset/train/masks').glob('*.png')))==1
    rows[1]['building']='A';manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='building'):prepare(manifest,tmp_path/'leaked')
    rows[1]['building']='B';rows[1]['archive']='a.zip';manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Duplicate'):prepare(manifest,tmp_path/'duplicate')
