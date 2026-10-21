"""Build supervised masks from reviewed dashboard exports, split by building."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(manifest, output):
    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    groups, images, samples = {}, {}, []
    for row in rows:
        split, building = row['split'], str(row['building']).strip()
        if split not in ('train', 'val', 'test') or not building:
            raise ValueError('Each export needs a building and train/val/test split.')
        if building in groups and groups[building] != split:
            raise ValueError('A building cannot appear in multiple splits.')
        groups[building] = split
        with zipfile.ZipFile(manifest.parent / row['archive']) as archive:
            if sum(info.file_size for info in archive.infolist()) > 200_000_000:
                raise ValueError('Export exceeds 200 MB expanded limit.')
            annotation = json.loads(archive.read('annotation.json'))
            if not annotation.get('reviewed') or not annotation.get('footprint_confirmed'):
                raise ValueError('Only reviewed boundary-confirmed exports are training labels.')
            image_names = [n for n in archive.namelist() if n.startswith('images/') and n.endswith('.png')]
            mask_names = [n for n in archive.namelist() if n.startswith('masks/') and n.endswith('.png')]
            if len(image_names) != 1 or len(mask_names) != 1:
                raise ValueError('Each export must contain one image and one mask.')
            image = Image.open(io.BytesIO(archive.read(image_names[0]))).convert('RGB')
            mask = Image.open(io.BytesIO(archive.read(mask_names[0])))
            pixels = np.asarray(mask)
            if mask.mode != 'L' or image.size != mask.size or not np.isin(pixels, [0,1,2,3,4,5,255]).all():
                raise ValueError('Invalid mask dimensions or class IDs.')
            if not np.any((pixels != 255) & (pixels != 0)):
                raise ValueError('Export has no labeled spaces or walls.')
            signature = hashlib.sha256(image.tobytes()).hexdigest()
            if signature in images:
                raise ValueError('Duplicate drawing; export each drawing only once.')
            images[signature] = split
            samples.append((split, building, signature, image, mask.copy()))
    if not {'train','val'}.issubset({s[0] for s in samples}):
        raise ValueError('Separate training and validation buildings are required.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory to avoid mixing datasets.')
    entries = []
    for split, building, signature, image, mask in samples:
        for folder, value in [('images', image), ('masks', mask)]:
            target = output/split/folder/f'{signature}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            value.save(target)
        entries.append({'split':split, 'building':building, 'file':f'{signature}.png'})
    (output/'manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='datasets/reviewed')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest,args.output),indent=2))
