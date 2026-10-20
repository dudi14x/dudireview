"""Train a small reproducible CPU semantic pixel classifier on paired images/masks."""
import argparse
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from .vision import features, CLASSES


def synthetic(destination, n=48):
    destination = Path(destination)
    used = set()
    for split, count in [('train', n), ('val', 12)]:
        for name in ('images', 'masks'):
            (destination/split/name).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42 if split == 'train' else 2026)
        for i in range(count):
            image = np.full((256, 256, 3), 255, np.uint8)
            mask = np.zeros((256, 256), np.uint8)
            while True:
                xmid, ymid = rng.integers(95, 160, size=2)
                signature = (int(xmid), int(ymid))
                if signature not in used:
                    used.add(signature)
                    break
            for x1, y1, x2, y2 in [(20,20,xmid,ymid),(xmid,20,235,ymid),(20,ymid,xmid,235),(xmid,ymid,235,235)]:
                cv2.rectangle(mask,(x1,y1),(x2,y2),2,-1)
                cv2.rectangle(image,(x1,y1),(x2,y2),(25,25,25),5)
                cv2.rectangle(mask,(x1,y1),(x2,y2),1,5)
            cv2.imwrite(str(destination/split/'images'/f'{i:03}.png'), image)
            cv2.imwrite(str(destination/split/'masks'/f'{i:03}.png'), mask)


def pairs(root, split):
    root = Path(root)/split
    images = sorted((root/'images').glob('*.png'))
    if not images:
        raise ValueError(f'No PNG training images in {root}/images')
    for path in images:
        image = cv2.imread(str(path))
        mask = cv2.imread(str(root/'masks'/path.name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f'Missing or mismatched mask: {path.name}')
        if np.any((mask >= len(CLASSES)) & (mask != 255)):
            raise ValueError('Mask IDs must be 0–5 or 255 (ignored).')
        yield path.name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask


def train(data, output, source='annotated'):
    rng = np.random.default_rng(42)
    xs, ys, train_names, train_signatures = [], [], set(), set()
    import hashlib
    for name, image, mask in pairs(data, 'train'):
        train_names.add(name)
        train_signatures.add(hashlib.sha256(image.tobytes()).hexdigest())
        x, y = features(image).reshape(-1, 5), mask.ravel()
        indices = np.concatenate([rng.choice(np.flatnonzero(y == c), min(600, np.sum(y == c)), replace=False) for c in np.unique(y) if c != 255])
        xs.append(x[indices]); ys.append(y[indices])
    model = RandomForestClassifier(n_estimators=40, max_depth=14, n_jobs=2, random_state=42, class_weight='balanced')
    model.fit(np.concatenate(xs), np.concatenate(ys))
    matrix = np.zeros((6, 6), dtype=np.int64)
    count = 0
    for name, image, mask in pairs(data, 'val'):
        if hashlib.sha256(image.tobytes()).hexdigest() in train_signatures:
            raise ValueError('Identical drawing appears in training and validation.')
        pred = model.predict(features(image).reshape(-1, 5))
        valid = mask.ravel()!=255
        matrix += confusion_matrix(mask.ravel()[valid], pred[valid], labels=list(range(6)))
        count += 1
    denom = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    ious = {name: float(matrix[i,i]/denom[i]) if denom[i] else None for i, name in enumerate(CLASSES)}
    report = {'source': source, 'train_drawings': len(xs), 'validation_drawings': count, 'class_iou': ious, 'mean_iou_present_classes': float(np.mean([v for v in ious.values() if v is not None])), 'trained_classes': [CLASSES[int(i)] for i in model.classes_], 'confusion_matrix': matrix.tolist()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'source': source, 'classes': CLASSES, 'metrics': report}, output)
    output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True); parser.add_argument('--output', default='models/segmenter.joblib')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic: synthetic(args.data)
    print(json.dumps(train(args.data, args.output, 'synthetic' if args.synthetic else 'annotated'), indent=2))
