"""Fine-tune the ArchVision detector on a reviewed YOLO-seg dataset.

The dataset YAML must be authored/reviewed locally. YOLO dataset configuration
can include download actions; this wrapper rejects those before training.
"""
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

parser=argparse.ArgumentParser()
parser.add_argument('--data',required=True);parser.add_argument('--epochs',type=int,default=60)
parser.add_argument('--batch',type=int,default=4);parser.add_argument('--device',default='cpu')
args=parser.parse_args()
path=Path(args.data).resolve();config=yaml.safe_load(path.read_text(encoding='utf-8'))
if not isinstance(config,dict) or any(k in config for k in ('download',)):
    raise ValueError('Use a local, reviewed dataset YAML without executable download actions.')
for split in ('train','val'):
    if split not in config:raise ValueError('Both train and val splits are required.')
# Dataset generation and project-level splitting must precede this supervised run.
model=YOLO('models/archvision-rooms.pt')
model.train(data=str(path),epochs=args.epochs,batch=args.batch,device=args.device,imgsz=1024,workers=2,project='models/training',name='reviewed-rooms',patience=12,seed=42,exist_ok=False)
