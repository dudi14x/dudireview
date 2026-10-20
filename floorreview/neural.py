"""Compact U-Net: supervised image-to-mask segmentation; CPU or CUDA training."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from .training import pairs, synthetic
from .vision import CLASSES


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())

class PlanUNet(nn.Module):
    def __init__(self):
        super().__init__();self.enc1=block(3,16);self.enc2=block(16,32);self.bottleneck=block(32,64);self.dec2=block(96,32);self.dec1=block(48,16);self.out=nn.Conv2d(16,6,1)
    def forward(self,x):
        a=self.enc1(x);b=self.enc2(nn.functional.max_pool2d(a,2));c=self.bottleneck(nn.functional.max_pool2d(b,2));d=self.dec2(torch.cat([nn.functional.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));return self.out(self.dec1(torch.cat([nn.functional.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))


def tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255


def load_model(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=PlanUNet();model.load_state_dict(state['state_dict']);model.eval()
    return model,state


def infer(image,path):
    torch.set_num_threads(2)
    model,state=load_model(path)
    # Overlapping tiles preserve thin lines at the analysis image resolution.
    h,w=image.shape[:2];votes=np.zeros((6,h,w),np.float32);counts=np.zeros((h,w),np.float32)
    with torch.inference_mode():
        for y in range(0,h,224):
            for x in range(0,w,224):
                crop=image[y:y+256,x:x+256];ch,cw=crop.shape[:2]
                padded=cv2.copyMakeBorder(crop,0,256-ch,0,256-cw,cv2.BORDER_REPLICATE)
                out=model(tensor(padded)[None]).softmax(1)[0,:,:ch,:cw].numpy()
                votes[:,y:y+ch,x:x+cw]+=out;counts[y:y+ch,x:x+cw]+=1
    return (votes/counts[None]).argmax(0).astype(np.uint8),state['source']


def read_sample(path):
    path=Path(path)
    im=cv2.imread(str(path));mask=cv2.imread(str(path.parent.parent/'masks'/path.name),0)
    if im is None or mask is None or im.shape[:2]!=mask.shape:raise ValueError(f'Missing or mismatched mask: {path.name}')
    if np.any((mask>5)&(mask!=255)):raise ValueError('Mask IDs must be 0–5 or 255.')
    return path.name,cv2.cvtColor(im,cv2.COLOR_BGR2RGB),mask


def train_unet(data,output,epochs=12,source='annotated'):
    torch.manual_seed(42);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    training=sorted((Path(data)/'train/images').glob('*.png'));validation=sorted((Path(data)/'val/images').glob('*.png'))
    if not training or not validation:raise ValueError('Both train and val PNG image directories are required.')
    signatures={hashlib.sha256(read_sample(path)[1].tobytes()).hexdigest() for path in training}
    if any(hashlib.sha256(read_sample(path)[1].tobytes()).hexdigest() in signatures for path in validation):raise ValueError('Drawing leakage between training and validation.')
    model=PlanUNet().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    present=np.zeros(6,dtype=np.int64)
    for path in training:
        _,_,mask=read_sample(path)
        present+=np.bincount(mask[mask!=255],minlength=6)
    weights=torch.tensor(np.where(present>0,1/np.sqrt(np.maximum(present,1)),0),dtype=torch.float32,device=device)
    weights/=weights[weights>0].mean()
    criterion=nn.CrossEntropyLoss(weight=weights,ignore_index=255)
    history=[];best=-1
    rng=np.random.default_rng(42)
    for epoch in range(epochs):
        model.train();losses=[]
        for i in rng.permutation(len(training)):
            _,im,mask=read_sample(training[i])
            # Random crops retain original line width; flips augment orientation.
            h,w=mask.shape;y=int(rng.integers(max(1,h-255)));x=int(rng.integers(max(1,w-255)))
            im=im[y:y+256,x:x+256];mask=mask[y:y+256,x:x+256]
            im=cv2.resize(im,(256,256));mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
            if rng.random()<.5:im=im[:,::-1].copy();mask=mask[:,::-1].copy()
            if not np.any(mask!=255):continue
            optimizer.zero_grad();prediction=model(tensor(im)[None].to(device));loss=criterion(prediction,torch.from_numpy(mask.astype(np.int64))[None].to(device));loss.backward();optimizer.step();losses.append(float(loss.detach()))
        if not losses:raise ValueError('No labeled training pixels.')
        model.eval();matrix=np.zeros((6,6),np.int64)
        with torch.inference_mode():
            for path in validation:
                _,im,mask=read_sample(path)
                # Validation at fixed resolution is explicit in the report.
                im=cv2.resize(im,(256,256));truth=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
                pred=model(tensor(im)[None].to(device)).argmax(1)[0].cpu().numpy();valid=truth!=255
                matrix+=np.bincount((truth[valid].astype(int)*6+pred[valid]).ravel(),minlength=36).reshape(6,6)
        den=matrix.sum(0)+matrix.sum(1)-matrix.diagonal();iou={name:float(matrix[i,i]/den[i]) if den[i] else None for i,name in enumerate(CLASSES)}
        mean=float(np.mean([v for v in iou.values() if v is not None]));row={'epoch':epoch+1,'loss':float(np.mean(losses)),'mean_iou':mean};history.append(row);print(json.dumps(row),flush=True)
        if mean>best:
            best=mean;output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
            report={'source':source,'architecture':'PlanUNet-16','train_drawings':len(training),'validation_drawings':len(validation),'validation_resolution':[256,256],'class_iou':iou,'mean_iou':mean,'trained_classes':[CLASSES[i] for i in range(6) if present[i]],'best_epoch':epoch+1,'device':device,'confusion_matrix':matrix.tolist()}
            torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'source':source,'metrics':report},output)
            output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    output.with_suffix('.history.json').write_text(json.dumps(history,indent=2), encoding='utf-8')
    return json.loads(output.with_suffix('.metrics.json').read_text(encoding='utf-8'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',default='models/plan-unet.pt');parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--synthetic',action='store_true');args=parser.parse_args()
    if args.synthetic:synthetic(args.data,n=96)
    print(json.dumps(train_unet(args.data,args.output,args.epochs,'synthetic' if args.synthetic else 'annotated'),indent=2))
