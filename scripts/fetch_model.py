"""Fetch the exact reviewed upstream checkpoint, verifying its SHA-256."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

COMMIT='5b1b4a307e43c94164b7b859e7fb2f5e656c115f'
EXPECTED='639c5deab18bcde81104f958f991175a609d0ebe909970ee5265bc7370b23b17'
target=Path(__file__).resolve().parents[1]/'models/archvision-rooms.pt'
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==EXPECTED:
    print('ArchVision checkpoint verified.')
else:
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run(['git','init','-q',temporary],check=True)
        subprocess.run(['git','-C',temporary,'fetch','--depth','1','https://github.com/Parth-Talsania/Archvision.git',COMMIT],check=True)
        blob=subprocess.check_output(['git','-C',temporary,'show',COMMIT+':models/best.pt'])
        if hashlib.sha256(blob).hexdigest()!=EXPECTED:raise RuntimeError('Checkpoint checksum mismatch.')
        target.parent.mkdir(parents=True,exist_ok=True)
        staging=target.with_suffix('.tmp');staging.write_bytes(blob);staging.replace(target)
        print('ArchVision checkpoint fetched and verified.')
