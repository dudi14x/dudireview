"""Retrieve the pinned public checkpoint over verified HTTPS; Git is not required."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request

COMMIT='5b1b4a307e43c94164b7b859e7fb2f5e656c115f'
EXPECTED='639c5deab18bcde81104f958f991175a609d0ebe909970ee5265bc7370b23b17'
URL=f'https://raw.githubusercontent.com/Parth-Talsania/Archvision/{COMMIT}/models/best.pt'

def fetch(target):
    target=Path(target)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==EXPECTED:
        print('ArchVision checkpoint verified.')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as folder:
        temporary=Path(folder)/'checkpoint.pt'
        digest=hashlib.sha256()
        with urllib.request.urlopen(URL,timeout=120) as response, temporary.open('wb') as output:
            while block:=response.read(1024*1024):
                digest.update(block);output.write(block)
        if digest.hexdigest()!=EXPECTED:
            raise RuntimeError('Checkpoint checksum mismatch; the downloaded file was rejected.')
        temporary.replace(target)
    print('ArchVision checkpoint downloaded and verified.')

if __name__=='__main__':
    fetch(Path(__file__).resolve().parents[1]/'models/archvision-rooms.pt')
