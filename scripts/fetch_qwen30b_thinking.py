"""Fetch one pinned public checkpoint through the working local proxy."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

MODEL='Qwen/Qwen3-30B-A3B-Thinking-2507'
REVISION='144afc2f379b542fdd4e85a1fcd5e1f79112d95d'
FOLDER=Path(__file__).resolve().parents[1]/'runs/phase3/models/Qwen3-30B-A3B-Thinking-2507'


def session():
    result=requests.Session();result.trust_env=False
    result.proxies={'https':'http://127.0.0.1:41227','http':'http://127.0.0.1:41227'}
    return result


def fetch(entry):
    name=entry['rfilename'];target=FOLDER/name
    expected=entry.get('lfs',{}).get('sha256')
    def checksum(path):
        value=hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda:source.read(8*1024**2),b''):value.update(block)
        return value.hexdigest()
    if target.exists() and target.stat().st_size==entry['size'] and (not expected or checksum(target)==expected):
        print('already complete',name,flush=True);return
    temp=target.with_suffix(target.suffix+'.partial');value=hashlib.sha256();size=0
    if temp.exists():
        with temp.open('rb') as source:
            for block in iter(lambda:source.read(8*1024**2),b''):value.update(block);size+=len(block)
    assert size<entry['size']
    print('fetching',name,entry['size'],flush=True)
    with session().get('https://huggingface.co/'+MODEL+'/resolve/'+REVISION+'/'+name,
            headers={'Range':'bytes='+str(size)+'-'} if size else {},stream=True,timeout=(30,60)) as response:
        response.raise_for_status()
        if response.status_code==206:assert response.headers['Content-Range'].startswith('bytes '+str(size)+'-')
        elif size:size=0;value=hashlib.sha256()
        with temp.open('ab' if size else 'wb') as out:
            for block in response.iter_content(8*1024**2):
                out.write(block);value.update(block);size+=len(block)
    assert size==entry['size'],(name,size,entry['size'])
    if expected:assert value.hexdigest()==expected,name
    temp.replace(target);print('complete',name,size,flush=True)


if __name__=='__main__':
    FOLDER.mkdir(parents=True,exist_ok=True)
    response=session().get('https://huggingface.co/api/models/'+MODEL+'/revision/'+REVISION,params={'blobs':'true'},timeout=30)
    response.raise_for_status();metadata=response.json();assert metadata['sha']==REVISION
    (FOLDER/'hf_metadata.json').write_text(json.dumps(metadata,indent=2))
    names={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt',
        'model.safetensors.index.json','README.md','LICENSE'}
    entries=[e for e in metadata['siblings'] if e['rfilename'] in names or e['rfilename'].endswith('.safetensors')]
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,entries))
    print('COMPLETE pinned checkpoint; every LFS SHA256 checked',flush=True)
