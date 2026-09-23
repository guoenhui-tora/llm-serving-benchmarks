#!/usr/bin/env python3
"""Restore this archived study into a NEW local workspace; never start a model."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'bench').is_file())
sys.path.insert(0,str(REPO/'src'))
from serving_bench.config import resolve
from serving_bench.executors.docker import cache_directory,preflight
from serving_bench.common import write_json,save_command

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--source-data',type=Path,required=True)
    p.add_argument('--cache-source',type=Path)
    p.add_argument('--offline-only',action='store_true',help='Derive data and validate/plan; do not copy caches or preflight Docker')
    a=p.parse_args();started=time.time();w=a.workspace.resolve()
    if w.parent!=REPO/'experiments':p.error('workspace must be a NEW direct child of repository experiments/')
    if w.exists():p.error('refusing to overwrite an existing workspace')
    if not a.offline_only and a.cache_source is None:p.error('--cache-source is required for runtime preparation')
    raw=a.source_data.read_bytes();assert hashlib.sha256(raw).hexdigest()=='025246b18d8ba7e1bc4d98c1a20b32b570860bb06a0baf0cea4ae97403d982e7'
    w.mkdir();shutil.copytree(HERE/'configs',w/'configs');shutil.copytree(HERE/'scripts',w/'scripts')
    for folder in ['data','evidence','reports','results']:(w/folder).mkdir()
    shutil.copy2(HERE/'analysis.py',w/'reports/analysis.py')
    rows=[json.loads(line) for line in raw.splitlines()]
    for row in rows:row['output_tokens']=1
    derived=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    dest=w/'data/govreport-16k-prefix-1-v1.jsonl';dest.write_text(derived)
    workload=yaml.safe_load((w/'configs/workloads/prefill-16k-1.yaml').read_text());assert hashlib.sha256(dest.read_bytes()).hexdigest()==workload['dataset']['sha256']
    patch_root=REPO/'projects/dsv4-rtx6000d/patches';caches=[]
    for mode in ['on','off']:
        target=w/f'configs/targets/prefill-{mode}.yaml';t=yaml.safe_load(target.read_text());t['cache_root']=str(w/'cache'/mode);target.write_text(yaml.safe_dump(t,sort_keys=False))
        campaign=w/f'configs/campaigns/prefill-{mode}.yaml';plan=resolve(campaign);write_json(w/f'evidence/plan-{mode}.json',plan);c=plan['cases'][0]
        for action in ['validate','plan']:
            result=subprocess.run([str(REPO/'bench'),action,str(campaign)],capture_output=True,text=True,check=True)
            (w/f'evidence/{action}-{mode}.txt').write_text(result.stdout)
        if a.offline_only:continue
        cache=cache_directory(c,c['runtime']['image_id']);cache.parent.mkdir(parents=True,exist_ok=True)
        # A read-only source mount can read root-owned cache files without changing them.
        code='import shutil,os; from pathlib import Path; shutil.copytree("/source","/destination/cache"); [os.chown(p,'+str(os.getuid())+','+str(os.getgid())+') for p in [Path("/destination/cache"),*Path("/destination/cache").rglob("*")]]'
        stage=cache.parent/(cache.name+'-staging');stage.mkdir()
        subprocess.run(['docker','run','--rm','--pull','never','--network','none','--user','0','--label','io.serving-bench.run='+w.name+'-prepare','-v',str(a.cache_source.resolve())+':/source:ro','-v',str(stage)+':/destination:rw','--entrypoint','python3',c['runtime']['image_id'],'-c',code],check=True)
        (stage/'cache').rename(cache)
        for folder,source in [('dspark-native-mxfp4','dspark-native-mxfp4'),('dsv4-input-warmup','input-kernel-warmup'),('dsv4-mhc-warmup','mhc-startup-warmup-extended/dspark')]:
            (cache/folder).rename(cache/('inherited-'+folder));(cache/folder).mkdir()
            for f in (patch_root/source).iterdir():
                if f.suffix in ('.py','.json'):shutil.copy2(f,cache/folder/f.name)
        assert (cache/'pd-logging.json').is_file(),'Historical logging dependency is missing'
        caches.append({'mode':mode,'source':str(a.cache_source.resolve()),'candidate':str(cache)})
        write_json(w/f'evidence/preflight-{mode}.json',preflight(c))
    write_json(w/'evidence/candidate-caches.json',caches)
    write_json(w/'evidence/authorized-budget.json',{'resumed_at_unix':started,'deadline_unix':started+14400,'startup_seconds':2700,'cell_seconds':1200,'performance_retries':0})
    if not a.offline_only:
        files=list((REPO/'src').rglob('*.py'))+list((w/'scripts').glob('*.py'))+list((w/'configs').rglob('*.yaml'))
        for item in caches:
            cache=Path(item['candidate'])
            files += [f for folder in ['dspark-native-mxfp4','dsv4-input-warmup','dsv4-mhc-warmup'] for f in (cache/folder).iterdir() if f.suffix in ('.py','.json')]
        write_json(w/'evidence/frozen.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    print(json.dumps({'workspace':str(w),'status':'OFFLINE_VALIDATED' if a.offline_only else 'PREPARED_NOT_STARTED'}))
if __name__=='__main__':main()
