"""Bounded inference-only comparison, reusing the completed trial environment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time

from backend.voice_research.expanded_comparison_cloud import read_comparison_archive,comparison_sha

ROOT=Path('/content/anyanwu-conversational-voice-comparison-v1')
WARM=Path('/content/anyanwu-conversational-style-lora-v1')


def restore_conversational_comparison(plan, *, root=ROOT, warm=WARM):
    root=Path(root);warm=Path(warm);assets=root/'assets';assets.mkdir(parents=True,exist_ok=True)
    prompt=plan['prompt_id'];required=['plan.json','training-result.json','adapter.safetensors']
    previous=read_comparison_archive(plan['archives']['previous'],required+['data-result.json','speech-targets.json',f'data/{prompt}.wav',f'data/{prompt}.mp3'])
    candidate=read_comparison_archive(plan['archives']['candidate'],required)
    old_plan=json.loads(previous['plan.json']);new_plan=json.loads(candidate['plan.json'])
    old_result=json.loads(previous['training-result.json']);new_result=json.loads(candidate['training-result.json'])
    reserved={t['id'] for t in plan['texts']}
    if len(reserved)!=4 or reserved&({c['id'] for c in old_plan['clips']}|set(new_plan['schedule'])):raise RuntimeError('Reserved texts overlap training/validation')
    if old_result['optimizer_updates']!=60 or new_result['optimizer_updates']!=30 or new_result['cumulative_lora_updates']!=90:raise RuntimeError('Unexpected source trial update counts')
    if new_plan['starting_adapter_sha256']!=plan['adapter_hashes']['base'] or not new_result['baseline_unchanged'] or not new_result['adapter_reloaded']:raise RuntimeError('Candidate starting checkpoint/integrity differs')
    for variant,data in (('base',previous),('adapter',candidate)):
        if hashlib.sha256(data['adapter.safetensors']).hexdigest()!=plan['adapter_hashes'][variant]:raise RuntimeError('Comparison adapter integrity failure')
    source=next(c for c in json.loads(previous['data-result.json'])['clips'] if c['id']==prompt)
    target=next(c for c in json.loads(previous['speech-targets.json']) if c['id']==prompt)
    if (source['speaker_id'],source['gender'],source['adapter_split'])!=('5','Female','train'):raise RuntimeError('Fixed female reference differs')
    for suffix,key in (('.wav','audio_sha256'),('.mp3','mp3_sha256')):
        if hashlib.sha256(previous[f'data/{prompt}{suffix}']).hexdigest()!=source[key]:raise RuntimeError('Fixed reference source changed')
    outputs={'plan.json':previous['plan.json'],'previous-adapter.safetensors':previous['adapter.safetensors'],
             'adapter.safetensors':candidate['adapter.safetensors'],'speech-targets.json':json.dumps([target],ensure_ascii=False).encode(),
             'data-result.json':json.dumps({'clips':[{**source,'cloud_audio_path':str(assets/'data'/(prompt+'.wav'))}]},ensure_ascii=False).encode(),
             f'data/{prompt}.wav':previous[f'data/{prompt}.wav'],f'data/{prompt}.mp3':previous[f'data/{prompt}.mp3']}
    for name,raw in outputs.items():
        path=assets/name;path.parent.mkdir(exist_ok=True,parents=True)
        if path.exists() and path.read_bytes()!=raw:raise RuntimeError('Different comparison input preserved')
        if not path.exists():path.write_bytes(raw)
    cache=assets/'cache/hf/hub';cache.parent.mkdir(parents=True,exist_ok=True)
    if (warm/'cache/hf/hub').is_dir() and not cache.exists():cache.symlink_to(warm/'cache/hf/hub',target_is_directory=True)
    return {'phase':'conversational_comparison_inputs_verified','adapter_hashes':plan['adapter_hashes'],'prompt_id':prompt,'warm_cache':cache.is_symlink(),'optimizer_updates_this_stage':0}


def run_conversational_comparison(plan,worker,generator,runtime,adapter_source,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop audio/models/install')
    nvidia=shutil.which('nvidia-smi')
    if not nvidia:raise RuntimeError('Existing hosted GPU required; no CPU fallback or automatic allocation')
    gpu=subprocess.run([nvidia,'--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if gpu.returncode or len(gpu.stdout.strip().splitlines())!=1 or int(gpu.stdout.strip().rsplit(',',1)[1])<14000:raise RuntimeError('One hosted GPU>=14000MiB required')
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Connect existing research Drive before inference setup')
    ROOT.mkdir(exist_ok=True);stored=ROOT/'plan.json';result_path=ROOT/'generation-result.json'
    if stored.exists() and json.loads(stored.read_text())!=plan:raise RuntimeError('Different registered comparison preserved')
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result.get('completed_wavs')!=8 or result.get('optimizer_updates_this_stage')!=0:raise RuntimeError('Invalid completed comparison')
        for clip in result['clips']:
            if comparison_sha(ROOT/clip['audio_filename'])!=clip['audio_sha256']:raise RuntimeError('Existing comparison audio changed')
        return result
    if (ROOT/'attempt.json').exists():raise RuntimeError('Prior attempt preserved; inspect before a declared repair')
    if shutil.disk_usage('/content').free<30*1024**3 or sum(f['size'] for f in plan['model_files'])>4*1024**3:raise RuntimeError('Comparison cloud disk/model cap')
    stored.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');(ROOT/'attempt.json').write_text('{"phase":"started","optimizer_updates_this_stage":0}\n')
    assets=ROOT/'assets';started=time.monotonic();initial_free=shutil.disk_usage('/content').free;peak=0;phase='restoring';latest={}
    env={k:v for k,v in os.environ.items() if k not in {'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}}
    env.update(ANYANWU_COMPARE_VOICE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               HF_HOME=str(assets/'cache/hf'),XDG_CACHE_HOME=str(ROOT/'cache'),UV_CACHE_DIR=str(ROOT/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(ROOT/'python'),
               PIP_CONFIG_FILE='/dev/null',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
               TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def bounded(command,name,offline=False):
        nonlocal peak,phase,latest
        phase=name;log=ROOT/(name+'.log');publish({'phase':'conversational_comparison_'+name,'optimizer_updates_this_stage':0});stage_start=time.monotonic()
        child={**env,**({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'} if offline else {})}
        with log.open('w') as out:
            process=subprocess.Popen(command,env=child,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);last=None
            try:
                while process.poll() is None:
                    free=shutil.disk_usage('/content').free;statm=Path(f'/proc/{process.pid}/statm');rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0;peak=max(peak,rss)
                    if time.monotonic()-started>1800 or (name=='generation' and time.monotonic()-stage_start>900) or initial_free-free>20*1024**3 or free<5*1024**3 or rss>9*1024**3 or log.stat().st_size>512*1024:raise RuntimeError('Comparison time/disk/RSS/log cap')
                    progress=ROOT/'generation-progress.json'
                    if name=='generation' and progress.exists():
                        current=progress.read_text()
                        if current!=last:last=current;latest=json.loads(current);publish({**{k:v for k,v in latest.items() if k!='clips'},'phase':'conversational_comparison_'+latest['phase'],'completed_wavs':len(latest['clips'])})
                    time.sleep(2)
                if process.returncode:raise RuntimeError(f'{name} failed ({process.returncode}); inspect bounded cloud log')
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        publish(restore_conversational_comparison(plan))
        for name,source in (('comparison_worker.py',worker),('flash_generation.py',generator),('cv3_runtime.py',runtime)):(ROOT/name).write_text(source)
        (assets/'flash_adapter.py').write_text(adapter_source)
        python=WARM/'venv/bin/python';warm=python.is_file()
        if not warm:
            bounded([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--only-binary=:all:','--target',str(ROOT/'bootstrap'),'uv==0.9.13','--index-url','https://pypi.org/simple'],'bootstrap')
            bounded([str(ROOT/'bootstrap/bin/uv'),'venv','--python','3.11.14','--managed-python','--seed',str(assets/'venv')],'python311');python=assets/'venv/bin/python'
        bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','torch==2.8.0','torchaudio==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'],'inference_torch')
        constraints=ROOT/'constraints.txt';constraints.write_text('torch==2.8.0\ntransformers==4.51.3\nhuggingface_hub==0.36.0\nnumpy==1.26.4\nonnxruntime==1.23.2\nsafetensors==0.5.3\n')
        bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','-c',str(constraints),
                 'transformers==4.51.3','huggingface_hub==0.36.0','numpy==1.26.4','onnxruntime==1.23.2','safetensors==0.5.3',
                 'x-transformers==2.11.24','einops==0.8.1','einx==0.3.0','omegaconf==2.3.0','scipy==1.11.4','librosa==0.10.2','--index-url','https://pypi.org/simple'],'inference_dependencies')
        download='''from pathlib import Path
import hashlib,json,shutil
from huggingface_hub import hf_hub_download
root=Path('/content/anyanwu-conversational-voice-comparison-v1');assets=root/'assets';plan=json.loads((root/'plan.json').read_text());acoustic={}
for item in plan['model_files']:
 path=Path(hf_hub_download(item['repo'],item['name'],revision=item['revision'],token=False,cache_dir=str(assets/'cache/hf/hub')))
 if path.stat().st_size!=item['size']:raise RuntimeError('Pinned inference file size changed')
 digest=hashlib.sha256() if item.get('sha256') else hashlib.sha1()
 if not item.get('sha256'):digest.update(('blob '+str(path.stat().st_size)+'\\0').encode())
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(1024**2),b''):digest.update(block)
 if digest.hexdigest()!=item.get('sha256',item.get('git_blob_sha1')):raise RuntimeError('Pinned inference asset hash mismatch')
 if item['name'] in ('flow.pt','hift.pt','campplus.onnx'):acoustic[item['name']]=str(path)
 if item['name'].split('/')[-1] in ('vocab.json','merges.txt','tokenizer_config.json'):
  target=assets/'text-tokenizer'/item['name'].split('/')[-1];target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
(root/'acoustic-files.json').write_text(json.dumps(acoustic,indent=2)+'\\n')
import cv3_runtime
print('Pinned acoustic runtime/assets verified')
'''
        compile(download,'inference-download','exec');(ROOT/'download_models.py').write_text(download)
        bounded([str(python),str(ROOT/'download_models.py')],'inference_assets')
        (ROOT/'setup-result.json').write_text(json.dumps({'phase':'inference_ready','warm_training_environment_reused':warm,'selected_model_bytes':sum(f['size'] for f in plan['model_files']),'optimizer_updates_this_stage':0})+'\n')
        bounded([str(python),str(ROOT/'comparison_worker.py')],'generation',True)
        result=json.loads(result_path.read_text())
        if result['completed_wavs']!=8 or result['optimizer_updates_this_stage']!=0 or not result['source_checkpoints_unchanged']:raise RuntimeError('Required eight inference-only outputs missing')
        result.update(warm_training_environment_reused=warm,peak_worker_rss_bytes=peak,total_setup_generation_seconds=round(time.monotonic()-started,2),cloud_disk_increase_bytes=initial_free-shutil.disk_usage('/content').free)
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(ROOT/'attempt.json').write_text('{"phase":"complete","optimizer_updates_this_stage":0,"completed_wavs":8}\n')
        return result
    except Exception as error:
        report={'phase':'conversational_comparison_stopped','failed_phase':phase,'error_type':type(error).__name__,'error':str(error)[:450],'optimizer_updates_this_stage':0}
        if (ROOT/'generation-progress.json').exists():report['worker_progress']=json.loads((ROOT/'generation-progress.json').read_text())
        (ROOT/'attempt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report
