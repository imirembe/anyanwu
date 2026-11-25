"""Register six fresh published texts and build a matched-volume comparison."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/'.build/VoiceResearch'
PRIOR=LAB/'auditions/conversational-reference-conditioning-v1'
NAME='conversational-reference-validation-v2'
FOLDER=LAB/'auditions'/NAME


def once(source,before,after):
    if source.count(before)!=1:raise RuntimeError('Template changed: '+before[:80])
    return source.replace(before,after)


def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True)
    old=json.loads((PRIOR/'plan.json').read_text());feedback=json.loads((PRIOR/'submitted-feedback-summary.json').read_text())
    if feedback['complete_reviews']!=2 or feedback['preference_counts']['adapter']!=2 or not feedback['snapshot_verified']:
        raise RuntimeError('Verified prior conversational-reference preference required')
    source_path=LAB/'datasets/waxal-female5-expansion-v1/source-rows-517-616.json'
    source_raw=source_path.read_bytes();page=json.loads(source_raw)
    selected=['ibo_1705','ibo_1732','ibo_1753','ibo_1799','ibo_1701','ibo_1729']
    used=set(old['previously_reviewed_or_trained_ids'])|{t['id'] for t in old['texts']};used_text=set()
    def scan(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('id','text_id') and isinstance(item,str):used.add(item)
                if key in ('text','requested_text') and isinstance(item,str):used_text.add(hashlib.sha256(item.encode()).hexdigest())
                scan(item)
        elif isinstance(value,list):
            for item in value:scan(item)
    for path in LAB.rglob('plan.json'):
        if FOLDER in path.parents:continue
        scan(json.loads(path.read_text()))
    texts=[]
    for key in selected:
        entry=next(e for e in page['rows'] if e['row']['id']==key);text=entry['row']['text']
        digest=hashlib.sha256(text.encode()).hexdigest()
        if key in used or digest in used_text or entry.get('truncated_cells') or not 10<len(text)<260:
            raise RuntimeError('Fresh whole published text required: '+key)
        texts.append({'id':key,'text':text,'text_sha256':digest,'row_idx':entry['row_idx'],'source':'google/WaxalNLP',
            'config':'ibo_tts/train','revision':'5f4d8ca24f2b9d168b2ee545f1febaaff4b40580',
            'role':'Published voice-test text, unchanged; not a live forecast, factual assistant answer or independently verified language gold.'})
    plan={**old,'id':'anyanwu-'+NAME,'status':'registered_not_run',
        'reason':'Prior user review preferred conversational ref03 in both pairs, rhythm4/5 versus2/5; test more fresh sentences while controlling listening volume.',
        'hypothesis':'The conversational reference improves perceived connected speech beyond the two development sentences, at comparable listening loudness.',
        'texts':texts,'outputs':12,'previously_reviewed_or_trained_ids':sorted(used),
        'prior_feedback_snapshot_sha256':feedback['snapshot_sha256'],
        'source_page':{'path':str(source_path.relative_to(ROOT)),'sha256':hashlib.sha256(source_raw).hexdigest(),
            'note':'Cached official viewer metadata; Hub revision was pinned in acquisition receipts, viewer row revision is not independently guaranteed.'},
        'listening_loudness':{'method':'One constant gain per clip, measured integrated LUFS; no compression, pitch/time edits or silence removal.',
            'requested_integrated_lufs':-20.0,'true_peak_headroom_db':-1.5,'maximum_gain_db':24.0,'maximum_lufs_error':0.5,
            'raw_wavs_preserved':True,'gain_selection':'Common target lowered when necessary to preserve headroom for every clip.'},
        'scope':'Private noncommercial research, current Colab/Drive only. Twelve one-take raw outputs plus separate matched-volume listening copies; same v1, zero training.',
        'limits':{**old['limits'],'archive_bytes':32*1024**2,'listening_copy_minutes':5,'listening_copy_growth_mib':64},
        'limitations':old['limitations']+['Six new published reading texts, one reviewer, explicitly labelled reference conditions; not a blinded or population benchmark.',
            'Both conditions retain the previous all-Igbo text-language setting; numbers and English names/units may reveal front-end limitations.',
            'Only constant gain is applied to separately hashed listening copies. Raw audio and all previous ratings remain unchanged.']}
    path=FOLDER/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Different registered validation preserved')
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    core=(ROOT/'backend/voice_research/reference_conditioning.py').read_text()
    core=once(core,'len(reserved) != 2','len(reserved) != 6').replace('reference_conditioning_inputs_verified','reference_validation_inputs_verified')
    core=core.replace('    from backend.voice_research.expanded_comparison_cloud import read_comparison_archive\n','')
    worker=(PRIOR/'reference_worker.py').read_text().replace('anyanwu-conversational-reference-conditioning-v1','anyanwu-'+NAME)
    worker=worker.replace('Generate four one-take clips','Generate twelve one-take clips').replace('len(tokens)!=4','len(tokens)!=12').replace('completed_wavs=4','completed_wavs=12').replace('four_generated_reference_clips_ready','twelve_generated_reference_clips_ready')
    cloud=(PRIOR/'reference_cloud.py').read_text().replace('anyanwu-conversational-reference-conditioning-v1','anyanwu-'+NAME)
    cloud=cloud.replace('run_reference_comparison','run_reference_validation').replace('reference_conditioning_','reference_validation_').replace('!=4','!=12').replace('completed_wavs":4','completed_wavs":12').replace('Required four','Required twelve')
    archive=(PRIOR/'reference_archive.py').read_text().replace('anyanwu-conversational-reference-conditioning-v1','anyanwu-'+NAME).replace('conversational-reference-conditioning-v1',NAME)
    archive=archive.replace('backup_reference_comparison','backup_reference_validation').replace('Archive four','Archive twelve').replace('!=4','!=12').replace("'audio_clips':4","'audio_clips':12").replace('Four completed','Twelve completed').replace('16*1024**2','32*1024**2')
    archive=once(archive,"'reference-preparation.json']","'reference-preparation.json','listening-result.json','generation-result.raw.json','reference_loudness.py']")
    archive=once(archive,"names.extend([clip['filename'],clip['audio_filename']])","names.extend([clip['filename'],clip['audio_filename'],clip['listening_audio_filename']])")
    archive=once(archive,"raise RuntimeError('Audio integrity failure')", "raise RuntimeError('Audio integrity failure')\n        if hashlib.sha256((root/clip['listening_audio_filename']).read_bytes()).hexdigest()!=clip['listening_audio_sha256']:raise RuntimeError('Listening audio integrity failure')")
    review=(PRIOR/'combined_review_panel.py').read_text().replace('anyanwu-conversational-reference-conditioning-v1','anyanwu-'+NAME).replace('conversational-reference-conditioning-v1',NAME)
    review=review.replace('reference_conditioning','reference_validation').replace('anyanwu:reference-conditioning-drafts:v1','anyanwu:reference-validation-drafts:v2')
    review=review.replace('validate_reference_feedback','validate_reference_validation_feedback').replace('restore_reference_listening','restore_reference_validation_listening').replace('show_reference_comparison','show_reference_validation')
    review=review.replace("len(plan['texts'])!=2","len(plan['texts'])!=6").replace("len(result['clips'])!=4","len(result['clips'])!=12").replace("result['completed_wavs']!=4","result['completed_wavs']!=12").replace('len(allowed)!=2','len(allowed)!=6').replace("'pairs':2","'pairs':6").replace('r.pairs===2','r.pairs===6').replace("'audio_players':4","'audio_players':12").replace('Expected two complete comparison pairs','Expected six complete comparison pairs')
    review=review.replace('16*1024**2','32*1024**2').replace("len(manifest['files'])>30","len(manifest['files'])>80")
    review=review.replace("matches[0]['audio_sha256']","matches[0]['listening_audio_sha256']")
    review=review.replace("root/clip['audio_filename']","root/clip['listening_audio_filename']").replace("clip['audio_sha256']","clip['listening_audio_sha256']")
    review=review.replace('Listen and give feedback','Volume-matched voice comparison').replace('Use Next for pair two.','Use Next for the next pair.')
    review=review.replace('Same model, different references. The conversational clips are quieter; adjust playback volume comfortably.', 'Same v1 model, different references. Listening volume is matched. These are published voice-test sentences, not live assistant answers.')
    review=review.replace('Two unused published texts','Six unused published texts')
    runtime_path=LAB/'auditions/waxal-female5-adapter-comparison-v1'
    runtime=(runtime_path/'cv3_inference_runtime.py').read_text()
    if hashlib.sha256(runtime.encode()).hexdigest()!=plan['runtime_sha256']:raise RuntimeError('Pinned decoder runtime changed')
    helper=(ROOT/'backend/voice_research/expanded_comparison_cloud.py').read_text().split('\ndef restore_expanded_comparison',1)[0]
    generator=(ROOT/'backend/voice_research/flash_generation.py').read_text();adapter=(ROOT/'backend/voice_research/flash_adapter.py').read_text()
    loudness=(ROOT/'backend/voice_research/reference_loudness.py').read_text()
    extras={'runtime-source-receipt.json':(runtime_path/'runtime-source-receipt.json').read_text(),'COSYVOICE-LICENSE':(runtime_path/'reference/LICENSE').read_text(),
            'MATCHA-LICENSE':(runtime_path/'reference/MATCHA-LICENSE').read_text(),'comparison_review.py':review}
    cell='# Anyanwu six-pair reference validation — inference only, matched listening volume\n'+helper+'\n'+core+'\n'+cloud+'\n'+loudness+'\n'+archive+'\n'+review
    for name,value in [('PLAN',plan),('WORKER',worker),('GENERATOR',generator),('RUNTIME',runtime),('ADAPTER',adapter),('LOUDNESS_SOURCE',loudness),('EXTRAS',extras)]:cell+='\n'+name+'='+repr(value)+'\n'
    cell+='''
from IPython.display import Javascript,display,Markdown
def publish(value):
 if not str(value.get('phase','')).startswith('reference_validation_'):value={**value,'phase':'reference_validation_'+value.get('phase','progress')}
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('## Six-pair voice validation\\nSix unused published texts, same v1 model, two references. Twelve one-take clips; matched-volume listening copies. No training.'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop audio or models')
 if not shutil.which('nvidia-smi'):raise RuntimeError('Existing hosted GPU required; no automatic allocation or CPU synthesis')
 if not Path('/content/drive/MyDrive').is_dir():
  publish({'phase':'reference_validation_waiting_for_drive','optimizer_updates_this_stage':0})
  from google.colab import drive
  drive.mount('/content/drive',timeout_ms=300000)
 report=run_reference_validation(PLAN,WORKER,GENERATOR,RUNTIME,ADAPTER,publish);publish(report)
 if report.get('completed_wavs')==12 and report.get('source_checkpoints_unchanged'):
  (ROOT/'reference_loudness.py').write_text(LOUDNESS_SOURCE)
  matched=normalize_listening_copies(ROOT,PLAN,publish);publish(matched)
  publish({**json.loads((ROOT/'generation-result.json').read_text()),'phase':'reference_validation_matched_generation_metadata'})
  receipt=backup_reference_validation(EXTRAS);publish(receipt)
  from IPython.display import clear_output
  clear_output(wait=True)
  show_reference_validation(receipt)
  print('Twelve raw clips and twelve matched-volume copies verified in Drive. Review one pair at a time; no training starts automatically.')
except Exception as error:
 publish({'phase':'reference_validation_cell_stopped','error_type':type(error).__name__,'error':str(error)[:400],'optimizer_updates_this_stage':0})
 raise
'''
    sources={'validation_worker.py':worker,'validation_cloud.py':cloud,'validation_review.py':review,'validation_archive.py':archive,'validation-cell.py':cell}
    for name,code in sources.items():compile(code,name,'exec');(FOLDER/name).write_text(code)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · six-pair matched-volume voice validation\n','Same existing Colab notebook/free hosted GPU and research Drive. Twelve one-take raw clips plus separate constant-gain listening copies. No training or laptop media/models.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
    (ROOT/'Voice Research/Anyanwu-Reference-Validation-v2.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    receipt={'phase':'reference_validation_prepared','outputs':12,'pairs':6,'new_training':False,'texts':selected,
        'source_sha256':{name:hashlib.sha256(code.encode()).hexdigest() for name,code in sources.items()},
        'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_model_bytes':sum(f['size'] for f in plan['model_files'])}
    (FOLDER/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
