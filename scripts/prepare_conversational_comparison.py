"""Prepare one-take testing of preserved v1 versus the conversational candidate."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def prepare():
    lab=ROOT/'.build/VoiceResearch';folder=lab/'auditions/conversational-voice-comparison-v1';folder.mkdir(parents=True,exist_ok=True)
    trained=json.loads((lab/'training/conversational-style-lora-v1/training-completion-summary.json').read_text())
    trial=json.loads((lab/'training/conversational-style-lora-v1/plan.json').read_text())
    prior=json.loads((lab/'auditions/waxal-female5-expanded-comparison-v2/plan.json').read_text())
    original=lab/'auditions/waxal-female5-adapter-comparison-v1'
    runtime=(original/'cv3_inference_runtime.py').read_text()
    if hashlib.sha256(runtime.encode()).hexdigest()!=prior['runtime_sha256']:raise RuntimeError('Pinned acoustic runtime changed')
    if trained['optimizer_updates']!=30 or not trained['drive_backup_verified'] or not trained['baseline_preserved']:raise RuntimeError('Verified completed trial required')
    texts=trial['comparison']['texts']
    if len(texts)!=4 or {t['id'] for t in texts}&set(trial['schedule']):raise RuntimeError('Reserved texts overlap training')
    plan={'id':'anyanwu-conversational-voice-comparison-v1','texts':texts,'variants':['base','adapter'],
          'variant_labels':{'base':'Previous voice · v1','adapter':'Conversational trial'},
          'adapter_hashes':{'base':trial['starting_adapter_sha256'],'adapter':trained['adapter_sha256']},
          'archives':{'previous':trial['baseline_archive'],'candidate':{'path':trained['drive_archive'],'bytes':trained['archive_bytes'],'sha256':trained['archive_sha256']}},
          'base_repo':prior['base_repo'],'base_revision':prior['base_revision'],'base_checkpoint_sha256':prior['base_checkpoint_sha256'],
          'acoustic_repo':prior['acoustic_repo'],'acoustic_revision':prior['acoustic_revision'],'model_files':prior['model_files'],
          'runtime_sha256':prior['runtime_sha256'],'prompt_id':'ibo_616','seed':17,'outputs':8,'max_speech_tokens':400,
          'optimizer_updates_this_stage':0,'policy':'Four new published texts, one whole-sentence take per adapter. Same reference, decoder, seed and sampler; no retakes or pitch editing.',
          'limits':prior['limits'],'attribution':trial['attribution'],
          'limitations':['Private research comparison; preserve all source licence conditions, no combined redistribution licence asserted.',
                         'Requested published text is not an independently verified transcript of generated speech.',
                         'Four texts are excluded from all adapter training/rated comparison sets; possible base-pretraining exposure remains.',
                         'LM source conditioning changes in the candidate while acoustic reference is fixed; no isolated causal claim about cadence alone.',
                         'Only actual user listening establishes preference; no automatic adoption, training, transcript approval or evaluation score.']}
    path=folder/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Different registered comparison preserved')
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    worker=(ROOT/'backend/voice_research/expanded_comparison_worker.py').read_text()
    changes={"ROOT=Path('/content/anyanwu-waxal-female5-expanded-comparison-v2')":"ROOT=Path('/content/anyanwu-conversational-voice-comparison-v1')",
             "'source_adapter_training_updates':{'base':60,'adapter':120}":"'source_adapter_training_updates':{'base':60,'adapter':90}"}
    for old,new in changes.items():
        if worker.count(old)!=1:raise RuntimeError('Comparison worker template changed')
        worker=worker.replace(old,new)
    helper=(ROOT/'backend/voice_research/expanded_comparison_cloud.py').read_text().split('\ndef restore_expanded_comparison',1)[0]
    cloud=(ROOT/'backend/voice_research/conversational_comparison_cloud.py').read_text().replace('from backend.voice_research.expanded_comparison_cloud import read_comparison_archive,comparison_sha\n','')
    archive=(ROOT/'backend/voice_research/expanded_comparison_archive.py').read_text()
    review=(ROOT/'backend/voice_research/expanded_comparison_review.py').read_text()
    for old,new in (('anyanwu-waxal-female5-expanded-comparison-v2','anyanwu-conversational-voice-comparison-v1'),
                    ('waxal-female5-expanded-comparison-v2','conversational-voice-comparison-v1')):
        archive=archive.replace(old,new);review=review.replace(old,new)
    archive=archive.replace('backup_comparison','backup_conversational_comparison').replace('anyanwu-female-expanded-comparison-v2.zip','anyanwu-conversational-voice-comparison-v1.zip')
    for old,new in (('validate_comparison_feedback','validate_conversational_comparison_feedback'),
                    ('register_expanded_comparison_connection','register_conversational_comparison_connection'),
                    ('restore_comparison','restore_conversational_listening'),('show_comparison','show_conversational_comparison'),
                    ('anyanwu.expanded_comparison','anyanwu.conversational_comparison'),
                    ('anyanwu:expanded-comparison-drafts:v2','anyanwu:conversational-comparison-drafts:v1'),
                    ('expanded_comparison_listening_ready','conversational_comparison_listening_ready'),
                    ('expanded_comparison_form_displayed','conversational_comparison_form_displayed'),
                    ('Expanded voice · v2','Conversational trial'),('Expanded voice','Conversational trial')):
        review=review.replace(old,new)
    old='Conversational trial = v2 after 60 additional updates and 14 new reviewed recordings.'
    if old not in review:raise RuntimeError('Review comparison description changed')
    review=review.replace(old,'Conversational trial = the candidate after 30 additional updates using two complete user-reviewed conversational responses and read-speech rehearsal.')
    review=review.replace('These four texts were excluded from both adapter runs; the base may have seen them.',
                          'These four texts were excluded from adapter training. Conversational training references: Silencio Network, Igbo Speech Dataset, CC BY-NC 4.0. Private research; the base may have seen the requested texts.')
    generator=(ROOT/'backend/voice_research/flash_generation.py').read_text();adapter=(ROOT/'backend/voice_research/flash_adapter.py').read_text()
    extras={'runtime-source-receipt.json':(original/'runtime-source-receipt.json').read_text(),
            'COSYVOICE-LICENSE':(original/'reference/LICENSE').read_text(),'MATCHA-LICENSE':(original/'reference/MATCHA-LICENSE').read_text(),
            'comparison_review.py':review}
    tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
 if not str(value.get('phase','')).startswith('conversational_comparison_'):value={**value,'phase':'conversational_comparison_'+value.get('phase','progress')}
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('## Test conversational delivery\\nFour unused published texts, one take for each voice. Same reference, decoder and seed. Inference only; no model training.'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop audio or model work')
 if not shutil.which('nvidia-smi'):raise RuntimeError('Existing hosted GPU required; no CPU synthesis or automatic allocation')
 if not Path('/content/drive/MyDrive').is_dir():
  publish({'phase':'conversational_comparison_waiting_for_drive','optimizer_updates_this_stage':0})
  from google.colab import drive
  drive.mount('/content/drive',timeout_ms=300000)
 report=run_conversational_comparison(PLAN,WORKER,GENERATOR,RUNTIME,ADAPTER,publish)
 publish(report)
 if report.get('completed_wavs')==8 and report.get('source_checkpoints_unchanged'):
  receipt=backup_conversational_comparison(EXTRAS)
  publish(receipt)
  show_conversational_comparison(receipt)
  print('Eight comparison clips verified and backed up. Listen and save each pair; no further training starts automatically.')
except Exception as error:
 publish({'phase':'conversational_comparison_cell_stopped','error_type':type(error).__name__,'error':str(error)[:400],'optimizer_updates_this_stage':0})
 raise
'''
    cell='# Anyanwu conversational voice comparison — inference only,8fixedtakes\n'+helper+'\n'+cloud+'\n'+archive+'\n'+review
    for name,value in (('PLAN',plan),('WORKER',worker),('GENERATOR',generator),('RUNTIME',runtime),('ADAPTER',adapter),('EXTRAS',extras)):cell+='\n'+name+'='+repr(value)+'\n'
    cell+=tail
    for name,code in (('worker',worker),('cloud',cloud),('archive',archive),('review',review),('cell',cell)):compile(code,name,'exec')
    (folder/'comparison-cell.py').write_text(cell);(folder/'comparison_worker.py').write_text(worker)
    (folder/'comparison_review.py').write_text(review);(folder/'comparison_archive.py').write_text(archive)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · conversational voice test\n','Four unused published texts, eight one-take clips. Existing free hosted T4 and research Drive; warm training cache reused when present. No training, paid upgrade or laptop models/audio.\n']},
                 {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
    (ROOT/'Voice Research/Anyanwu-Conversational-Voice-Test.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    receipt={'phase':'conversational_comparison_prepared','cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),
             'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runtime_sha256':prior['runtime_sha256'],
             'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),'outputs':8,'texts':[t['id'] for t in texts],
             'optimizer_updates_this_stage':0,'selected_model_bytes':sum(f['size'] for f in plan['model_files'])}
    (folder/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
