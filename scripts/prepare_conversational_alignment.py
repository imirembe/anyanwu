"""Prepare offline word/pause timing for exactly the two saved user transcripts."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.build/VoiceResearch/conversational-igbo-v1'
audit=json.loads((folder/'submitted-transcript-review-audit.json').read_text())
drafts=json.loads((folder/'transcription-cloud-result.json').read_text())
if not audit.get('all_marked_exact_and_clean') or not audit.get('snapshot_verified'):
    raise RuntimeError('Expected two explicit saved transcript reviews')
cases=[]
for record in audit['records']:
    original=next(r for r in drafts['records'] if r['id']==record['id'])
    cases.append({'id':record['id'],'transcript_sha256':hashlib.sha256(record['transcript'].encode()).hexdigest(),
                  'analysis_sha256':original['analysis_sha256'],'quiet_candidates':original['pause_analysis']['candidates']})
plan={'id':'anyanwu-conversational-alignment-v1','submitted_snapshot':audit['submitted_snapshot'],
      'submitted_review_sha256':audit['saved_file_sha256'],'asr_model_sha256':drafts['asr_model_sha256'],
      'cases':cases,'purpose':'Machine word timings and possible phrase gaps; preserve user wording including spoken English.',
      'limits':{'minutes':10,'child_rss_gib':6,'outputs_mib':12,'log_kib':256,'cloud_free_floor_gib':5,'network_requests':0},
      'source_transcript_edits':False,'audio_cuts':False,'optimizer_updates':0,'training_ready':False}
(folder/'word-timing-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
rules=(ROOT/'backend/voice_research/alignment_rules.py').read_text()
worker=(ROOT/'backend/voice_research/conversational_alignment_worker.py').read_text()
worker=rules+'\n'+worker.replace('from backend.voice_research.alignment_rules import alignment_text, caption_token_ranges\n','')
compile(worker,'conversational-timing-worker','exec')
setup=(ROOT/'backend/voice_research/alignment_setup.py').read_text()
tail='''
import hashlib,html
from IPython.display import Javascript,HTML,display
def publish(value):
 if not str(value.get('phase','')).startswith('conversational_asr_'):value={**value,'phase':'conversational_asr_'+value.get('phase','timing')}
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
 report=run_alignment(PLAN,WORKER,publish)
 if report.get('phase')=='conversational_asr_alignment_complete':
  root=Path('/content')/PLAN['id'];backup=Path('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/references/conversational-igbo-v1/transcription-v1/alignment-v1')
  if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive unavailable; cloud timings preserved, no backup claimed')
  backup.mkdir(exist_ok=True)
  files=['plan.json','alignment-summary.json','attempt.json','worker.py']+[r['timing_file'] for r in report['cases'] if r.get('timing_file')]
  receipts={}
  for name in files:
   path=root/name;raw=path.read_bytes()
   if path.is_symlink() or len(raw)>512*1024:raise RuntimeError('Timing metadata backup cap')
   target=backup/name
   if target.exists() and target.read_bytes()!=raw:raise RuntimeError('Different timing backup preserved')
   if not target.exists():
    with target.open('xb') as handle:handle.write(raw)
   if target.read_bytes()!=raw:raise RuntimeError('Timing metadata Drive verification failed')
   receipts[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  report={**report,'drive_backup_verified':True,'drive_backup':str(backup),'backup_files':receipts}
  blocks=[]
  for case in report['cases']:
   lines=['<h3>'+html.escape(case['id'])+'</h3><p>'+html.escape(case['status'])+'</p>']
   if case.get('warnings'):lines.append('<p>Timing flags: '+html.escape(', '.join(case['warnings']))+'</p>')
   for group in case.get('phrase_preview',[]):lines.append('<p>'+str(round(group['start_seconds'],2))+'–'+str(round(group['end_seconds'],2))+'s · '+html.escape(group['alignment_view_text'])+'</p>')
   if case.get('error'):lines.append('<p>'+html.escape(case['error'])+'</p>')
   blocks.append(''.join(lines))
  display(HTML('<section style="max-width:850px;padding:24px;background:#f7f5eb;color:#173f34;font:17px/1.7 system-ui"><h2>Conversational phrasing · timing candidates</h2><p>Your saved wording, including spoken English, is preserved. Below is a separate normalized alignment view. Groups reflect acoustic gaps, not verified linguistic phrases. No audio was cut and no model trained.</p>'+''.join(blocks)+'</section>'))
 publish(report)
except Exception as error:
 publish({'phase':'conversational_asr_alignment_stopped','error_type':type(error).__name__,'error':str(error)[:350],'optimizer_updates':0})
 raise
'''
cell='# Anyanwu offline timing of user-reviewed conversational Igbo — no downloads or training\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\n'+tail
compile(cell,'conversational-alignment-cell','exec')
(folder/'word-timing-cell.py').write_text(cell)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
    'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · conversational word and pause timing\n','Existing warm CPU environment only. No downloads, cuts, transcript edits or training.\n']},
             {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
(ROOT/'Voice Research/Anyanwu-Conversational-Timing.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
receipt={'phase':'conversational_timing_prepared','cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'cases':2,
         'snapshot_sha256':audit['saved_file_sha256'],'network_requests_allowed':0,'training':False}
(folder/'word-timing-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
