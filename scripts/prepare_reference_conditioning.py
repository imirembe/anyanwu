"""Build a bounded fixed-checkpoint comparison without changing earlier trials."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT/'.build/VoiceResearch'
FOLDER = LAB/'auditions/conversational-reference-conditioning-v1'
NAME = 'conversational-reference-conditioning-v1'
CLOUD = '/content/anyanwu-'+NAME


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('Pinned comparison template changed: '+before[:80])
    return source.replace(before, after)


def prepare():
    plan = json.loads((FOLDER/'plan.json').read_text())
    previous = json.loads((LAB/'auditions/conversational-voice-comparison-v1/plan.json').read_text())
    trial = json.loads((LAB/'training/conversational-style-lora-v1/plan.json').read_text())
    reviewed = json.loads((LAB/'conversational-igbo-v1/submitted-transcript-review-audit.json').read_text())
    if plan['status'] != 'prepared_not_run': raise RuntimeError('Executed diagnostic preserved')
    plan.update(variants=['base','adapter'], variant_labels={'base':'Reading reference · v1','adapter':'Conversational reference · v1'},
        adapter_hashes={v:plan['checkpoint_sha256'] for v in ('base','adapter')},
        prompt_id='ibo_616', reference_by_variant={'base':'ibo_616','adapter':'conversation-ref-03'},
        archives={'previous':trial['baseline_archive']},
        submitted_snapshot=reviewed['submitted_snapshot'], encoded_targets_path=trial['encoded_targets']['path'],
        reference_audio_path='/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/references/conversational-igbo-v1/conversation-ref-03.wav',
        previously_reviewed_or_trained_ids=sorted(set(trial['comparison']['prior_comparison_texts_excluded'])|
            {t['id'] for t in trial['comparison']['texts']}|set(trial['schedule'])),
        optimizer_updates_this_stage=0)
    plan['conditions'][1]['analysis_sha256']=trial['capacity_cases'][0]['analysis_sha256']
    for key in ('base_repo','base_revision','base_checkpoint_sha256','acoustic_repo','acoustic_revision','model_files','runtime_sha256','limits'):
        plan[key]=previous[key]
    plan['reference_boundary_policy']='Keep complete WAVs and exact reviewed text; publisher 2:1 mel/token boundary matching only, maximum two speech tokens/four mel frames of edge rounding, recorded explicitly. No internal edits or arbitrary context shortening.'
    plan['scope']='Private noncommercial research, current Colab/Drive only. Four one-take outputs, fixed v1, zero training, no automatic adoption.'
    (FOLDER/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    core=(ROOT/'backend/voice_research/reference_conditioning.py').read_text()
    helpers=core[core.index('def context_positions'):core.index('def restore_reference_inputs')]
    worker=(ROOT/'backend/voice_research/expanded_comparison_worker.py').read_text()
    worker=worker.replace('Generate eight fixed one-take clips from the preserved and expanded adapters.', 'Generate four one-take clips with fixed v1 and two reference recordings.')
    worker=worker.replace('anyanwu-waxal-female5-expanded-comparison-v2','anyanwu-'+NAME)
    worker=replace_once(worker,"'adapter':120", "'adapter':60")
    worker=replace_once(worker,'def main():',helpers+'\ndef main():')
    start=worker.index("        source=next(c for c in data['clips']")
    end=worker.index("        token_receipt=ROOT/'token-result.json'",start)
    preparation='''        references={};reference_receipts={}
        def load_pcm(path,rate):
            with wave.open(str(path),'rb') as handle:
                if (handle.getnchannels(),handle.getsampwidth(),handle.getframerate())!=(1,2,rate):raise RuntimeError('Unexpected reference PCM format')
                values=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
            if not 1<=len(values)/rate<=30:raise RuntimeError('Whole reference duration cap')
            return torch.from_numpy(values.copy())[None]
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(acoustic['campplus.onnx'],options,providers=['CPUExecutionProvider'])
        for variant in ('base','adapter'):
            reference_id=plan['reference_by_variant'][variant]
            source=next(c for c in data['clips'] if c['id']==reference_id)
            prompt=next(c for c in targets if c['id']==reference_id)
            wav16=Path(source['cloud_audio_path'])
            if variant=='base':
                mp3=TRAIN/'data'/(reference_id+'.mp3')
                if sha(mp3)!=source['mp3_sha256']:raise RuntimeError('Read reference integrity failure')
                wav24=ROOT/'prompt-24k.wav'
                if not wav24.exists():
                    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-i',str(mp3),'-ac','1','-ar','24000','-c:a','pcm_s16le',str(wav24)],check=True,capture_output=True,timeout=30)
            else:
                wav24=Path(source['cloud_audio24_path'])
                if sha(wav24)!=source['wave24_sha256']:raise RuntimeError('Conversational WAV integrity failure')
                if not wav16.exists():
                    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-i',str(wav24),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav16)],check=True,capture_output=True,timeout=45)
            if sha(wav16)!=source['audio_sha256']:raise RuntimeError('Whole reference analysis integrity failure')
            speech16=load_pcm(wav16,16000);speech24=load_pcm(wav24,24000)
            pfeat=mel_spectrogram(speech24,n_fft=1920,num_mels=80,sampling_rate=24000,hop_size=480,win_size=1920,fmin=0,fmax=None,center=False).squeeze(0).transpose(0,1)[None]
            frames=pfeat.shape[1];n=match_reference_boundary(len(prompt['speech_ids']),frames)
            pfeat=pfeat[:,:2*n].contiguous();prompt_tokens=prompt['speech_ids'][:n]
            fbank=kaldi.fbank(speech16,num_mel_bins=80,dither=0,sample_frequency=16000);fbank=fbank-fbank.mean(0,keepdim=True)
            speaker=torch.tensor(session.run(None,{session.get_inputs()[0].name:fbank[None].numpy()})[0].reshape(1,-1))
            if speaker.shape!=(1,192) or not torch.isfinite(speaker).all():raise RuntimeError('Reference speaker embedding invalid')
            references[variant]=(prompt,prompt_tokens,pfeat,speaker)
            reference_receipts[variant]={'id':reference_id,'original_speech_tokens':len(prompt['speech_ids']),'matched_speech_tokens':n,
                'original_mel_frames':frames,'matched_mel_frames':2*n,'edge_tokens_removed':len(prompt['speech_ids'])-n,
                'edge_mel_frames_removed':frames-2*n,'whole_waveform_seconds':speech24.shape[1]/24000,'text_tokens':len(prompt['text_ids'])}
        (ROOT/'reference-preparation.json').write_text(json.dumps(reference_receipts,indent=2)+'\\n')
        del session,fbank,speech16,speech24;gc.collect()
'''
    worker=worker[:start]+preparation+worker[end:]
    worker=worker.replace('len(tokens)!=8','len(tokens)!=4')
    worker=worker.replace('loading_previous_and_expanded_speech_lm','loading_fixed_v1_speech_lm')
    worker=replace_once(worker,"            for variant in ('base','adapter'):\n                path=", "            for variant in ('base','adapter'):\n                prompt,prompt_tokens,_,_=references[variant]\n                path=")
    worker=replace_once(worker,"                    generated=generate_speech_tokens", "                    used=context_positions(len(prefix),len(prompt['text_ids']),len(prompt_tokens),len(text_ids),plan['max_speech_tokens'],plan['max_context_positions'])\n                    generated=generate_speech_tokens")
    worker=replace_once(worker,"'count':generated['count'],'stop_reason'", "'context_positions_reserved':used,'reference_id':plan['reference_by_variant'][variant],\n                                   'count':generated['count'],'stop_reason'")
    worker=replace_once(worker,"        pfeat=pfeat.to('cuda');speaker=speaker.to('cuda');ptok=torch.tensor([prompt_tokens],dtype=torch.int32,device='cuda')\n",'')
    worker=replace_once(worker,"        for item in tokens:\n            path=", "        for item in tokens:\n            _,prompt_tokens,pfeat,speaker=references[item['variant']]\n            pfeat=pfeat.to('cuda');speaker=speaker.to('cuda');ptok=torch.tensor([prompt_tokens],dtype=torch.int32,device='cuda')\n            path=")
    worker=worker.replace("'eight_generated_comparison_clips_ready'","'four_generated_reference_clips_ready'").replace('completed_wavs=8','completed_wavs=4')
    worker=replace_once(worker,"        report['source_checkpoints_unchanged']=True", "        report['source_checkpoints_unchanged']=True\n        report['reference_preparation']=reference_receipts\n        report['fixed_adapter_in_both_conditions']=True")
    # Keep the bounded warm/cold environment runner, changing its declared job and restore only.
    template=(ROOT/'backend/voice_research/conversational_comparison_cloud.py').read_text()
    cloud=template[:template.index('def restore_conversational_comparison')]+template[template.index('def run_conversational_comparison'):]
    cloud=cloud.replace('from backend.voice_research.expanded_comparison_cloud import read_comparison_archive,comparison_sha\n','')
    cloud=cloud.replace('run_conversational_comparison','run_reference_comparison').replace('conversational_comparison_','reference_conditioning_')
    cloud=cloud.replace('anyanwu-conversational-voice-comparison-v1','anyanwu-'+NAME)
    cloud=cloud.replace('!=8','!=4').replace('completed_wavs":8','completed_wavs":4').replace('Required eight','Required four')
    cloud=replace_once(cloud,'publish(restore_conversational_comparison(plan))','publish(restore_reference_inputs(plan,root=ROOT,warm=WARM))')
    # Every generated source is included in the immutable audio archive.
    archive=(ROOT/'backend/voice_research/expanded_comparison_archive.py').read_text()
    archive=archive.replace('Archive eight generated clips','Archive four generated clips')
    archive=archive.replace('anyanwu-waxal-female5-expanded-comparison-v2','anyanwu-'+NAME).replace('waxal-female5-expanded-comparison-v2',NAME)
    archive=archive.replace('backup_comparison','backup_reference_comparison').replace('anyanwu-female-expanded-comparison-v2.zip','anyanwu-'+NAME+'.zip')
    archive=archive.replace("!=8","!=4").replace("'audio_clips':8","'audio_clips':4").replace('Eight completed','Four completed')
    archive=replace_once(archive,"'cv3_runtime.py']","'cv3_runtime.py','reference-preparation.json']")
    review=(ROOT/'backend/voice_research/expanded_comparison_review.py').read_text()
    review=review.replace('anyanwu-waxal-female5-expanded-comparison-v2','anyanwu-'+NAME).replace('waxal-female5-expanded-comparison-v2',NAME)
    review=review.replace('anyanwu.expanded_comparison','anyanwu.reference_conditioning').replace('anyanwu:expanded-comparison-drafts:v2','anyanwu:reference-conditioning-drafts:v1')
    review=review.replace('expanded_comparison_','reference_conditioning_').replace('register_expanded_comparison_connection','register_reference_connection')
    review=review.replace('validate_comparison_feedback','validate_reference_feedback').replace('show_comparison','show_reference_comparison').replace('restore_comparison','restore_reference_listening')
    review=review.replace("len(plan['texts'])!=4","len(plan['texts'])!=2").replace("len(result['clips'])!=8","len(result['clips'])!=4").replace("result['completed_wavs']!=8","result['completed_wavs']!=4")
    review=review.replace('len(allowed)!=4','len(allowed)!=2').replace("'pairs':4","'pairs':2").replace('r.pairs===4','r.pairs===2').replace("'audio_players':8","'audio_players':4").replace('Expected four complete comparison pairs','Expected two complete comparison pairs')
    description='Same preserved v1 model in both versions. Only the reference changes: reading versus the complete conversational recording you selected. Reference identity, style and duration change together. One take each; no new training or pitch edits.'
    old='Previous voice = the preserved 60-update v1 adapter. Expanded voice = v2 after 60 additional updates and 14 new reviewed recordings. Both use the same female reference, decoder and seed. One take each; no pitch edits or new training in this comparison.'
    review=replace_once(review,old,description)
    review=review.replace('Previous voice · v1','Reading reference · v1').replace('Expanded voice · v2','Conversational reference · v1').replace('Previous voice','Reading reference').replace('Expanded voice','Conversational reference')
    review=review.replace('Listen to the female Igbo comparison','Compare connected Igbo speech')
    review=review.replace('These four texts were excluded from both adapter runs; the base may have seen them.', 'Two unused published texts; base pretraining exposure is possible. Conversational reference: Silencio Network, Igbo Speech Dataset, 2026, CC BY-NC 4.0. Private noncommercial research only.')
    generator=(ROOT/'backend/voice_research/flash_generation.py').read_text()
    adapter=(ROOT/'backend/voice_research/flash_adapter.py').read_text()
    original=LAB/'auditions/waxal-female5-adapter-comparison-v1'
    runtime=(original/'cv3_inference_runtime.py').read_text()
    if hashlib.sha256(runtime.encode()).hexdigest()!=plan['runtime_sha256']:raise RuntimeError('Pinned acoustic runtime changed')
    helper=(ROOT/'backend/voice_research/expanded_comparison_cloud.py').read_text().split('\ndef restore_expanded_comparison',1)[0]
    core=core.replace('    from backend.voice_research.expanded_comparison_cloud import read_comparison_archive\n','')
    extras={'runtime-source-receipt.json':(original/'runtime-source-receipt.json').read_text(),
        'COSYVOICE-LICENSE':(original/'reference/LICENSE').read_text(),
        'MATCHA-LICENSE':(original/'reference/MATCHA-LICENSE').read_text(), 'comparison_review.py':review}
    cell='# Anyanwu conversational reference diagnostic — four one-take clips, fixed v1\n'+helper+'\n'+core+'\n'+cloud+'\n'+archive+'\n'+review
    for name,value in [('PLAN',plan),('WORKER',worker),('GENERATOR',generator),('RUNTIME',runtime),('ADAPTER',adapter),('EXTRAS',extras)]:cell+='\n'+name+'='+repr(value)+'\n'
    cell+='''
from IPython.display import Javascript,display,Markdown
def publish(value):
 if not str(value.get('phase','')).startswith('reference_conditioning_'):value={**value,'phase':'reference_conditioning_'+value.get('phase','progress')}
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('## Compare conversational flow\\nSame v1 model, two reference recordings, two new published texts. Four takes; no training.'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop audio or models')
 if not shutil.which('nvidia-smi'):raise RuntimeError('Existing hosted GPU required; no automatic allocation or CPU synthesis')
 if not Path('/content/drive/MyDrive').is_dir():
  publish({'phase':'reference_conditioning_waiting_for_drive','optimizer_updates_this_stage':0})
  from google.colab import drive
  drive.mount('/content/drive',timeout_ms=300000)
 report=run_reference_comparison(PLAN,WORKER,GENERATOR,RUNTIME,ADAPTER,publish)
 publish(report)
 if report.get('completed_wavs')==4 and report.get('source_checkpoints_unchanged'):
  receipt=backup_reference_comparison(EXTRAS);publish(receipt)
  show_reference_comparison(receipt)
  print('Four one-take clips verified and saved. Listen for word connections, pauses and tone; save each pair. No training starts automatically.')
except Exception as error:
 publish({'phase':'reference_conditioning_cell_stopped','error_type':type(error).__name__,'error':str(error)[:400],'optimizer_updates_this_stage':0})
 raise
'''
    outputs={'reference_worker.py':worker,'reference_cloud.py':cloud,'reference_review.py':review,'reference_archive.py':archive,'reference-cell.py':cell}
    for name,source in outputs.items():
        compile(source,name,'exec');(FOLDER/name).write_text(source)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · conversational reference diagnostic\n','Existing free hosted GPU and research Drive only. Fixed v1 model, four one-take clips. No training or laptop model/audio downloads.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
    (ROOT/'Voice Research/Anyanwu-Conversational-Reference-Test.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    receipt={'phase':'reference_conditioning_prepared','outputs':4,'checkpoint_sha256':plan['checkpoint_sha256'],
        'plan_sha256':hashlib.sha256((FOLDER/'plan.json').read_bytes()).hexdigest(),
        'source_sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in outputs.items()},
        'optimizer_updates_this_stage':0,'executed':False}
    (FOLDER/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
