"""Inference-only Flash LM sampling; no pitch edits, retakes or optimizer calls."""


def generate_speech_tokens(model,prompt_text_ids,prompt_speech_ids,text_ids,seed=17,max_tokens=400):
    import torch
    from torch.nn import functional as F
    silence={1,2,28,29,55,248,494,2241,2242,2322,2323}
    with torch.inference_mode():
        device=model.speech.device
        ids=torch.tensor(prompt_text_ids+text_ids,device=device,dtype=torch.long)
        prompt=torch.tensor(prompt_speech_ids,device=device,dtype=torch.long)
        embedding=model.backbone.embed_tokens
        parts=[model.speech[model.V][None],embedding(model.prefix),embedding(ids)+model.language[1],
               (model.source_vectors[0]+model.rate[3])[None],model.speech[model.V+2][None],F.embedding(prompt,model.speech)]
        inputs=torch.cat(parts)[None];past=None;tokens=[]
        generator=torch.Generator(device=device).manual_seed(seed)
        minimum=2*len(text_ids)
        stop='token_limit'
        for step in range(max_tokens):
            output=model.backbone(inputs_embeds=inputs,past_key_values=past,use_cache=True,return_dict=True)
            past=output.past_key_values
            logp=F.linear(output.last_hidden_state[:,-1],model.decoder).float().log_softmax(-1)[0]
            eos=logp[model.V+1].clone();logp[model.V:]=-float('inf')
            if step>=minimum:logp[model.V+1]=eos
            probabilities=logp.exp()
            ranked,indices=probabilities.sort(descending=True)
            keep=(ranked.cumsum(0)-ranked)<.8;keep[25:]=False
            selected=ranked[keep];selected_ids=indices[keep]
            token=int(selected_ids[torch.multinomial(selected/selected.sum(),1,generator=generator)])
            exempt=False
            if token in silence:
                run=0
                for previous in reversed(tokens):
                    if previous not in silence:break
                    run+=1
                exempt=run<25
            if not exempt and tokens and token!=model.V+1 and sum(t==token for t in tokens[-10:])>=1:
                token=int(torch.multinomial(probabilities,1,generator=generator))
            if token>=model.V:
                stop='eos' if token==model.V+1 else 'unexpected_special_token';break
            tokens.append(token)
            inputs=model.speech[torch.tensor([token],device=device)][:,None]
        if not tokens:raise RuntimeError('LM emitted no acoustic tokens')
        return {'tokens':tokens,'count':len(tokens),'stop_reason':stop,'seed':seed,'minimum_tokens':minimum,
                'warnings':[] if stop=='eos' else ['Generation reached a guard instead of EOS; completeness needs listening']}
