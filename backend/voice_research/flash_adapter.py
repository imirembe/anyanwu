"""Small Flash/CosyVoice3 LM adapter. No model imports or downloads at import time.

Sequence and target offsets follow Modular's pinned flash_lm.py FLM.pack;
CosyVoice3 is Apache-2.0; the Flash checkpoint is CC BY-SA 4.0.
This pilot trains speech-token prediction, not the acoustic flow or vocoder.
"""


def speech_targets(prefix_count, text_count, speech_ids, vocabulary):
    if prefix_count < 1 or text_count < 1 or not speech_ids:
        raise ValueError("Prefix, text and speech targets must be nonempty")
    if any(type(t) is not int or t < 0 or t >= vocabulary for t in speech_ids):
        raise ValueError("Invalid acoustic token")
    length = prefix_count + text_count + 3 + len(speech_ids)
    start = prefix_count + text_count + 2
    targets = [-100] * length
    targets[start:start + len(speech_ids)] = speech_ids
    targets[start + len(speech_ids)] = vocabulary + 1
    return targets


def load_qwen_backbone(config, state):
    import torch
    from transformers import Qwen2Model
    with torch.device('meta'):
        backbone = Qwen2Model(config)
    backbone.load_state_dict(state, strict=True, assign=True)
    # Rotary inverse frequencies are nonpersistent buffers, absent from state_dict.
    # Rebuild them from the pinned config rather than leaving a meta placeholder.
    backbone.rotary_emb = type(backbone.rotary_emb)(config=config, device='cpu')
    if any(t.is_meta for t in list(backbone.parameters()) + list(backbone.buffers())):
        raise RuntimeError('Unmaterialised Qwen parameter or buffer')
    return backbone


def install_lora(backbone, rank=4, alpha=8):
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.empty(rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x, self.A), self.B) * (alpha / rank)

    backbone.requires_grad_(False)
    names = []
    for name, module in list(backbone.named_modules()):
        if name.endswith((".self_attn.q_proj", ".self_attn.v_proj")):
            parent_name, attribute = name.rsplit(".", 1)
            setattr(backbone.get_submodule(parent_name), attribute, LoRALinear(module))
            names.append(name)
    if not names:
        raise RuntimeError("No matching Qwen attention projections")
    return names


def make_pilot_model(backbone, speech_weight, decoder_weight, language_weight,
                     rate_weight, speaker_prior, prefix_ids, source_groups):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Pilot(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.modules_adapted = install_lora(backbone)
            for name, weight in (("speech", speech_weight), ("decoder", decoder_weight),
                                 ("language", language_weight), ("rate", rate_weight)):
                self.register_buffer(name, weight.detach(), persistent=False)
            self.register_buffer("prefix", torch.tensor(prefix_ids), persistent=False)
            self.source_vectors = nn.Parameter(speaker_prior.detach()[None].repeat(len(source_groups), 1).clone())
            self.V = speech_weight.shape[0] - 200
            self.sources = list(source_groups)

        def forward(self, text_ids, speech_ids, source_index):
            device = self.speech.device
            text = torch.tensor(text_ids, dtype=torch.long, device=device)
            speech = torch.tensor(speech_ids, dtype=torch.long, device=device)
            embedding = self.backbone.embed_tokens
            parts = [self.speech[self.V][None], embedding(self.prefix),
                     embedding(text) + self.language[1],
                     (self.source_vectors[source_index] + self.rate[3])[None],
                     self.speech[self.V + 2][None], F.embedding(speech, self.speech)]
            x = torch.cat(parts)[None]
            y = torch.tensor(speech_targets(len(self.prefix), len(text_ids), speech_ids, self.V), device=device)
            h = self.backbone(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=device),
                              use_cache=False, return_dict=True).last_hidden_state
            logits = F.linear(h, self.decoder)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y, ignore_index=-100)

        def adapter_state(self):
            return {name: param.detach().contiguous().clone() for name, param in self.named_parameters() if param.requires_grad}

    return Pilot()
