"""Extract the pinned official inference classes without training/server imports.

No tensor/model libraries are imported on the laptop. Class implementations are
copied verbatim with source hashes; checkpoint loading remains strict.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'.build/VoiceResearch/auditions/waxal-female5-adapter-comparison-v1'
REF=FOLDER/'reference'


def extract(filename,names=None):
    path=REF/filename;source=path.read_text();lines=source.splitlines(keepends=True)
    chosen=[];found=[]
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names):
            first=min([node.lineno]+[d.lineno for d in node.decorator_list])
            chosen.append(''.join(lines[first-1:node.end_lineno]));found.append(node.name)
    if names is not None and set(found)!=set(names):raise RuntimeError('Missing pinned class/function: '+filename)
    return '\n\n'.join(chosen),{'file':filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'definitions':found}


def build():
    header='''# Inference subset adapted from QwenAudio/CosyVoice, commit074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc (Apache-2.0),
# and its Matcha-TTS submodule dd9105b34bf2be2230f4aa1e4769fb586a3c824e (MIT).
# Original class/method implementations below are verbatim. Imports are reduced;
# online training-only token extraction is disabled because prompt tokens are supplied.
from __future__ import annotations
from abc import ABC
from typing import Dict,Optional,List,Tuple
import math,random,logging
import numpy as np
import torch
from torch import nn,sin,pow
from torch.nn import Parameter,Conv1d,ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.distributions.uniform import Uniform
import torchaudio
from scipy.signal import get_window
from omegaconf import DictConfig
from x_transformers.x_transformers import RotaryEmbedding,apply_rotary_pos_emb
from einops import repeat
from librosa.filters import mel as librosa_mel_fn
online_feature=False
mel_basis={}
hann_window={}
MAX_WAV_VALUE=32768.0
'''
    specs=[('cosyvoice__utils__mask.py',None),('cosyvoice__utils__common.py',{'get_padding','init_weights','set_all_random_seed'}),
      ('cosyvoice__transformer__convolution.py',{'CausalConv1d','CausalConv1dDownSample','CausalConv1dUpsample'}),
      ('cosyvoice__transformer__activation.py',{'Snake'}),('cosyvoice__transformer__upsample_encoder.py',{'PreLookaheadLayer'}),
      ('matcha-flow.py',{'BASECFM'}),('cosyvoice__flow__flow_matching.py',{'ConditionalCFM','CausalConditionalCFM'}),
      ('cosyvoice__flow__DiT__modules.py',None),('cosyvoice__flow__DiT__dit.py',None),
      ('cosyvoice__flow__flow.py',{'CausalMaskedDiffWithDiT'}),('cosyvoice__hifigan__f0_predictor.py',{'ConvRNNF0Predictor','CausalConvRNNF0Predictor'}),
      ('cosyvoice__hifigan__generator.py',None),('matcha-audio.py',{'dynamic_range_compression_torch','spectral_normalize_torch','mel_spectrogram'})]
    pieces=[header];receipt=[]
    for filename,names in specs:
        source,info=extract(filename,names);receipt.append(info);pieces.append('\n# Source: '+filename+'\n'+source)
    source='\n\n'.join(pieces)+'\n'
    compile(source,'cv3_inference_runtime.py','exec')
    (FOLDER/'cv3_inference_runtime.py').write_text(source)
    (FOLDER/'runtime-source-receipt.json').write_text(json.dumps({'upstream_commit':'074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc','matcha_commit':'dd9105b34bf2be2230f4aa1e4769fb586a3c824e','changes':['Inference-only class extraction; unused trainer/server imports omitted','online_feature=False; supplied cached prompt tokens avoid implicit tokenizer loading'],'definitions':receipt,'runtime_sha256':hashlib.sha256(source.encode()).hexdigest()},indent=2)+'\n')
    return source


if __name__=='__main__':
    source=build();print(json.dumps({'runtime_bytes':len(source.encode()),'tensor_libraries_imported_locally':False}))
