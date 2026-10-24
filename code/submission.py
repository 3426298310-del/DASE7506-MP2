"""Offline selector; reads passed inputs and checkpoint only."""
import json
import math
import re
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import parse_answer

TEXT_DIM = 16


def features(bank, arrays, mode='full'):
    n = len(bank)
    e = np.asarray(arrays['embeddings'], dtype=np.float32).reshape(n,4,1024)
    t = np.zeros((n,4,TEXT_DIM), dtype=np.float32)
    for k, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(s) for s in texts]
        qnums = set(re.findall(r'\d+(?:\.\d+)?', row['question']))
        for j, s in enumerate(texts):
            a = answers[j]
            nums = set(re.findall(r'\d+(?:\.\d+)?', s))
            rational = Fraction(a) if a is not None else Fraction(0)
            try:
                magnitude = np.log1p(abs(float(rational)))
            except OverflowError:
                magnitude = max(0., math.log(abs(rational.numerator))-math.log(rational.denominator))
            t[k,j] = [sum(a is not None and a == b for b in answers)/4,
                float(a is not None), np.log1p(len(s)), np.log1p(len(s.split())),
                np.log1p(s.count('\n')), np.log1p(len(re.findall(r'\d+',s))),
                np.log1p(s.count('=')), np.log1p(len(re.findall(r'[+*/×÷]',s))),
                len(qnums & nums)/max(1,len(qnums)), magnitude,
                float(rational < 0), float(a is not None and '/' in a),
                float('boxed' in s), float('####' in s),
                np.log1p(len(re.findall(r'\b(?:wait|however|mistake|instead|reconsider)\b',s.lower()))),
                np.log1p(len(re.findall(r'\b(?:therefore|thus|so|hence)\b',s.lower())))]
    if mode == 'text': return t.reshape(n*4,TEXT_DIM)
    if mode == 'embedding': return e.reshape(n*4,1024)
    if mode == 'no_consensus': t[:,:,0] = 0
    return np.concatenate([e, e-e.mean(axis=1,keepdims=True), t],axis=2).reshape(n*4,2064)


def majority(bank):
    choices=[]
    for row in bank:
        a=[parse_answer(c['text']) for c in row['candidates']]
        choices.append(int(np.argmax([sum(x is not None and x==y for y in a) for x in a])))
    return choices


def model_scores(bank, arrays, bundle, meta):
    x=(features(bank,arrays,meta['mode'])-bundle['mean'])/bundle['scale']
    for layer in range(meta['layers']):
        x=x@bundle[f'w{layer}']+bundle[f'b{layer}']
        if layer<meta['layers']-1: x=np.maximum(x,0)
    return x.reshape(len(bank),4)


def predict(bank, arrays, checkpoint):
    if not bank: return []
    if checkpoint is None: return majority(bank)
    checkpoint=Path(checkpoint)
    meta=json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    with np.load(checkpoint/'weights.npz',allow_pickle=False) as bundle:
        scores=model_scores(bank,arrays,bundle,meta)
    return [int(i) for i in scores.argmax(axis=1)]
