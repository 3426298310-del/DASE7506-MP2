"""Train-only fitting and validation-only selection; never loads test data."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='4'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs, read_jsonl, parse_answer
from submission import features, majority, model_scores


def labels(bank, split):
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert len(refs)==len(bank) and all(r['id']==b['id'] for r,b in zip(refs,bank))
    return np.array([[int(parse_answer(c['text'])==parse_answer(r['answer'])) for c in b['candidates']] for r,b in zip(refs,bank)])


def accuracy(y,scores):
    return float(100*y[np.arange(len(y)),np.asarray(scores).argmax(axis=1)].mean())


def export(model,scaler):
    result={'mean':scaler.mean_.astype('float32'),'scale':scaler.scale_.astype('float32')}
    ws=model.coefs_ if hasattr(model,'coefs_') else [model.coef_.T]
    bs=model.intercepts_ if hasattr(model,'intercepts_') else [model.intercept_]
    for i,(w,b) in enumerate(zip(ws,bs)):
        result[f'w{i}']=w.astype('float32'); result[f'b{i}']=b.astype('float32')
    return result,len(ws)


def main():
    start=time.perf_counter(); out=Path('runs'); out.mkdir(exist_ok=True)
    train,ta=load_inputs('train'); val,va=load_inputs('validation')
    yt=labels(train,'train'); yv=labels(val,'validation')
    baseline=majority(val)
    results=[{'name':'majority','validation_accuracy':float(100*yv[np.arange(len(yv)),baseline].mean()),'fit_seconds':0}]
    best=-1; selected=None
    with threadpool_limits(limits=4):
        for mode in ('embedding','text','no_consensus','full'):
            xt=features(train,ta,mode); xv=features(val,va,mode)
            scaler=StandardScaler().fit(xt); xt=scaler.transform(xt); xv=scaler.transform(xv)
            options=[('logistic',c) for c in (0.01,0.1,1.0)]
            if mode=='full': options += [('mlp',a) for a in (0.1,1.0,10.0)]
            for kind,param in options:
                tick=time.perf_counter()
                model=LogisticRegression(C=param,max_iter=400,random_state=42) if kind=='logistic' else MLPClassifier(hidden_layer_sizes=(32,),alpha=param,max_iter=120,early_stopping=True,n_iter_no_change=10,random_state=42,batch_size=256)
                model.fit(xt,yt.reshape(-1))
                scores=model.predict_proba(xv)[:,1].reshape(-1,4); acc=accuracy(yv,scores)
                bundle,layers=export(model,scaler)
                meta={'mode':mode,'layers':layers,'kind':kind,'parameter':param,'seed':42,'validation_accuracy':acc,'feature_dimension':xt.shape[1]}
                exported=model_scores(val,va,bundle,meta)
                agree=np.array_equal(exported.argmax(axis=1),scores.argmax(axis=1))
                raw=xv
                if kind=='mlp':
                    for w,b in zip(model.coefs_[:-1],model.intercepts_[:-1]): raw=np.maximum(raw@w+b,0)
                    raw=(raw@model.coefs_[-1]+model.intercepts_[-1]).reshape(-1,4)
                else: raw=model.decision_function(xv).reshape(-1,4)
                np.testing.assert_allclose(exported,raw,rtol=2e-4,atol=2e-4)
                # Select the actual exported inference method, including its tie behavior.
                acc=accuracy(yv,exported); meta['validation_accuracy']=acc
                rec={'name':f'{mode}_{kind}_{param}','validation_accuracy':acc,'fit_seconds':time.perf_counter()-tick,'iterations':int(model.n_iter_ if np.isscalar(model.n_iter_) else model.n_iter_[0]),'export_choices_match':bool(agree),'probability_choice_disagreements':int(np.count_nonzero(exported.argmax(axis=1)!=scores.argmax(axis=1))),'raw_logits_allclose':True,'asset_array_bytes':sum(v.nbytes for v in bundle.values())}
                results.append(rec); print(json.dumps(rec),flush=True)
                np.savez(out/(rec['name']+'.npz'),scores=exported,selected=exported.argmax(axis=1))
                if acc>best: best=acc; selected=(bundle,meta)
            (out/'experiments.json').write_text(json.dumps(results,indent=2))
    bundle,meta=selected; checkpoint=out/'checkpoint'; checkpoint.mkdir(exist_ok=True)
    np.savez(checkpoint/'weights.npz',**bundle); (checkpoint/'config.json').write_text(json.dumps(meta,indent=2))
    freeze={'selected_model':meta,'checkpoint_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()},'training_total_seconds':time.perf_counter()-start,'test_not_used_for_selection':True,'search_models':len(results)-1,'selection_rule':'highest validation accuracy; first grid entry wins ties','internal_mlp_early_stopping':'candidate-level holdout within training; related candidates may share questions'}
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2)); print(json.dumps(freeze,indent=2),flush=True)


if __name__=='__main__': main()
