"""Fresh-process official evaluation, run only after model freeze."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    out=Path('runs')
    freeze=json.loads((out/'freeze.json').read_text())
    for name,digest in freeze['checkpoint_sha256'].items():
        assert hashlib.sha256((out/'checkpoint'/name).read_bytes()).hexdigest()==digest
    source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('submission.py','train.py','evaluate.py','common.py','predict.py','score_predictions.py')}
    (out/'source-freeze.json').write_text(json.dumps(source_hashes,indent=2))
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): env[name]='4'
    results=[]
    for split,kind in [('validation','model'),('test','model'),('validation','majority'),('test','majority')]:
        prefix=out/f'{split}-{kind}'
        cmd=[sys.executable,'predict.py','--method','submission.py','--split',split,'--output',str(prefix)+'-predictions.json']
        if kind=='model': cmd+=['--checkpoint',str(out/'checkpoint')]
        tick=time.perf_counter(); subprocess.run(cmd,env=env,check=True)
        prediction=time.perf_counter()-tick
        tick=time.perf_counter()
        subprocess.run([sys.executable,'score_predictions.py','--split',split,'--predictions',str(prefix)+'-predictions.json','--output',str(prefix)+'-score.json'],env=env,check=True)
        scoring=time.perf_counter()-tick
        score=json.loads(Path(str(prefix)+'-score.json').read_text())
        results.append(dict(method=kind,wall_prediction_seconds=prediction,wall_scoring_seconds=scoring,wall_total_seconds=prediction+scoring,**score))
    measurement={'cpu':'AMD Ryzen 7 5800H with Radeon Graphics','platform':platform.platform(),'python':sys.version,'threads':4,'checkpoint_file_bytes':sum(p.stat().st_size for p in (out/'checkpoint').iterdir()),'results':results,'reference_cpu_eligibility_not_measured':True}
    (out/'measurements.json').write_text(json.dumps(measurement,indent=2))


if __name__=='__main__': main()
