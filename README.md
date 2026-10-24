# DASE7506 MP2 - Compact candidate selection

Student implementation for the course's open-hard-verifier-v5 bank. See `guide/GUIDE.md` for the assignment and `code/README.md` for official tools. See `AI_DISCLOSURE.md` for substantive AI assistance and reuse.

Author: Wang Zheng. Student ID: 3036801785. Submission preparation reviewed on 22 October 2026; frozen model developed on 13 October 2026.

## Reproduce frozen score without training

Use Python 3.12. Commands below use Windows PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.3.5
cd code
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
..\.venv\Scripts\python.exe check_data.py
..\.venv\Scripts\python.exe predict.py --method submission.py --checkpoint runs/checkpoint --split test --output runs/reproduced-predictions.json
..\.venv\Scripts\python.exe score_predictions.py --split test --predictions runs/reproduced-predictions.json --output runs/reproduced-score.json
..\.venv\Scripts\python.exe -m unittest discover -v
```

For Linux use `python3 -m venv .venv`, `.venv/bin/python`, and `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`; use POSIX path separators. The checkpoint is platform-independent.

## Reproduce training and validation search

From root install `requirements-training.txt`, then enter `code` and run `../.venv/Scripts/python.exe train.py` on Windows (use `.venv/bin/python` on Linux). This fits only training data and selects on validation. It overwrites checkpoint and freeze records; preserve the submitted files first if you wish to compare exact hashes. Seed is 42. The grid and selection rule are in train.py; NumPy/BLAS versions can cause numerical differences, so use the supplied checkpoint for exact reported-score reproduction.

`evaluate.py` runs fresh official prediction/scoring processes, four threads, and records local full wall times. Run it only after freezing a method. Never use test results to revise hyperparameters. A separate audit bank works with official `predict.py --split audit --data-dir PATH` using the same checkpoint.

## Artifacts

- `code/runs/checkpoint/`: all required inference assets, no retraining needed.
- `code/runs/freeze.json`: selected hyperparameters, model hashes and training cost.
- `code/runs/source-freeze.json`: code version hashes frozen before final test.
- `code/runs/experiments.json`: complete validation search, ablations and costs.
- `code/runs/measurements.json`: fresh-process local timing and official results.
- `code/runs/test-model-score.json`: official final evaluation JSON.
- `report.md` and `report.pdf`: five-page experimental report.

Only checkpoint/config files are needed for inference, not validation prediction arrays. Source hashes define the delivered code version until a public Git commit is created. The reference Xeon CPU determines runtime eligibility; local timing is not a certification.

To regenerate the report, install `REPORT_DEPENDENCIES.txt` in addition to training dependencies, enter `code`, and run `build_report.py`. The English report is five pages; `中文说明.md` explains the method and submission state in Chinese. The retained upstream PACKAGE_MANIFEST describes the original course release and therefore includes the original empty submission.py hash; use ARTIFACT_MANIFEST.json for this student artifact.
