# MP2 code — installation and usage

Read [the project guide](../guide/GUIDE.md) for assessment, constraints, submission and peer review. The complete repository includes all data and representations; no API key or model download is needed. After obtaining the repository, run prediction and scoring commands from **code/**.

## 1. Install and check the data

Clone the complete repository and use **Python 3.12**:

```bash
git clone https://github.com/xudongwu-0/DASE7506-MP2.git
cd DASE7506-MP2/code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

Alternatively, use **Code → Download ZIP** on [GitHub](https://github.com/xudongwu-0/DASE7506-MP2), extract it, enter its `code/` directory and run the environment/install commands above.

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. NumPy is sufficient for the supplied tools. Add and pin your own method's dependencies. After installation, prediction and scoring must work offline.

## 2. Develop your method

Implement `predict(bank, arrays, checkpoint)` in `submission.py`, or provide another Python file with that function. The supplied function is empty and raises `NotImplementedError` until you implement it.

- `bank` is a list in fixed question order. Each row has `id`, `question` and four `candidates`; each candidate has `index` and `text`.
- `arrays['embeddings']` is a float32 NumPy array of shape **[N, 4, 1024]**, aligned with the candidate order. Candidate text and these fixed representations are available to every method.
- `checkpoint` is a `pathlib.Path` or `None`. Save every learned weight, transformation, statistic and configuration needed for inference; return exactly one **Python integer from 0 to 3** per question, in bank order.
- Write any training code you need. Training-free methods are equally eligible. Fit using training inputs/references only; use validation for development. Question IDs are for alignment only.

After you have implemented your method, evaluate validation:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-predictions.json --output runs/validation-score.json
```

`runs/model_bundle` is an asset you create; it is not supplied. Omit `--checkpoint` if your method needs no asset. On Windows, set the thread environment variables before running the same Python commands. The runner requires `--method` and never opens reference answers.

Freeze your method and settings before final testing:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Submit the scorer's **`accuracy`** field, in percent. It checks the selected candidate's final numeric answer using `common.parse_answer`, including equivalent fractions/decimals. It grades final answers, not reasoning validity. A score claimed in a predictions file is ignored; missing/reordered IDs, invalid indices and a mismatched split/protocol are rejected.

## 3. Data and files

| Files | Purpose |
|---|---|
| `submission.py` | Empty prediction interface for your implementation. |
| `predict.py` | Runs your frozen method; writes predictions without reading references. |
| `score_predictions.py`, `common.py` | Fixed independent scorer, numeric parser and data loading. |
| `check_data.py` | Checks hashes, split disjointness and feature/reference alignment. |
| `data/candidates/gsm8k_{split}.jsonl` | Questions and four candidate traces. |
| `data/candidates/gsm8k_{split}.npz` | Fixed whole-trace embeddings; no model weights or correctness labels. |
| `data/gsm8k_{split}.jsonl` | Aligned references: training supervision, validation selection, test scoring only. |
| `data/manifest.json` | Protocol, counts, provenance and data hashes. |
| `data/LICENSE.txt`, `data/NOTICE.txt` | Source license and attribution. |
| `PACKAGE_MANIFEST.json` | Release file hashes, relative to the package root containing code/ and guide/. |

Protocol: **`dase7506-mp2-open-hard-verifier-v5`**. Splits contain **4,755 / 798 / 467** train/validation/test questions and do not overlap. Every group has a correct final answer. Training and validation include easier groups; test questions have at least two different valid parsed answers. Candidate order is fixed. Test references are public for self-reporting and peer reproduction; they must not enter training, lookup tables or method selection.

The instructor may supply a separate audit bank without references:

```bash
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

The same frozen method must accept that bank without retraining or hard-coded sample counts. It should use the bank/arrays passed to `predict()`, rather than opening reference files or assuming that every input is the public test set.

## 4. Resource measurement and reproducibility

- **CPU time:** measure one complete fresh prediction process plus the independent scoring process on the full test bank. Include startup, imports, asset/data loading, preprocessing, all ensemble members and prediction. The total must be at most **30 seconds** on the reference Xeon Platinum 8457C with **four threads across all libraries/processes**. The separate JSON timing fields do not include all process startup; measure externally and report your machine. Local timings are estimates; the instructor's reference CPU decides eligibility.
- **Assets:** at most **16 MiB uncompressed** in total, including all weights, learned transforms/statistics, configurations, indexes and ensemble members. Supplied data and ordinary source code are excluded; arrays hidden in code still count. Compression or reconstruction at startup does not exempt an asset. One MiB is 2^20 bytes.
- **Training:** CPU or one GPU no more powerful than RTX 4090. No course GPU is supplied. Record seeds, dependencies and training/search cost. Inference must run on CPU.
- **Final repository:** include your report (at most 10 pages), exact installation/training/evaluation commands, official evaluation JSON, experiment/resource records, asset hashes and AI/reuse disclosure. Compare your initial and improved methods and analyse a key ablation and failures. Include all files needed for checkpoint evaluation without retraining.
- **Peer reproduction:** use the submitted code version and matching checkpoint, follow its setup, then run the official prediction/scoring commands. Report the measured score and submission number through **Peer Review Report** or the email in the guide. Installation failures or small numerical differences alone do not establish score falsification.

## 5. Attribution

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math), distributed under its retained MIT license. Candidate solutions and fixed representations were prepared with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); the model revision is recorded in the data manifest. No model weights are distributed. Keep `data/LICENSE.txt` and `data/NOTICE.txt` when redistributing this package.
