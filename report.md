# Compact Selection of Reliable LLM Answers

DASE7506 Mini-project 2 | Experimental report | 22 October 2026

Author: Wang Zheng | Student ID: 3036801785

Experiments completed on 13 October 2026; checkpoint reproduction rechecked on 22 October 2026. Substantive AI assistance is disclosed in Section 11.

## 1. Research question and hypothesis

Can a small learned candidate scorer outperform answer majority voting under a strict CPU and asset budget? The hypothesis is that fixed trace embeddings and inexpensive text statistics provide complementary evidence. A second hypothesis is that explicit agreement between candidate answers improves selection; this is tested rather than assumed.

## 2. Task and protocol

The bank provides four candidate solutions for each question and fixed 1024-dimensional embeddings. We choose an index from 0 to 3. Any candidate with the correct final numeric answer receives credit. The metric does not certify intermediate reasoning. Training/validation/test contain 4,755/798/467 questions. Each question has at least one correct candidate; the test bank additionally requires at least two different valid parsed answers, unlike easier groups present in training and validation.

Only supplied training references supervise fitting. Validation selects hyperparameters. The checkpoint was frozen and hashed before test evaluation. No external datasets, pretrained weights, answer lookup tables, or LLM generation/API calls are used. Official data, numeric parser and scorer remain unchanged.

## 3. Main result

| Method | Validation | Test |
| --- | --- | --- |
| Majority vote | 688/798 (86.22%) | 323/467 (69.16%) |
| Frozen linear scorer | 739/798 (92.61%) | 391/467 (83.73%) |

The frozen method improves test accuracy by 14.56 percentage points over majority voting. The test result was not used to revise the model. Results apply to this course bank; they are not a general guarantee of reasoning reliability.


---

# Method and Experimental Design

## 4. Candidate scoring

The baseline uses the official parser to canonicalize equivalent fractions and decimals, counts valid answer agreement, and selects the first maximum. Invalid parses do not vote together. For the learned model, each candidate receives a binary training label indicating whether its parsed final answer equals the training reference. Multiple candidates can be positive.

Features concatenate the supplied embedding, its difference from the four-candidate mean, and 16 inexpensive statistics. These include valid answer agreement, parse validity, logarithmic character/word/line/number/equality/operator counts, question-number coverage, logarithmic absolute final-answer magnitude, negative/fraction flags, answer-marker flags and counts of revision/conclusion terms. They are statistical cues, not a symbolic verification of arithmetic or problem conditions.

Each coordinate is standardized using training statistics only. Logistic regression minimizes binary cross entropy with L2 regularization. At inference the highest raw logit wins. The selected C is 0.01. Explicit answer agreement is zeroed in the selected no_consensus configuration. The model contains 2,064 feature coordinates, their means/scales and one linear weight vector.

For a linear scorer, subtracting the same within-question embedding mean contributes only a shared score offset; it cancels in candidate ranking. Therefore the centered block is algebraically redundant at inference, and these experiments do not establish a benefit from comparative embedding features. The redundant block changes the effective training regularization and is retained to reproduce the frozen checkpoint.

## 5. Controlled comparisons and search

Four feature settings are compared: embeddings alone; text statistics alone; full features without explicit agreement; and full features with agreement. Each uses C in {0.01, 0.1, 1.0}. Full features additionally use a 32-unit ReLU MLP with alpha in {0.1, 1.0, 10.0}. All 15 fitted models use seed 42. The highest exported-model validation accuracy wins; exact ties use the first grid entry. The baseline was recorded but not a candidate in this grid; all fitted models outperform it here.

MLP uses training-only internal early stopping: 10% candidate-level holdout, patience 10, maximum 120 epochs, batch 256. Related candidates can share questions across this internal split, making that stopping estimate optimistic; external validation remains question-disjoint. Logistic regression maximum iterations is 400. Training runs on CPU with four library threads.


---

# Validation Search and Ablations

| Configuration | Validation accuracy | Fit/export seconds |
| --- | --- | --- |
| majority | 86.22% | 0.00 |
| embedding_logistic_0.01 | 89.22% | 1.19 |
| embedding_logistic_0.1 | 89.35% | 1.94 |
| embedding_logistic_1.0 | 88.85% | 2.28 |
| text_logistic_0.01 | 89.22% | 0.26 |
| text_logistic_0.1 | 89.35% | 0.26 |
| text_logistic_1.0 | 89.22% | 0.26 |
| no_consensus_logistic_0.01 | 92.61% | 2.38 |
| no_consensus_logistic_0.1 | 92.11% | 4.23 |
| no_consensus_logistic_1.0 | 91.60% | 6.21 |
| full_logistic_0.01 | 91.85% | 2.03 |
| full_logistic_0.1 | 91.48% | 3.69 |
| full_logistic_1.0 | 91.35% | 5.64 |
| full_mlp_0.1 | 92.48% | 4.89 |
| full_mlp_1.0 | 91.60% | 4.51 |
| full_mlp_10.0 | 91.98% | 2.90 |

## 6. Interpretation

At fixed C=0.01, full features without agreement achieve 92.61% versus 91.85% with agreement: explicit consensus costs six correct validation selections (0.75 percentage points). This is a small observed difference, not evidence that agreement is universally harmful. Embedding-only and text-only selected models each reach 89.35%; combining their evidence gives a stronger result here.

The best MLP reaches 92.48%, one validation question below the selected linear model, with substantially larger arrays. The experiment supports using the compact linear method within this search, rather than claiming that nonlinear modeling is intrinsically inferior. The number of searched models can bias the best validation estimate upward; there is no repeated-seed or external-dataset evaluation.


---

# Reliability and Error Analysis

## 7. Dataset difficulty and uncertainty

| Validation subgroup | Questions | Selected accuracy |
| --- | --- | --- |
| one valid answer value | 443 | 100.00% |
| two or more answer values | 355 | 83.38% |

The observed test accuracy is 8.88 percentage points below validation. Differences in answer diversity are one plausible contributor because the test protocol excludes easy single-answer-value groups. This is not a causal attribution: other distribution differences and validation selection optimism can also contribute.

| Logit-margin quartile | Questions | Margin range | Accuracy |
| --- | --- | --- | --- |
| Q1 | 200 | 0.000 - 0.278 | 93.50% |
| Q2 | 200 | 0.280 - 0.701 | 93.50% |
| Q3 | 199 | 0.702 - 1.370 | 90.95% |
| Q4 | 199 | 1.378 - 8.347 | 92.46% |

Accuracy does not increase monotonically with margin: Q1 reaches 93.50% while Q4 reaches 92.46%. Thus these results do not support treating larger margins as greater reliability. Easy groups with agreeing answers can have low or zero margins, confounding this comparison. Margins are uncalibrated ranking evidence, not probabilities of correctness. Calibration and abstention would need a separate validation study.

The 95% Wilson interval for the test proportion is 80.11% to 86.80%, if questions are treated as independent draws. This interval does not remove bank-selection bias or account for a future audit distribution. All 467 public test questions were scored.

## 8. Concrete validation failures

Example 1: Lizette scored 92 on her third quiz giving her an average of 94. What is Lizette's average in her first two quizzes?

Selected candidate 2: 98; reference: 95; score margin: 0.071.

Example 2: Eddy’s spider plant produces 2 baby plants 2 times a year.  After 4 years, how many baby plants will the mother plant have produced?

Selected candidate 0: 8; reference: 16; score margin: 0.826.

Example 3: Megan bought 2 dozen eggs. As she was walking to her car, she dropped a tray of eggs. 3 eggs broke, and twice as many cracked. What is the difference between the eggs that are still in perfect condition and those that are cracked?

Selected candidate 2: 6; reference: 9; score margin: 0.036.

These are the first three incorrect validation questions in bank order, not hand-picked to support a causal story. Final-answer mismatch is established; the model itself does not identify the flawed reasoning step. Text length, formatting and answer magnitude can act as shortcuts. A separate perturbation study would be needed to determine causal reliance on such cues.


---

# Resources, Reproduction and Disclosure

## 9. Measured resources

Local machine: AMD Ryzen 7 5800H with Radeon Graphics; Windows 11; Python 3.12.14. Four CPU threads were configured. A fresh prediction process took 0.397 s and the separate official scoring process took 0.183 s, totaling 0.580 s. These wall times include process startup, imports, loading, preprocessing and scoring. Processes were fresh but OS disk cache was not cleared. Reference Xeon Platinum 8457C eligibility remains unmeasured.

Inference assets total 25,929 bytes including configuration and uncompressed NPZ archive. Array payload is 24,772 bytes; both are far below 16 MiB. The final successful search took 51.52 seconds including input loading, feature generation, fitting and export checks. Two interrupted debugging attempts preceded it; their total end-to-end duration was not recorded, so this is not the complete development cost.

## 10. Reproduction and code version

Use the root README commands and Python 3.12. Inference requires only NumPy 2.3.5 and runs/checkpoint. Training requires the pinned requirements-training.txt. Run check_data.py first, then official predict.py and score_predictions.py. Saved experiments.json records all validation results; measurements.json records process timing; freeze.json records the selected settings and checkpoint SHA-256; source-freeze.json identifies code files before final testing. The frozen checkpoint reproduces 391/467 using the official scorer without retraining.

Behavioral tests cover fraction/decimal majority equivalence, invalid answers, empty/different-size banks, large numeric answers, candidate permutations, unseen IDs, checkpoint round trips and batch independence. All six tests pass. Inference reads passed bank/arrays rather than reference files or question IDs. Audit inputs can be supplied through the official runner without retraining.

Float32 MLP probabilities can saturate or tie and differ from exported raw-logit rankings. Exported logits are checked against scikit-learn raw logits (rtol=atol=2e-4), and selection uses actual exported predictions. Two MLP settings differed on 5 and 1 probability-ranked validation choices; this is recorded. The selected logistic export matches all validation choices.

## 11. AI assistance and sources

OpenAI Codex assisted substantively with design, code, execution, debugging, tests, documentation and this report. The student must review and understand the artifacts. No claim of unaided authorship is made. See AI_DISCLOSURE.md. Source package: github.com/xudongwu-0/DASE7506-MP2. Questions derive from github.com/openai/grade-school-math (MIT); supplied traces/representations derive from Qwen3-0.6B as documented in the retained data NOTICE. Standard estimators: scikit-learn.org/stable/modules/linear_model.html and scikit-learn.org/stable/modules/neural_networks_supervised.html.
