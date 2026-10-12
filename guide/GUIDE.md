# MP2 — Selecting Reliable LLM Reasoning

DASE7506 · Individual coursework · 10% of the course grade  
**Submission deadline and leaderboard opening: to be announced.**

A language model gives four solutions to the same maths problem. They may sound convincing and still disagree. Build a method that chooses a correct final answer, and investigate what makes your choice reliable. Use learning, checking rules, or your own combination; see the [README](../code/README.md) for an example and research questions.

## 1. What we provide

- **guide:** this assignment guide.
- **code:** data, fixed text representations, an empty submission interface and evaluation tools. The [code README](../code/README.md) explains installation, formats and commands. Write your own method and training code.
- **Data:** 4,755 training, 798 validation and 467 test questions. Each group has a correct final answer; test groups contain different valid answers. Correct answers may have flawed reasoning.

## 2. Ranking and assessment

- **Leaderboard:** complete-test selection accuracy, **higher is better**. Report the official scorer's `accuracy` percentage.
- **Experimental evidence:** compare your initial and improved methods under the same protocol; ablate a key design choice.
- **Analysis matters:** submit a report of **at most 10 pages** explaining decisions, comparisons, errors, failures and accuracy–compute trade-offs. Ranking is provisional until reproduction and review; a separate audit bank checks generalization.

## 3. Constraints

- **Data:** fit only on supplied training data; training-only augmentation is allowed. Select on validation. Test references are for final scoring/reproduction only. Keep data, parser and scorer unchanged; no external datasets or additional pretrained weights.
- **Evaluation:** select a supplied candidate offline, without new LLM generation or API calls. Accept new bank inputs without retraining. Use IDs only for alignment; no answer lookups or cached predictions.
- **Resources:** complete CPU evaluation **≤30 seconds** on a four-thread Xeon Platinum 8457C; additional inference assets **≤16 MiB**, excluding supplied data. These limits discourage large lookup indexes. Train on CPU or one GPU no stronger than RTX 4090; GPUs are not provided.
- **AI is allowed:** understand your implementation, acknowledge reuse and disclose substantive AI help. Architecture, features, training and fusion are open.

## 4. Submission and peer review

- **Submit:** use MP2 on the [course website](https://xudongwu-0.github.io/courses/dase7506/) when it opens. Early entries need your student ID and score; your final entry needs **code and checkpoint links**. Complete the generated GitHub issue; use the same account for updates.
- **Artifacts:** include your report and exact installation/training/evaluation commands. The checkpoint must reproduce results **without retraining**; training-free methods may submit a configuration.
- **Peer review:** instructors release links and frozen scores for **seven days**. File a **Peer Review Report** with submission number and reproduced score, or email **wu.xudong@connect.hku.hk**. Confirmed discrepancies can earn rewards; reports alone never change grades.
- <span style="color: #b42318;"><strong>We use automated checkpoint evaluation scripts and instructor spot checks to verify scores. Confirmed serious score misreporting will be formally recorded.</strong></span>
