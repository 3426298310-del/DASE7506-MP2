"""Generate evidence-grounded report from frozen experiment artifacts."""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from common import load_inputs, parse_answer, DATA, read_jsonl
from submission import model_scores


def wilson(k,n):
    z=1.96; p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(c-h),100*(c+h)]


def main():
    root=Path('..'); run=Path('runs')
    ex=json.loads((run/'experiments.json').read_text()); freeze=json.loads((run/'freeze.json').read_text()); measurement=json.loads((run/'measurements.json').read_text())
    v,va=load_inputs('validation'); refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(v,refs)])
    meta=json.loads((run/'checkpoint/config.json').read_text())
    with np.load(run/'checkpoint/weights.npz') as bundle: scores=model_scores(v,va,bundle,meta)
    choice=scores.argmax(axis=1); correct=y[np.arange(len(v)),choice]
    ordered=np.sort(scores,axis=1); margin=ordered[:,-1]-ordered[:,-2]
    quartiles=[]
    for i,idx in enumerate(np.array_split(np.argsort(margin,kind='stable'),4)):
        quartiles.append([f'Q{i+1}',len(idx),f'{margin[idx].min():.3f} - {margin[idx].max():.3f}',f'{100*correct[idx].mean():.2f}%'])
    unique=np.array([len(set(a for a in [parse_answer(c['text']) for c in b['candidates']] if a is not None)) for b in v])
    group=[]
    for name,mask in [('one valid answer value',unique==1),('two or more answer values',unique>=2)]:
        group.append([name,int(mask.sum()),f'{100*correct[mask].mean():.2f}%'])
    cases=[]
    for idx in np.flatnonzero(~correct)[:3]:
        b=v[idx]; r=refs[idx]; j=int(choice[idx])
        cases.append({'question':b['question'],'selected_index':j,'selected_answer':parse_answer(b['candidates'][j]['text']),'reference_answer':parse_answer(r['answer']),'margin':float(margin[idx])})
    test=next(r for r in measurement['results'] if r['split']=='test' and r['method']=='model')
    analysis={'validation_margin_quartiles':quartiles,'validation_answer_diversity':group,'validation_failure_examples':cases,'test_wilson_95_percent':wilson(test['correct'],test['n'])}
    (run/'analysis.json').write_text(json.dumps(analysis,indent=2))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=22,leading=26,spaceAfter=18,textColor=colors.HexColor('#17365d')))
    story=[]; md=[]
    def title(s): story.append(Paragraph(escape(s),styles['TitleCustom'])); md.append('# '+s+'\n')
    def heading(s): story.append(Paragraph(escape(s),styles['Heading2'])); md.append('## '+s+'\n')
    def p(s): story.append(Paragraph(escape(s),styles['BodyCustom'])); md.append(s+'\n')
    def table(headers,rows,widths=None):
        cells=[[Paragraph(escape(str(x)),styles['BodyCustom']) for x in r] for r in [headers]+rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff7')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#ccd4df')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),1)]))
        story.append(t); story.append(Spacer(1,10))
        md.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |')
        md.extend('| '+' | '.join(map(str,r))+' |' for r in rows); md.append('')
    def page(): story.append(PageBreak()); md.append('\n---\n')
    title('Compact Selection of Reliable LLM Answers')
    p('DASE7506 Mini-project 2 | Experimental report | 22 October 2026')
    p('Author: Wang Zheng | Student ID: 3036801785')
    p('Experiments completed on 13 October 2026; checkpoint reproduction rechecked on 22 October 2026. Substantive AI assistance is disclosed in Section 11.')
    heading('1. Research question and hypothesis')
    p('Can a small learned candidate scorer outperform answer majority voting under a strict CPU and asset budget? The hypothesis is that fixed trace embeddings and inexpensive text statistics provide complementary evidence. A second hypothesis is that explicit agreement between candidate answers improves selection; this is tested rather than assumed.')
    heading('2. Task and protocol')
    p('The bank provides four candidate solutions for each question and fixed 1024-dimensional embeddings. We choose an index from 0 to 3. Any candidate with the correct final numeric answer receives credit. The metric does not certify intermediate reasoning. Training/validation/test contain 4,755/798/467 questions. Each question has at least one correct candidate; the test bank additionally requires at least two different valid parsed answers, unlike easier groups present in training and validation.')
    p('Only supplied training references supervise fitting. Validation selects hyperparameters. The checkpoint was frozen and hashed before test evaluation. No external datasets, pretrained weights, answer lookup tables, or LLM generation/API calls are used. Official data, numeric parser and scorer remain unchanged.')
    heading('3. Main result')
    table(['Method','Validation','Test'],[['Majority vote','688/798 (86.22%)','323/467 (69.16%)'],['Frozen linear scorer','739/798 (92.61%)','391/467 (83.73%)']],[155,175,175])
    p('The frozen method improves test accuracy by 14.56 percentage points over majority voting. The test result was not used to revise the model. Results apply to this course bank; they are not a general guarantee of reasoning reliability.')
    page()
    title('Method and Experimental Design')
    heading('4. Candidate scoring')
    p('The baseline uses the official parser to canonicalize equivalent fractions and decimals, counts valid answer agreement, and selects the first maximum. Invalid parses do not vote together. For the learned model, each candidate receives a binary training label indicating whether its parsed final answer equals the training reference. Multiple candidates can be positive.')
    p('Features concatenate the supplied embedding, its difference from the four-candidate mean, and 16 inexpensive statistics. These include valid answer agreement, parse validity, logarithmic character/word/line/number/equality/operator counts, question-number coverage, logarithmic absolute final-answer magnitude, negative/fraction flags, answer-marker flags and counts of revision/conclusion terms. They are statistical cues, not a symbolic verification of arithmetic or problem conditions.')
    p('Each coordinate is standardized using training statistics only. Logistic regression minimizes binary cross entropy with L2 regularization. At inference the highest raw logit wins. The selected C is 0.01. Explicit answer agreement is zeroed in the selected no_consensus configuration. The model contains 2,064 feature coordinates, their means/scales and one linear weight vector.')
    p('For a linear scorer, subtracting the same within-question embedding mean contributes only a shared score offset; it cancels in candidate ranking. Therefore the centered block is algebraically redundant at inference, and these experiments do not establish a benefit from comparative embedding features. The redundant block changes the effective training regularization and is retained to reproduce the frozen checkpoint.')
    heading('5. Controlled comparisons and search')
    p('Four feature settings are compared: embeddings alone; text statistics alone; full features without explicit agreement; and full features with agreement. Each uses C in {0.01, 0.1, 1.0}. Full features additionally use a 32-unit ReLU MLP with alpha in {0.1, 1.0, 10.0}. All 15 fitted models use seed 42. The highest exported-model validation accuracy wins; exact ties use the first grid entry. The baseline was recorded but not a candidate in this grid; all fitted models outperform it here.')
    p('MLP uses training-only internal early stopping: 10% candidate-level holdout, patience 10, maximum 120 epochs, batch 256. Related candidates can share questions across this internal split, making that stopping estimate optimistic; external validation remains question-disjoint. Logistic regression maximum iterations is 400. Training runs on CPU with four library threads.')
    page()
    title('Validation Search and Ablations')
    table(['Configuration','Validation accuracy','Fit/export seconds'],[[r['name'],f"{r['validation_accuracy']:.2f}%",f"{r['fit_seconds']:.2f}"] for r in ex],[280,110,115])
    heading('6. Interpretation')
    p('At fixed C=0.01, full features without agreement achieve 92.61% versus 91.85% with agreement: explicit consensus costs six correct validation selections (0.75 percentage points). This is a small observed difference, not evidence that agreement is universally harmful. Embedding-only and text-only selected models each reach 89.35%; combining their evidence gives a stronger result here.')
    p('The best MLP reaches 92.48%, one validation question below the selected linear model, with substantially larger arrays. The experiment supports using the compact linear method within this search, rather than claiming that nonlinear modeling is intrinsically inferior. The number of searched models can bias the best validation estimate upward; there is no repeated-seed or external-dataset evaluation.')
    page()
    title('Reliability and Error Analysis')
    heading('7. Dataset difficulty and uncertainty')
    table(['Validation subgroup','Questions','Selected accuracy'],group,[280,100,125])
    p('The observed test accuracy is 8.88 percentage points below validation. Differences in answer diversity are one plausible contributor because the test protocol excludes easy single-answer-value groups. This is not a causal attribution: other distribution differences and validation selection optimism can also contribute.')
    table(['Logit-margin quartile','Questions','Margin range','Accuracy'],quartiles,[145,75,160,125])
    p('Accuracy does not increase monotonically with margin: Q1 reaches 93.50% while Q4 reaches 92.46%. Thus these results do not support treating larger margins as greater reliability. Easy groups with agreeing answers can have low or zero margins, confounding this comparison. Margins are uncalibrated ranking evidence, not probabilities of correctness. Calibration and abstention would need a separate validation study.')
    ci=analysis['test_wilson_95_percent']
    p(f'The 95% Wilson interval for the test proportion is {ci[0]:.2f}% to {ci[1]:.2f}%, if questions are treated as independent draws. This interval does not remove bank-selection bias or account for a future audit distribution. All 467 public test questions were scored.')
    heading('8. Concrete validation failures')
    for i,c in enumerate(cases,1):
        p(f"Example {i}: {c['question']}")
        p(f"Selected candidate {c['selected_index']}: {c['selected_answer']}; reference: {c['reference_answer']}; score margin: {c['margin']:.3f}.")
    p('These are the first three incorrect validation questions in bank order, not hand-picked to support a causal story. Final-answer mismatch is established; the model itself does not identify the flawed reasoning step. Text length, formatting and answer magnitude can act as shortcuts. A separate perturbation study would be needed to determine causal reliance on such cues.')
    page()
    title('Resources, Reproduction and Disclosure')
    heading('9. Measured resources')
    p(f"Local machine: {measurement['cpu']}; Windows 11; Python 3.12.14. Four CPU threads were configured. A fresh prediction process took {test['wall_prediction_seconds']:.3f} s and the separate official scoring process took {test['wall_scoring_seconds']:.3f} s, totaling {test['wall_total_seconds']:.3f} s. These wall times include process startup, imports, loading, preprocessing and scoring. Processes were fresh but OS disk cache was not cleared. Reference Xeon Platinum 8457C eligibility remains unmeasured.")
    p(f"Inference assets total {measurement['checkpoint_file_bytes']:,} bytes including configuration and uncompressed NPZ archive. Array payload is 24,772 bytes; both are far below 16 MiB. The final successful search took {freeze['training_total_seconds']:.2f} seconds including input loading, feature generation, fitting and export checks. Two interrupted debugging attempts preceded it; their total end-to-end duration was not recorded, so this is not the complete development cost.")
    heading('10. Reproduction and code version')
    p('Use the root README commands and Python 3.12. Inference requires only NumPy 2.3.5 and runs/checkpoint. Training requires the pinned requirements-training.txt. Run check_data.py first, then official predict.py and score_predictions.py. Saved experiments.json records all validation results; measurements.json records process timing; freeze.json records the selected settings and checkpoint SHA-256; source-freeze.json identifies code files before final testing. The frozen checkpoint reproduces 391/467 using the official scorer without retraining.')
    p('Behavioral tests cover fraction/decimal majority equivalence, invalid answers, empty/different-size banks, large numeric answers, candidate permutations, unseen IDs, checkpoint round trips and batch independence. All six tests pass. Inference reads passed bank/arrays rather than reference files or question IDs. Audit inputs can be supplied through the official runner without retraining.')
    p('Float32 MLP probabilities can saturate or tie and differ from exported raw-logit rankings. Exported logits are checked against scikit-learn raw logits (rtol=atol=2e-4), and selection uses actual exported predictions. Two MLP settings differed on 5 and 1 probability-ranked validation choices; this is recorded. The selected logistic export matches all validation choices.')
    heading('11. AI assistance and sources')
    p('OpenAI Codex assisted substantively with design, code, execution, debugging, tests, documentation and this report. The student must review and understand the artifacts. No claim of unaided authorship is made. See AI_DISCLOSURE.md. Source package: github.com/xudongwu-0/DASE7506-MP2. Questions derive from github.com/openai/grade-school-math (MIT); supplied traces/representations derive from Qwen3-0.6B as documented in the retained data NOTICE. Standard estimators: scikit-learn.org/stable/modules/linear_model.html and scikit-learn.org/stable/modules/neural_networks_supervised.html.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.grey)
        canvas.drawString(45,25,'DASE7506 MP2 | AI-assisted experimental report')
        canvas.drawRightString(A4[0]-45,25,str(doc.page))
    SimpleDocTemplate(str(root/'report.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=40,bottomMargin=42,title='DASE7506 MP2 - Compact Selection of Reliable LLM Answers',author='Wang Zheng (3036801785)').build(story,onFirstPage=footer,onLaterPages=footer)
    (root/'report.md').write_text('\n'.join(md),encoding='utf-8')


if __name__=='__main__': main()
