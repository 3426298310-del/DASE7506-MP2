import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import submission


def row(texts):
    return {'id': 'unseen', 'question': 'What is one half?', 'candidates': [{'index': i, 'text': t} for i, t in enumerate(texts)]}


class SelectorTests(unittest.TestCase):
    def test_numeric_equivalence_majority(self):
        bank = [row(['#### 9', '#### 1/2', '#### 0.5', '#### 2'])]
        self.assertEqual(submission.predict(bank, {'embeddings': np.zeros((1,4,1024))}, None), [1])

    def test_invalid_answers_do_not_form_majority(self):
        bank = [row(['no answer', 'missing', '#### 5', '#### 6'])]
        self.assertEqual(submission.predict(bank, {'embeddings': np.zeros((1,4,1024))}, None), [2])

    def test_empty_and_dynamic_bank(self):
        self.assertEqual(submission.predict([], {'embeddings': np.zeros((0,4,1024))}, None), [])
        b = row(['#### 3']*4)
        self.assertEqual(submission.predict([b]*3, {'embeddings': np.zeros((3,4,1024))}, None), [0,0,0])

    def test_large_valid_answer_features_are_finite(self):
        b=row(['#### '+ '9'*400,'#### 1','missing','#### 2'])
        x=submission.features([b],{'embeddings':np.zeros((1,4,1024))},'text')
        self.assertTrue(np.isfinite(x).all())

    def test_candidate_permutation_moves_scores_with_candidate(self):
        b=row(['#### 1','#### 2','#### 2','#### 3'])
        a={'embeddings':np.random.default_rng(7).normal(size=(1,4,1024))}
        perm=[2,0,3,1]
        bp=dict(b,candidates=[b['candidates'][j] for j in perm],id='different')
        expected=submission.features([b],a).reshape(1,4,-1)[:,perm]
        actual=submission.features([bp],{'embeddings':a['embeddings'][:,perm]}).reshape(1,4,-1)
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_checkpoint_roundtrip_unseen_ids_and_batch_independence(self):
        b=row(['#### 1','#### 2','#### 2','#### 3'])
        a={'embeddings':np.zeros((1,4,1024),dtype='float32')}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            w=np.zeros((16,1),dtype='float32'); w[0]=1
            np.savez(p/'weights.npz',mean=np.zeros(16),scale=np.ones(16),w0=w,b0=np.zeros(1))
            (p/'config.json').write_text(json.dumps({'mode':'text','layers':1}))
            self.assertEqual(submission.predict([b],a,p),[1])
            bp=dict(b,id='new-id')
            self.assertEqual(submission.predict([b,bp],{'embeddings':np.zeros((2,4,1024))},p),[1,1])


if __name__ == '__main__':
    unittest.main()
