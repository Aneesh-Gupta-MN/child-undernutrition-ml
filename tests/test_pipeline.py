"""End-to-end and scientific-integrity checks using synthetic records only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_data import create_demo
from pipeline import (Config, clean_cohort, read_dataset, split_data, train, make_models,
                      prepare_features, probabilities, choose_threshold)
from predict import predict_records


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = create_demo(cls.root / 'SYNTHETIC_KR.dta')
        cls.raw, cls.meta = read_dataset(cls.source)
        cls.cohort, cls.x, cls.audit = clean_cohort(cls.raw, cls.meta)
        cls.config = Config(bootstrap_repeats=20)
        cls.output = cls.root / 'trained'
        cls.summary = train(cls.source, cls.output, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_group_and_state_separation(self):
        splits, states = split_data(self.cohort, self.config)
        sets = [set(self.cohort.iloc[idx].household) for idx in splits.values()]
        for i, left in enumerate(sets):
            for right in sets[i+1:]:
                self.assertFalse(left & right)
        self.assertEqual(set(self.cohort.iloc[splits['test']].state), set(states))
        for name in ['fit', 'calibration', 'validation']:
            self.assertFalse(set(self.cohort.iloc[splits[name]].state) & set(states))
        splits2, _ = split_data(self.cohort, self.config)
        for name in splits:
            np.testing.assert_array_equal(splits[name], splits2[name])

    def test_exclusions_and_special_codes(self):
        raw = self.raw.copy()
        raw.loc[0, 'hw70'] = 9998
        raw.loc[1, 'hw70'] = np.nan
        raw.loc[2, 'hw70'] = -200
        raw.loc[3, 'hw70'] = -201
        raw.loc[4, 'b9'] = 1
        raw.loc[5, 'hw1'] = 100
        cohort, x, audit = clean_cohort(raw, self.meta)
        self.assertEqual(len(cohort), len(raw)-4)
        self.assertEqual(cohort.loc[cohort.caseid.eq(raw.loc[2, 'caseid']) & cohort.bidx.eq(raw.loc[2, 'bidx']), 'target'].iloc[0], 0)
        self.assertEqual(cohort.loc[cohort.caseid.eq(raw.loc[3, 'caseid']) & cohort.bidx.eq(raw.loc[3, 'bidx']), 'target'].iloc[0], 1)
        self.assertIn('__missing__', x.v113.unique())
        self.assertNotIn('99', x.v113.unique())
        self.assertEqual(audit['sequential_exclusions']['valid_height_for_age'], 2)

    def test_bad_data_fail_closed(self):
        raw = self.raw.copy()
        raw.loc[1, ['caseid', 'bidx']] = raw.loc[0, ['caseid', 'bidx']]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            clean_cohort(raw, self.meta)
        raw = self.raw.copy()
        raw.loc[0, 'v005'] = 0
        with self.assertRaisesRegex(ValueError, 'weights'):
            clean_cohort(raw, self.meta)
        raw = self.raw.copy()
        raw['hw70'] = raw.hw70 / 100
        with self.assertRaisesRegex(ValueError, 'recode|scaled'):
            clean_cohort(raw, self.meta)

    def test_bundle_has_no_raw_records_and_can_predict_after_extract(self):
        bundle = joblib.load(self.output / 'model.joblib')
        self.assertTrue(bundle['synthetic'])
        self.assertEqual(bundle['name'], self.summary['selected_model'])
        winner = min(self.summary['validation_metrics'], key=lambda k: self.summary['validation_metrics'][k]['brier_score'])
        self.assertEqual(bundle['name'], winner)
        examples = json.loads((self.output / 'prediction-example.json').read_text())
        expected = predict_records(bundle, examples)
        with zipfile.ZipFile(self.output / 'trained-model-package.zip') as archive:
            self.assertFalse(any(name.endswith('.dta') for name in archive.namelist()))
            extracted = self.root / 'extracted'
            archive.extractall(extracted)
        result = subprocess.run([sys.executable, 'predict.py', '--model', 'model.joblib', '--input', 'prediction-example.json', '--output', 'predictions.json'], cwd=extracted, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads((extracted / 'predictions.json').read_text())
        self.assertEqual(actual, expected)
        self.assertGreaterEqual(actual[0]['estimated_probability_current_stunting'], 0)
        self.assertLessEqual(actual[0]['estimated_probability_current_stunting'], 1)

    def test_unseen_category_and_age_validation(self):
        bundle = joblib.load(self.output / 'model.joblib')
        example = json.loads((self.output / 'prediction-example.json').read_text())
        example[0]['v113'] = 876
        self.assertEqual(len(predict_records(bundle, example)), 1)
        example[0]['b19'] = 60
        with self.assertRaisesRegex(ValueError, 'age'):
            predict_records(bundle, example)

    def test_pipeline_preprocessors_independent_and_training_only(self):
        models = make_models(42)
        a = models['logistic_regression'].named_steps['preprocess'].transformers[0][1]
        b = models['xgboost'].named_steps['preprocess'].transformers[0][1]
        self.assertIsNot(a, b)
        splits, _ = split_data(self.cohort, self.config)
        fit = splits['fit']
        model = models['logistic_regression']
        x = self.x.copy()
        x.loc[splits['test'], 'v113'] = 'TEST_ONLY_CATEGORY'
        model.fit(x.iloc[fit], self.cohort.iloc[fit].target)
        categories = model.named_steps['preprocess'].named_transformers_['categorical'].categories_
        self.assertFalse(any('TEST_ONLY_CATEGORY' in values for values in categories))
        model.predict_proba(x.iloc[splits['test']])

    def test_zip_and_repeat_run_protection(self):
        path = self.root / 'input.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.write(self.source, '../../nested/SYNTHETIC_KR.dta')
        df, meta = read_dataset(path)
        self.assertEqual(len(df), len(self.raw))
        with self.assertRaisesRegex(ValueError, 'overwritten'):
            train(self.source, self.output, self.config)

    def test_threshold_weighted_sensitivity(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([.1, .8, .3, .6, .9])
        w = np.array([1, 1, 1, 2, 4])
        threshold = choose_threshold(y, p, w, .8)
        self.assertGreaterEqual(w[(y == 1) & (p >= threshold)].sum()/w[y == 1].sum(), .8)


if __name__ == '__main__':
    unittest.main()
