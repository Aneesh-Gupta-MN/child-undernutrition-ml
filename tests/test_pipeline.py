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

    def test_csv_complete_training_and_prediction(self):
        path = self.root / 'numeric.csv'
        raw = self.raw.copy()
        raw.loc[raw.v113.eq(99), 'v113'] = np.nan
        raw.to_csv(path, index=False)
        original = path.read_bytes()
        output = self.root / 'csv-trained'
        summary = train(path, output, self.config)
        self.assertEqual(summary['source']['format'], 'csv')
        self.assertTrue(summary['config']['synthetic'])
        self.assertEqual(summary['selected_model'], self.summary['selected_model'])
        self.assertEqual(summary['audit']['eligible'], len(raw))
        self.assertEqual(summary['audit']['predictor_missing_counts']['v113'], int(self.raw.v113.eq(99).sum()))
        bundle = joblib.load(output / 'model.joblib')
        examples = json.loads((output / 'prediction-example.json').read_text())
        self.assertEqual(len(predict_records(bundle, examples)), 1)
        self.assertIn('CSV contains no value-label codebook', (output / 'report.md').read_text())
        self.assertEqual(path.read_bytes(), original)

    def test_derived_india_kr_csv_mapping(self):
        raw = self.raw.copy()
        derived = pd.DataFrame({
            'child_age_months': raw.b19,
            'child_sex': raw.b4,
            'birth_order': raw.bord,
            'mother_education': raw.v106,
            'household_size': raw.v136,
            'drinking_water_source': raw.v113,
            'toilet_type': raw.v116,
            'residence_type': raw.v025,
            'state_code': raw.v024,
            'psu': raw.v001,
            'strata': raw.v022,
            'sample_weight_x1000000': raw.v005,
            'height_for_age_zscore_x100': raw.hw70,
        })
        path = self.root / 'india_kr.csv'
        derived.to_csv(path, index=False)
        data, metadata = read_dataset(path)
        cohort, _, audit = clean_cohort(data, metadata)
        self.assertEqual(metadata['source_kind'], 'derived_india_kr_csv')
        self.assertEqual(len(cohort), len(derived))
        self.assertEqual(cohort.household.nunique(), cohort.psu.nunique())
        self.assertNotIn('living_with_interviewed_mother', audit['cohort_flow'])
        self.assertTrue(any('household identifiers' in note for note in audit['cautions']))

    def test_csv_bom_headers_delimiters_and_identifiers(self):
        raw = self.raw.iloc[:4].copy()
        raw['caseid'] = ['00001', '00001', '00002', '00002']
        raw['irrelevant'] = 'ignored'
        raw.columns = ['  ' + col.upper() + '  ' for col in raw.columns]
        for delimiter in [',', ';', '\t']:
            path = self.root / 'header-variation.csv'
            raw.to_csv(path, index=False, sep=delimiter, encoding='utf-8-sig')
            result, _ = read_dataset(path)
            self.assertEqual(result.caseid.tolist(), ['00001', '00001', '00002', '00002'])
            self.assertNotIn('irrelevant', result.columns)
            self.assertEqual(result.b19.tolist(), self.raw.b19.iloc[:4].tolist())

    def test_csv_rejects_bad_schema_labels_and_rows(self):
        path = self.root / 'invalid.csv'
        self.raw.drop(columns='v005').to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'required NFHS columns: v005'):
            read_dataset(path)
        raw = self.raw.iloc[:4].copy()
        raw['b4'] = 'male'
        raw.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'b4 must contain original numeric'):
            read_dataset(path)
        text = self.raw.iloc[:4].to_csv(index=False)
        lines = text.splitlines()
        lines[0] += ',B19'
        path.write_text('\n'.join(lines))
        with self.assertRaisesRegex(ValueError, 'Duplicate CSV column'):
            read_dataset(path)
        lines = text.splitlines()
        lines[2] += ',extra'
        path.write_text('\n'.join(lines))
        with self.assertRaisesRegex(ValueError, 'fields; expected'):
            read_dataset(path)
        path.write_text('')
        with self.assertRaisesRegex(ValueError, 'CSV is empty'):
            read_dataset(path)


if __name__ == '__main__':
    unittest.main()
