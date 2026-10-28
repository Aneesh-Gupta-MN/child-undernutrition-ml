"""Synthetic software fixtures only; not NFHS observations or model evidence."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import REQUIRED, cohort_audit, run_audit


def fixture():
    df = pd.DataFrame({c: [1.0] * 8 for c in REQUIRED})
    df['caseid'] = [f'SYNTHETIC_{i}' for i in range(8)]
    df['b19'] = [6, 59, 20, 20, 20, 20, 5, 60]
    df['hw1'] = df.b19
    df['hw70'] = [-201, -200, 9996, float('nan'), -600, -601, -300, -300]
    return df


class AuditTests(unittest.TestCase):
    def test_age_boundaries_special_codes_and_strict_threshold(self):
        result = cohort_audit(fixture())
        self.assertEqual(result['living_age_6_59'], 6)
        self.assertEqual(result['provisional_usable_outcomes'], 3)
        self.assertEqual(result['provisional_stunting_count'], 2)
        self.assertEqual(result['outcome_system_missing_in_age_group'], 1)
        self.assertEqual(result['outcome_special_or_outside_range_in_age_group'], 2)

    def test_dead_children_and_duplicate_keys(self):
        df = fixture()
        df.loc[0, 'b5'] = 0
        df.loc[1, 'caseid'] = df.loc[0, 'caseid']
        result = cohort_audit(df)
        self.assertEqual(result['provisional_stunting_count'], 1)
        self.assertEqual(result['duplicate_child_key_rows'], 2)

    def test_missing_columns_fail(self):
        with self.assertRaises(ValueError):
            cohort_audit(fixture().drop(columns='hw70'))

    def test_real_stata_io_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'synthetic.dta'
            output = Path(temp) / 'report'
            pyreadstat.write_dta(fixture(), str(source), variable_value_labels={'b4': {1: 'male', 2: 'female'}})
            original = source.read_bytes()
            result = run_audit(source, output)
            self.assertEqual(result['counts']['provisional_usable_outcomes'], 3)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((output / 'data-quality-report.md').is_file())
            self.assertTrue((output / 'audit.json').is_file())
            with self.assertRaises(ValueError):
                run_audit(source, output)


if __name__ == '__main__':
    unittest.main()
