"""Read-only first audit of an approved, untransformed NFHS-5 Stata KR file."""
import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyreadstat

PREDICTORS = ['b19', 'b4', 'bord', 'v106', 'v136', 'v113', 'v116', 'v025']
DESIGN = ['v005', 'v021', 'v022', 'v024', 'v001', 'v002']
REQUIRED = PREDICTORS + DESIGN + ['caseid', 'bidx', 'b5', 'hw70', 'hw1']
OPTIONAL = ['b9', 'b16', 'v023', 'v003']


def cohort_audit(df):
    """Provisional counts only; this is not a finalized cleaning pipeline."""
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError('Required KR fields missing: ' + ', '.join(missing))
    age = pd.to_numeric(df.b19, errors='raise')
    haz_raw = pd.to_numeric(df.hw70, errors='raise')
    living = df.b5.eq(1)
    aged = living & age.between(6, 59)
    valid = haz_raw.between(-600, 600)
    cohort = aged & valid
    # Strict inequality is intentional: HAZ == -2 is not below -2.
    positive = cohort & haz_raw.lt(-200)
    return {
        'all_records': len(df),
        'living_children': int(living.sum()),
        'living_age_6_59': int(aged.sum()),
        'outcome_system_missing_in_age_group': int((aged & haz_raw.isna()).sum()),
        'outcome_special_or_outside_range_in_age_group': int((aged & haz_raw.notna() & ~valid).sum()),
        'provisional_usable_outcomes': int(cohort.sum()),
        'provisional_stunting_count': int(positive.sum()),
        'duplicate_child_key_rows': int(df.duplicated(['caseid', 'bidx'], keep=False).sum()),
        'missing_child_key_rows': int(df[['caseid', 'bidx']].isna().any(axis=1).sum()),
        'blank_caseid_rows': int(df.caseid.fillna('').astype(str).str.strip().eq('').sum()),
        'predictor_system_missing_in_provisional_cohort': {
            col: int(df.loc[cohort, col].isna().sum()) for col in PREDICTORS
        },
        'design_system_missing_in_provisional_cohort': {
            col: int(df.loc[cohort, col].isna().sum()) for col in DESIGN
        },
        'nonpositive_or_missing_weights_in_provisional_cohort': int(
            (cohort & (df.v005.isna() | df.v005.le(0))).sum()
        ),
        'age_disagreements_in_provisional_cohort': int(
            (cohort & df.hw1.notna() & df.hw1.ne(age)).sum()
        ),
    }


def run_audit(source, output):
    source, output = Path(source), Path(output)
    if not source.is_file() or source.suffix.lower() != '.dta':
        raise ValueError('Supply an extracted, approved Stata KR .dta file.')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory to preserve prior reports.')
    _, meta = pyreadstat.read_dta(str(source), metadataonly=True)
    columns = {col.lower(): col for col in meta.column_names}
    if len(columns) != len(meta.column_names):
        raise ValueError('Ambiguous column names after lowercasing.')
    missing = sorted(set(REQUIRED) - columns.keys())
    if missing:
        raise ValueError('This is not the expected KR schema. Missing: ' + ', '.join(missing))
    selected = [columns[col] for col in REQUIRED + OPTIONAL if col in columns]
    df, _ = pyreadstat.read_dta(str(source), usecols=selected, apply_value_formats=False)
    df.columns = df.columns.str.lower()
    counts = cohort_audit(df)
    with source.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = {
        'status': 'Provisional data audit; not model results',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_filename': source.name,
        'input_sha256': digest,
        'versions': {p: importlib.metadata.version(p) for p in ['pandas', 'pyreadstat']},
        'assumption': 'Original DHS hw70 stored as z-score times 100; verify against release documentation.',
        'counts': counts,
        'variable_labels': {col: meta.column_names_to_labels.get(columns[col]) for col in REQUIRED + OPTIONAL if col in columns},
        'value_labels': {col: meta.variable_value_labels.get(columns[col], {}) for col in REQUIRED + OPTIONAL if col in columns},
    }
    lines = ['# NFHS-5 initial data audit', '',
             '**Provisional audit only. No model has been trained.**', '',
             f'Input: `{source.name}`', '', f'SHA-256: `{digest}`', '',
             'Counts are unweighted. Cohort eligibility and stored units require review.', '',
             '| Check | Count |', '|---|---:|']
    for name, value in counts.items():
        if not isinstance(value, dict):
            lines.append(f'| {name.replace("_", " ")} | {value} |')
    lines += ['', '## System-missing predictor values in provisional cohort', '',
              '| Variable | Count |', '|---|---:|']
    for col, count in counts['predictor_system_missing_in_provisional_cohort'].items():
        lines.append(f'| {col} | {count} |')
    lines += ['', '## Review before modelling', '',
              '- Inspect labels and special codes in audit.json; system missingness is not total missingness.',
              '- Verify hw70 scaling, residence/co-residence eligibility and age at measurement.',
              '- Investigate duplicate or missing child keys, invalid weights and missing design fields.',
              '- Freeze geographic holdout and PSU keys before any preprocessing or fitting.',
              '- Do not interpret these counts as national prevalence or model performance.', '']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'data-quality-report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        run_audit(args.input, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Audit stopped: {exc}\n')
    print(f'Created local audit report in {args.output.resolve()}')
