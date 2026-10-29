"""Reproducible NFHS-5 KR training. No network calls; raw files stay unchanged."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import warnings
import zipfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from xgboost import XGBClassifier

FEATURES = ['b19', 'b4', 'bord', 'v106', 'v136', 'v113', 'v116', 'v025']
NUMERIC = ['b19', 'bord', 'v136']
CATEGORICAL = [x for x in FEATURES if x not in NUMERIC]
DESIGN = ['v005', 'v021', 'v022', 'v024', 'v001', 'v002']
REQUIRED = FEATURES + DESIGN + ['caseid', 'bidx', 'b5', 'b9', 'hw70', 'hw1']
DERIVED_INDIA_KR_COLUMNS = {
    'child_age_months': 'b19', 'child_sex': 'b4', 'birth_order': 'bord',
    'mother_education': 'v106', 'household_size': 'v136',
    'drinking_water_source': 'v113', 'toilet_type': 'v116',
    'residence_type': 'v025', 'sample_weight_x1000000': 'v005',
    'psu': 'v001', 'strata': 'v022', 'state_code': 'v024',
    'height_for_age_zscore_x100': 'hw70',
}
MISSING_LABEL = re.compile(r"don't know|do not know|unknown|missing|not applicable|refus|no response|inconsistent|flagged", re.I)
ALLOWED = {'b4': [1, 2], 'v106': [0, 1, 2, 3], 'v025': [1, 2]}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    seed: int = 42
    split_mode: str = 'geographic'
    test_states: tuple[int, ...] = ()
    test_fraction: float = .2
    validation_fraction: float = .2
    calibration_fraction: float = .2
    target_sensitivity: float = .8
    bootstrap_repeats: int = 200
    synthetic: bool = False

    def validate(self):
        if self.split_mode not in ('geographic', 'grouped'):
            raise ValueError('Choose geographic or grouped validation.')
        for fraction in (self.test_fraction, self.validation_fraction, self.calibration_fraction):
            if not .1 <= fraction <= .4:
                raise ValueError('Split fractions must be between 0.1 and 0.4.')
        if not .5 <= self.target_sensitivity <= .99:
            raise ValueError('Target sensitivity must be between 0.5 and 0.99.')
        if not 20 <= self.bootstrap_repeats <= 1000:
            raise ValueError('Use 20–1000 bootstrap repetitions.')


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_dataset(source):
    """Read a numeric-code KR CSV, original Stata file, or KR Stata ZIP."""
    source = Path(source)
    if not source.is_file():
        raise ValueError('The uploaded file is missing.')
    with source.open('rb') as handle:
        fingerprint = hashlib.file_digest(handle, 'sha256').hexdigest()
    if source.suffix.lower() == '.csv':
        return read_csv_dataset(source, fingerprint)
    with tempfile.TemporaryDirectory(prefix='nfhs-read-') as temp:
        data_path = source
        if source.suffix.lower() == '.zip':
            with zipfile.ZipFile(source) as archive:
                candidates = [i for i in archive.infolist() if not i.is_dir() and
                              i.filename.lower().endswith('.dta') and 'kr' in Path(i.filename).name.lower()]
                if len(candidates) != 1:
                    raise ValueError('ZIP must contain exactly one Children’s Recode (KR) .dta file. Upload an extracted KR file instead.')
                member = candidates[0]
                if member.file_size > 2_000_000_000:
                    raise ValueError('Expanded dataset exceeds the 2 GB limit.')
                # Copy to a fixed name, never extract an untrusted archive path.
                data_path = Path(temp) / 'children.dta'
                with archive.open(member) as src, data_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        elif source.suffix.lower() != '.dta':
            raise ValueError('Upload a CSV, Stata .dta file, or KR Stata ZIP archive.')
        _, meta = pyreadstat.read_dta(str(data_path), metadataonly=True)
        names = {c.lower(): c for c in meta.column_names}
        if len(names) != len(meta.column_names):
            raise ValueError('Duplicate column names after case normalization.')
        absent = sorted(set(REQUIRED) - names.keys())
        if absent:
            raise ValueError('Not a supported KR file. Required columns missing: ' + ', '.join(absent))
        data, _ = pyreadstat.read_dta(str(data_path), usecols=[names[c] for c in REQUIRED], apply_value_formats=False)
        data.columns = data.columns.str.lower()
    metadata = {
        'filename': source.name,
        'sha256': fingerprint,
        'file_label': meta.file_label,
        'labels': {c: meta.column_names_to_labels.get(names[c]) for c in REQUIRED},
        'value_labels': {c: {str(k): v for k, v in meta.variable_value_labels.get(names[c], {}).items()} for c in REQUIRED},
    }
    return data, metadata


def read_csv_dataset(source, fingerprint):
    """CSV has no Stata value labels: blank missing responses before export."""
    try:
        with source.open('r', encoding='utf-8-sig', newline='') as handle:
            sample = handle.read(65536)
            if not sample.strip():
                raise ValueError('The CSV is empty.')
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except csv.Error:
                dialect = csv.excel
            handle.seek(0)
            reader = csv.reader(handle, dialect, strict=True)
            headers = next(reader)
            normalized = [name.strip().lower() for name in headers]
            if len(set(normalized)) != len(normalized):
                raise ValueError('Duplicate CSV column names after trimming spaces and normalizing case.')
            source_kind = 'canonical_csv'
            absent = sorted(set(REQUIRED) - set(normalized))
            derived_absent = sorted(set(DERIVED_INDIA_KR_COLUMNS) - set(normalized))
            if absent and not derived_absent:
                source_kind = 'derived_india_kr_csv'
            elif absent:
                raise ValueError('CSV is missing required NFHS columns: ' + ', '.join(absent) +
                                 '. This app also supports the supplied derived India KR format; required columns missing from that format: ' + ', '.join(derived_absent) + '.')
            for row in reader:
                if row and len(row) != len(headers):
                    raise ValueError(f'CSV row ending at line {reader.line_num} has {len(row)} fields; expected {len(headers)}. Check delimiters and quoting.')
        names = dict(zip(normalized, headers))
        requested = REQUIRED if source_kind == 'canonical_csv' else list(DERIVED_INDIA_KR_COLUMNS)
        data = pd.read_csv(source, encoding='utf-8-sig', dialect=dialect,
                           usecols=[names[col] for col in requested], dtype=str,
                           keep_default_na=False, na_values=['', '.', 'NA', 'N/A', 'NaN', 'nan', 'NULL', 'null'])
        data.columns = [col.strip().lower() for col in data.columns]
        if source_kind == 'derived_india_kr_csv':
            data = data.rename(columns=DERIVED_INDIA_KR_COLUMNS)
            data['v021'] = data['v001']
            data['v002'] = 1
            # Generated IDs preserve one row per supplied record. The source lacks
            # household identifiers, so all split protection uses whole PSUs.
            data['caseid'] = pd.Series(np.arange(1, len(data) + 1), index=data.index).astype(str)
            data['bidx'] = 1
            data['b5'] = 1
            data['b9'] = np.nan
            data['hw1'] = data['b19']
        for col in REQUIRED:
            data[col] = data[col].astype('string').str.strip().replace('', np.nan)
            if col != 'caseid':
                try:
                    data[col] = pd.to_numeric(data[col], errors='raise')
                except ValueError as exc:
                    raise ValueError(f'CSV column {col} must contain original numeric NFHS codes or blank cells, not category labels.') from exc
    except (UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError('Cannot read the CSV. Use UTF-8 encoding and consistent comma, semicolon or tab delimiters.') from exc
    synthetic = data.caseid.fillna('').str.startswith('SYNTHETIC_').all() and len(data) > 0
    derived_notes = [
        'Derived India KR CSV: child-level identifiers, household identifiers, living-status, mother co-residence and separate measurement-age fields were not supplied.',
        'The analysis uses all supplied children aged 6–59 months with a plausible HAZ outcome. It cannot independently reproduce the raw KR eligibility restrictions.',
        'All children in the same PSU are kept together in fitting, calibration, validation and test partitions because household IDs are unavailable.',
    ] if source_kind == 'derived_india_kr_csv' else [
        'CSV contains no value-label codebook. Special missing responses must be blanked before upload. No special numeric missing codes are guessed.'
    ]
    metadata = {
        'filename': source.name, 'sha256': fingerprint, 'format': 'csv',
        'source_kind': source_kind,
        'file_label': 'SYNTHETIC SOFTWARE TEST ONLY' if synthetic else None,
        'labels': {col: None for col in REQUIRED},
        'value_labels': {col: {} for col in REQUIRED},
        'notes': derived_notes,
    }
    return data, metadata


def missing_codes(metadata):
    return {c: [float(k) for k, label in metadata.get('value_labels', {}).get(c, {}).items()
                if MISSING_LABEL.search(str(label))] for c in FEATURES}


def prepare_features(frame, codes):
    """Identical deterministic code handling for training and later inference."""
    absent = sorted(set(FEATURES) - set(frame.columns))
    if absent:
        raise ValueError('Missing predictor fields: ' + ', '.join(absent))
    x = frame[FEATURES].copy()
    for col in FEATURES:
        x[col] = pd.to_numeric(x[col], errors='raise').astype(float)
        x[col] = x[col].where(np.isfinite(x[col]))
        x[col] = x[col].mask(x[col].isin(codes.get(col, [])))
        x[col] = x[col].where(x[col].isna() | x[col].mod(1).eq(0))
        if col in ALLOWED:
            x[col] = x[col].where(x[col].isin(ALLOWED[col]))
        elif col in ('bord', 'v136'):
            x[col] = x[col].where(x[col].gt(0))
        elif col == 'b19':
            x[col] = x[col].where(x[col].between(6, 59))
        else:
            x[col] = x[col].where(x[col].ge(0))
    for col in CATEGORICAL:
        x[col] = x[col].map(lambda v: '__missing__' if pd.isna(v) else str(int(v)))
    return x


def clean_cohort(data, metadata):
    data = data.copy()
    if data.empty:
        raise ValueError('The dataset contains no records.')
    for col in REQUIRED:
        if col != 'caseid':
            data[col] = pd.to_numeric(data[col], errors='raise').astype(float)
            data[col] = data[col].where(np.isfinite(data[col]))
    if data[['caseid', 'bidx']].isna().any(axis=None):
        raise ValueError('Missing child identifiers. Fix the source before training.')
    data['caseid'] = data.caseid.astype(str).str.strip()
    if data.caseid.eq('').any() or data.duplicated(['caseid', 'bidx']).any():
        raise ValueError('Blank or duplicate child identifiers found; no records were silently deduplicated.')
    raw = data.hw70.dropna()
    if not raw.mod(1).eq(0).all():
        raise ValueError('hw70 is not an original integer recode. Supply raw HAZ × 100, not converted z-scores.')
    plausible = raw[raw.between(-600, 600)]
    if len(plausible) and plausible.abs().max() <= 6:
        raise ValueError('hw70 appears already scaled; this workflow requires original HAZ × 100.')
    flow = {'input_records': len(data)}
    if metadata.get('source_kind') == 'derived_india_kr_csv':
        rules = [
            ('age_6_to_59_months', data.b19.between(6, 59) & data.b19.mod(1).eq(0)),
            ('valid_height_for_age', data.hw70.between(-600, 600)),
        ]
    else:
        rules = [
            ('living', data.b5.eq(1)),
            ('age_6_to_59_months', data.b19.between(6, 59) & data.b19.mod(1).eq(0)),
            ('living_with_interviewed_mother', data.b9.eq(0)),
            ('valid_height_for_age', data.hw70.between(-600, 600)),
            ('matching_measurement_age', data.hw1.eq(data.b19)),
        ]
    keep = pd.Series(True, index=data.index)
    exclusions = {}
    for name, mask in rules:
        exclusions[name] = int((keep & ~mask).sum())
        keep &= mask
        flow[name] = int(keep.sum())
    cohort = data.loc[keep].copy().reset_index(drop=True)
    if len(cohort) < 400:
        raise ValueError(f'Only {len(cohort)} eligible children. This four-way research workflow requires at least 400; each partition also needs both outcomes.')
    if cohort[DESIGN].isna().any(axis=None) or (cohort.v005 <= 0).any():
        raise ValueError('Eligible records have missing design fields or nonpositive weights; review the data.')
    for col in DESIGN[1:]:
        if not cohort[col].mod(1).eq(0).all() or cohort[col].lt(0).any():
            raise ValueError(f'Invalid survey identifier: {col}.')
    cohort['state'] = cohort.v024.astype(int)
    cohort['psu'] = cohort.v024.astype(int).astype(str) + ':' + cohort.v001.astype(int).astype(str)
    cohort['household'] = cohort.psu if metadata.get('source_kind') == 'derived_india_kr_csv' else cohort.psu + ':' + cohort.v002.astype(int).astype(str)
    cohort['stratum'] = cohort.v024.astype(int).astype(str) + ':' + cohort.v022.astype(int).astype(str)
    # A state/cluster must map to a single PSU and stratum, not mixed survey releases.
    if (cohort.groupby('psu').v021.nunique() > 1).any() or (cohort.groupby('psu').stratum.nunique() > 1).any():
        raise ValueError('Conflicting PSU/stratum mappings. Review survey identifiers before splitting.')
    cohort['target'] = cohort.hw70.lt(-200).astype(int)
    cohort['weight'] = cohort.v005 / 1_000_000
    codes = missing_codes(metadata)
    x = prepare_features(cohort, codes)
    audit = {
        'cohort_flow': flow, 'sequential_exclusions': exclusions,
        'eligible': len(cohort), 'positives': int(cohort.target.sum()),
        'weighted_outcome_fraction': float(np.average(cohort.target, weights=cohort.weight)),
        'missing_codes_from_labels': codes,
        'predictor_missing_counts': {c: int(x[c].isna().sum() if c in NUMERIC else x[c].eq('__missing__').sum()) for c in FEATURES},
        'cautions': [
            'Restricted to children living with the interviewed mother; not national child prevalence.',
            'Measurement/interview age disagreements are excluded conservatively and reported.',
            'Units assume original DHS hw70 = HAZ × 100. Special codes are preserved in metadata.',
            'Review value labels and the sampling appendix for the actual NFHS release.',
        ],
    }
    audit['cautions'].extend(metadata.get('notes', []))
    return cohort, x, audit


def split_data(cohort, config):
    config.validate()
    all_idx = np.arange(len(cohort))
    states = sorted(cohort.state.unique().tolist())
    if config.split_mode == 'geographic':
        if len(states) < 3:
            raise ValueError('Geographic mode needs at least three states. For one-state data choose internal PSU validation, or supply a larger file.')
        test_states = list(config.test_states)
        if not test_states:
            count = max(1, int(math.ceil(len(states) * config.test_fraction)))
            test_states = np.random.default_rng(config.seed).choice(states, size=count, replace=False).tolist()
        if not set(test_states).issubset(states) or set(test_states) == set(states):
            raise ValueError('Test states must be present and leave development states available.')
        mask = cohort.state.isin(test_states).to_numpy()
        development, test = all_idx[~mask], all_idx[mask]
    else:
        if config.test_states:
            raise ValueError('State holdouts cannot be used in internal PSU mode.')
        development, test = next(GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed).split(all_idx, groups=cohort.psu))
        test_states = []
    def group_split(indices, fraction, seed):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(indices, groups=cohort.iloc[indices].psu))
        return indices[a], indices[b]
    fit_cal, validation = group_split(development, config.validation_fraction, config.seed + 1)
    fit, calibration = group_split(fit_cal, config.calibration_fraction, config.seed + 2)
    splits = {'fit': fit, 'calibration': calibration, 'validation': validation, 'test': test}
    seen_psu, seen_household = set(), set()
    for name, idx in splits.items():
        part = cohort.iloc[idx]
        if len(part) < 50 or part.target.value_counts().reindex([0, 1], fill_value=0).min() < 10 or part.psu.nunique() < 3:
            raise ValueError(f'{name} split is too small or lacks outcomes: need 50 records, 10 per outcome and 3 PSUs. Use more data or predeclare a different split.')
        psus, households = set(part.psu), set(part.household)
        if seen_psu & psus or seen_household & households:
            raise AssertionError('Group leakage detected.')
        seen_psu |= psus
        seen_household |= households
    if len(np.unique(np.concatenate(list(splits.values())))) != len(cohort):
        raise AssertionError('Split coverage failed.')
    return splits, sorted(test_states)


def make_models(seed):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)), ('scale', StandardScaler())])
    category = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    def preprocessor():
        return ColumnTransformer([('numeric', clone(numeric), NUMERIC), ('categorical', clone(category), CATEGORICAL)])
    age = Pipeline([('spline', SplineTransformer(n_knots=4, degree=3)), ('scale', StandardScaler())])
    return {
        'prevalence_baseline': Pipeline([('preprocess', ColumnTransformer([('age', 'passthrough', ['b19'])])), ('model', DummyClassifier(strategy='prior'))]),
        'age_baseline': Pipeline([('preprocess', ColumnTransformer([('age', age, ['b19'])])), ('model', LogisticRegression(C=1., max_iter=2000, solver='lbfgs'))]),
        'logistic_regression': Pipeline([('preprocess', preprocessor()), ('model', LogisticRegression(C=1., max_iter=2000, solver='lbfgs'))]),
        'xgboost': Pipeline([('preprocess', preprocessor()), ('model', XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.04, min_child_weight=20, subsample=.8, colsample_bytree=.8, reg_lambda=5., tree_method='hist', n_jobs=2, random_state=seed, eval_metric='logloss'))]),
    }


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1-p)).reshape(-1, 1)


def probabilities(bundle, x):
    p = bundle['pipeline'].predict_proba(x)[:, 1]
    if bundle['calibrator'] is not None:
        p = bundle['calibrator'].predict_proba(logit(p))[:, 1]
    return p


def choose_threshold(y, p, w, sensitivity):
    _, tpr, thresholds = roc_curve(y, p, sample_weight=w, drop_intermediate=False)
    choices = thresholds[(tpr >= sensitivity) & np.isfinite(thresholds)]
    return float(choices.max()) if len(choices) else 0.


def metrics(y, p, w, threshold):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1], sample_weight=w).ravel()
    def ratio(a, b):
        return float(a / b) if b else None
    both = len(np.unique(y)) == 2
    return {
        'roc_auc': float(roc_auc_score(y, p, sample_weight=w)) if both else None,
        'average_precision': float(average_precision_score(y, p, sample_weight=w)) if both else None,
        'brier_score': float(brier_score_loss(y, p, sample_weight=w)),
        'sensitivity': ratio(tp, tp + fn), 'specificity': ratio(tn, tn + fp),
        'precision': ratio(tp, tp + fp), 'accuracy': ratio(tp + tn, w.sum()),
        'weighted_outcome_fraction': float(np.average(y, weights=w)),
        'flagged_fraction': float(np.average(p >= threshold, weights=w)),
        'confusion_matrix_weighted': [[float(tn), float(fp)], [float(fn), float(tp)]],
    }


def cluster_intervals(part, p, threshold, repeats, seed):
    # Resample PSUs within observed state/stratum, retaining all children per PSU.
    psus = part[['psu', 'stratum']].drop_duplicates().reset_index(drop=True)
    lookup = {name: i for i, name in enumerate(psus.psu)}
    row_groups = part.psu.map(lookup).to_numpy()
    groups_by_stratum = [g.index.to_numpy() for _, g in psus.groupby('stratum')]
    samples = {k: [] for k in ['roc_auc', 'brier_score', 'sensitivity', 'specificity', 'precision']}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        counts = np.zeros(len(psus))
        for idx in groups_by_stratum:
            counts[idx] = rng.multinomial(len(idx), np.ones(len(idx))/len(idx))
        w = part.weight.to_numpy() * counts[row_groups]
        usable = w > 0
        if part.target.to_numpy()[usable].min() == part.target.to_numpy()[usable].max():
            continue
        result = metrics(part.target.to_numpy()[usable], np.asarray(p)[usable], w[usable], threshold)
        for key in samples:
            if result[key] is not None:
                samples[key].append(result[key])
    return {
        'method': 'Approximate percentile PSU bootstrap within observed state/stratum; conditional on this frozen model and holdout. Not full survey-design or model-selection uncertainty.',
        'requested_repetitions': repeats,
        'singleton_strata': sum(len(g) == 1 for g in groups_by_stratum),
        'intervals_95': {k: {'low': float(np.quantile(v, .025)), 'high': float(np.quantile(v, .975)), 'valid_repetitions': len(v)} if v else None for k, v in samples.items()},
    }


def save_charts(part, p, threshold, output, synthetic):
    y, w = part.target.to_numpy(), part.weight.to_numpy()
    title = 'SYNTHETIC SOFTWARE TEST — ' if synthetic else 'Held-out test — '
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
    ax[0].plot(fpr, tpr, color='#087f8c')
    ax[0].plot([0, 1], [0, 1], '--', color='gray')
    ax[0].set(xlabel='False-positive rate', ylabel='Sensitivity', title='ROC curve')
    precision, recall, _ = precision_recall_curve(y, p, sample_weight=w)
    ax[1].plot(recall, precision, color='#087f8c')
    ax[1].axhline(np.average(y, weights=w), ls='--', color='gray')
    ax[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall curve')
    bins = np.minimum((p * 10).astype(int), 9)
    xs, ys = [], []
    for b in range(10):
        mask = bins == b
        if mask.sum():
            xs.append(np.average(p[mask], weights=w[mask]))
            ys.append(np.average(y[mask], weights=w[mask]))
    ax[2].plot(xs, ys, 'o-', color='#087f8c')
    ax[2].plot([0, 1], [0, 1], '--', color='gray')
    ax[2].set(xlabel='Mean predicted probability', ylabel='Observed fraction', title='Weighted calibration', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(title + 'selected model')
    fig.savefig(output / 'evaluation.png', dpi=150)
    plt.close(fig)
    matrix = confusion_matrix(y, p >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
    ax.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', color='black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Below threshold', 'Above threshold'], yticklabels=['No stunting', 'Stunting'], xlabel='Model classification', ylabel='Observed outcome', title=title + 'raw child counts')
    fig.savefig(output / 'confusion-matrix.png', dpi=150)
    plt.close(fig)


def report_text(summary):
    demo = summary['config']['synthetic']
    lines = ['# ' + ('SYNTHETIC SOFTWARE TEST — ' if demo else '') + 'NFHS stunting model report', '',
             '**Synthetic data only; these numbers are not NFHS research findings.**' if demo else '**Research model of current stunting; not a clinical diagnosis or future forecast.**', '',
             f"Selected model: **{summary['selected_model']}**", '',
             'Selection used the lowest survey-weighted Brier score on validation data. The test set did not select the winner.', '',
             f"Input SHA-256: `{summary['source']['sha256']}`", '',
             f"Validation mode: {summary['config']['split_mode']}. Held-out state codes: {summary['test_states'] or 'none — internal PSU test only'}.", '',
             '## Cohort', '', '| Stage | Children remaining |', '|---|---:|']
    for key, value in summary['audit']['cohort_flow'].items():
        lines.append(f'| {key.replace("_", " ")} | {value} |')
    lines += ['', '## Independent partitions', '', '| Partition | Children | PSUs | Positives |', '|---|---:|---:|---:|']
    for key, value in summary['splits'].items():
        lines.append(f"| {key} | {value['children']} | {value['psus']} | {value['positives']} |")
    lines += ['', '## Validation comparison', '', '| Model | Weighted Brier ↓ | Weighted ROC-AUC ↑ |', '|---|---:|---:|']
    for name, result in summary['validation_metrics'].items():
        lines.append(f"| {name} | {result['brier_score']:.4f} | {result['roc_auc']:.4f} |")
    lines += ['', '## Selected-model test results', '', '| Metric | Weighted value |', '|---|---:|']
    for key, value in summary['test_metrics_weighted'].items():
        if not isinstance(value, list):
            lines.append(f'| {key.replace("_", " ")} | {value:.4f} |' if value is not None else f'| {key} | undefined |')
    lines += ['', f"Threshold: {summary['threshold']:.6f}, selected for at least {summary['config']['target_sensitivity']:.0%} sensitivity on validation data. This does not guarantee test or future sensitivity.", '',
              '## Approximate 95% intervals', '', '| Metric | Lower | Upper |', '|---|---:|---:|']
    for key, result in summary['uncertainty']['intervals_95'].items():
        if result:
            lines.append(f"| {key} | {result['low']:.4f} | {result['high']:.4f} |")
    lines += ['', summary['uncertainty']['method'], '',
              f"Singleton strata: {summary['uncertainty']['singleton_strata']}; these strata cannot contribute resampling variation.", '',
              '## Methods and limits', '',
              *['- ' + note for note in summary['source'].get('notes', [])],
              '- Four disjoint sets: model fitting, probability calibration, model/threshold selection, final testing. PSUs and households never cross sets.',
              '- Numeric imputation, scaling and categorical encoding are learned on the fitting set only. No oversampling. Survey weights enter model fitting, calibration and weighted evaluation.',
              '- Fixed model settings; no test-driven tuning. Logistic regression, an age-spline baseline, shallow XGBoost and a prevalence baseline are compared.',
              '- Each nonconstant model receives sigmoid calibration on the separate calibration set. The exported model is the exact evaluated model; it is not refit on the test set.',
              '- The outcome is valid original hw70 < −200. Age 6–59 months, living with the interviewed mother, matching measurement age, usable HAZ, and complete design fields are required.',
              '- Water/toilet codes remain nominal. Missing codes are identified from release labels; inspect the recorded mappings before using findings academically.',
              '- Model choice prioritizes probability accuracy (Brier score), not demonstrated field benefit. A baseline may win.',
              '- This is one prespecified split, not nested cross-validation. Geographic testing within one survey is not local prospective external validation.',
              '- Test subgroup estimates are descriptive; small subgroups are suppressed. No fairness or clinical-readiness certification is implied.',
              '- Do not use this output to withhold measurement, referral or care.', '',
              '## Reuse', '',
              'The download contains model.joblib, predict.py, pipeline.py, requirements.txt, input-schema.json, the report, charts and run metadata. Follow the bundled README. Load only a model artifact you created or trust.', '',
              '![Evaluation curves](evaluation.png)', '', '![Confusion matrix](confusion-matrix.png)', '']
    return '\n'.join(lines)


def train(source, output, config=None, progress=None):
    config = config or Config()
    config.validate()
    progress = progress or (lambda message: None)
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; existing runs are never overwritten.')
    progress('Reading the uploaded dataset')
    data, metadata = read_dataset(source)
    # Labelled fixtures can never masquerade as real data even if a caller forgets the flag.
    if 'synthetic' in str(metadata.get('file_label', '')).lower() and not config.synthetic:
        config = Config(**{**asdict(config), 'synthetic': True})
    progress('Checking eligibility, missing values and survey identifiers')
    cohort, x, audit = clean_cohort(data, metadata)
    splits, held_out = split_data(cohort, config)
    fit, cal, val, test = (splits[k] for k in ['fit', 'calibration', 'validation', 'test'])
    val_results, bundles = {}, {}
    for name, model in make_models(config.seed).items():
        progress('Training ' + name.replace('_', ' '))
        weights = cohort.iloc[fit].weight.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x.iloc[fit], cohort.iloc[fit].target, model__sample_weight=weights/weights.mean())
        calibrator = None
        if name != 'prevalence_baseline':
            calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
            cw = cohort.iloc[cal].weight.to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                calibrator.fit(logit(model.predict_proba(x.iloc[cal])[:, 1]), cohort.iloc[cal].target, sample_weight=cw/cw.mean())
        bundle = {'pipeline': model, 'calibrator': calibrator}
        p = probabilities(bundle, x.iloc[val])
        val_results[name] = metrics(cohort.iloc[val].target, p, cohort.iloc[val].weight, .5)
        bundles[name] = bundle
    winner = min(val_results, key=lambda name: val_results[name]['brier_score'])
    selected = bundles[winner]
    vp = probabilities(selected, x.iloc[val])
    threshold = choose_threshold(cohort.iloc[val].target, vp, cohort.iloc[val].weight, config.target_sensitivity)
    progress('Evaluating the selected model on untouched test data')
    test_part = cohort.iloc[test]
    tp = probabilities(selected, x.iloc[test])
    weighted = metrics(test_part.target, tp, test_part.weight, threshold)
    unweighted = metrics(test_part.target, tp, np.ones(len(test)), threshold)
    uncertainty = cluster_intervals(test_part, tp, threshold, config.bootstrap_repeats, config.seed + 3)
    subgroups = []
    for col, series in [('sex', test_part.b4), ('state', test_part.state), ('age_band', pd.cut(test_part.b19, bins=[5, 23, 35, 59], labels=['6–23', '24–35', '36–59']))]:
        for value in series.dropna().unique():
            mask = series.eq(value).to_numpy()
            sub = test_part.iloc[np.flatnonzero(mask)]
            if len(sub) < 30 or sub.target.value_counts().reindex([0, 1], fill_value=0).min() < 5:
                subgroups.append({'group': col, 'value': str(value), 'n': len(sub), 'status': 'suppressed: insufficient outcomes'})
            else:
                subgroups.append({'group': col, 'value': str(value), 'n': len(sub), 'metrics': metrics(sub.target, tp[mask], sub.weight, threshold)})
    versions = {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'pyreadstat', 'scikit-learn', 'xgboost', 'joblib', 'streamlit']}
    summary = {
        'schema_version': SCHEMA_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': asdict(config), 'source': metadata, 'versions': versions, 'audit': audit,
        'test_states': held_out, 'selected_model': winner, 'threshold': threshold,
        'validation_metrics': val_results, 'test_metrics_weighted': weighted,
        'test_metrics_unweighted': unweighted, 'uncertainty': uncertainty, 'subgroups': subgroups,
        'splits': {name: {'children': len(idx), 'psus': int(cohort.iloc[idx].psu.nunique()), 'positives': int(cohort.iloc[idx].target.sum()),
                          'psu_hashes': sorted(hashlib.sha256(g.encode()).hexdigest() for g in cohort.iloc[idx].psu.unique())} for name, idx in splits.items()},
    }
    selected.update({'schema_version': SCHEMA_VERSION, 'name': winner, 'threshold': threshold,
                     'missing_codes': missing_codes(metadata), 'features': FEATURES,
                     'synthetic': config.synthetic, 'versions': versions,
                     'input_sha256': metadata['sha256'], 'scope': 'Current stunting, ages 6–59 months; research use only'})
    progress('Saving the trained model and report')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a run only after all files have been written successfully.
    with tempfile.TemporaryDirectory(prefix='.training-', dir=output.parent) as temp:
        staging = Path(temp) / 'result'
        staging.mkdir()
        joblib.dump(selected, staging / 'model.joblib', compress=3)
        json_write(staging / 'metrics.json', summary)
        json_write(staging / 'input-schema.json', {'columns': FEATURES, 'numeric': NUMERIC, 'categorical': CATEGORICAL,
                   'value_labels': {c: metadata['value_labels'].get(c, {}) for c in FEATURES}, 'missing_codes': selected['missing_codes'],
                   'notes': 'Supply original DHS codes; null means missing. Age must be 6–59 months. Additional columns are ignored.'})
        json_write(staging / 'prediction-example.json', [{'b19': 24, 'b4': 1, 'bord': 2, 'v106': 2, 'v136': 5, 'v113': None, 'v116': None, 'v025': 2}])
        save_charts(test_part, tp, threshold, staging, config.synthetic)
        report = report_text(summary)
        (staging / 'report.md').write_text(report, encoding='utf-8')
        # Escaped report is safe to open even if an input metadata field contains markup.
        (staging / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Model report</title><style>body{max-width:1050px;margin:40px auto;font:16px/1.6 system-ui;padding:24px;color:#183541}pre{white-space:pre-wrap}img{width:100%}</style><pre>' + html.escape(report) + '</pre><img src="evaluation.png"><img src="confusion-matrix.png">', encoding='utf-8')
        for name in ['pipeline.py', 'predict.py', 'requirements.txt']:
            shutil.copy2(Path(__file__).parent / name, staging / name)
        (staging / 'README.txt').write_text('MODEL DOWNLOAD\n\n' + ('SYNTHETIC DEMONSTRATION ONLY.\n\n' if config.synthetic else '') +
            'Use Python 3.14 (the tested runtime) and install requirements.txt in a virtual environment.\n'
            'Predict: python predict.py --model model.joblib --input prediction-example.json --output predictions.json\n'
            'Input is a JSON array of objects with the eight fields in input-schema.json. Use original NFHS codes, not labels.\n'
            'The example is fictional, not a real child. No training records are included.\n'
            'The model includes trained preprocessing and probability calibration. It is the evaluated model, not refitted on test data.\n'
            'Read report.md/report.html and metrics.json for results, selection method and limitations.\n'
            'Only load trusted joblib files; they are Python executable serialization.\n', encoding='utf-8')
        with zipfile.ZipFile(staging / 'trained-model-package.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.iterdir()):
                if path.name != 'trained-model-package.zip':
                    archive.write(path, path.name)
        staging.rename(output)
    progress('Training complete')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-mode', choices=['geographic', 'grouped'], default='geographic')
    parser.add_argument('--test-states', type=int, nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-repeats', type=int, default=200)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    try:
        summary = train(args.input, args.output, Config(seed=args.seed, split_mode=args.split_mode, test_states=tuple(args.test_states), bootstrap_repeats=args.bootstrap_repeats, synthetic=args.synthetic), print)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'Training stopped: {exc}\n')
    print(f"Selected {summary['selected_model']}; download package: {args.output / 'trained-model-package.zip'}")


if __name__ == '__main__':
    main()
