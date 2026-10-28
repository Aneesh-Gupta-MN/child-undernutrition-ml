"""Predict from a trusted exported bundle and a JSON array of eight-field records."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from pipeline import prepare_features, probabilities, SCHEMA_VERSION


def predict_records(bundle, records):
    if bundle.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported model schema version.')
    if not isinstance(records, list) or not records:
        raise ValueError('Input must be a nonempty JSON array of objects.')
    data = pd.DataFrame(records)
    if 'b19' not in data:
        raise ValueError('Child age (b19) is required.')
    age = pd.to_numeric(data.b19, errors='raise')
    if not (age.between(6, 59) & age.mod(1).eq(0)).all():
        raise ValueError('Every age must be a whole number from 6 to 59 months.')
    features = prepare_features(data, bundle['missing_codes'])
    p = probabilities(bundle, features)
    return [{'estimated_probability_current_stunting': float(value),
             'above_research_threshold': bool(value >= bundle['threshold']),
             'threshold': float(bundle['threshold']), 'synthetic_model': bool(bundle['synthetic'])} for value in p]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename.')
    try:
        bundle = joblib.load(args.model)
        result = predict_records(bundle, json.loads(args.input.read_text()))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Prediction stopped: {exc}\n')
    print(f'Wrote {len(result)} predictions to {args.output}')


if __name__ == '__main__':
    main()
