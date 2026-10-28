"""Generate labelled synthetic fixtures for software testing, never study results."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pyreadstat


def create_demo(path, seed=101):
    rng = np.random.default_rng(seed)
    rows = []
    for state in range(1, 7):
        for psu in range(1, 31):
            for child in range(24):
                age = int(rng.integers(6, 60))
                education = int(rng.integers(0, 4))
                residence = int(rng.integers(1, 3))
                probability = 1 / (1 + np.exp(-(-1.0 + .018 * (age-24) - .25 * education + .3 * (residence-1))))
                y = rng.binomial(1, probability)
                rows.append(dict(caseid=f'SYNTHETIC_{state}_{psu}_{child//2}', bidx=child%2+1,
                    b5=1, b9=0, b19=age, hw1=age, b4=int(rng.integers(1, 3)), bord=int(rng.integers(1, 6)),
                    v106=education, v136=int(rng.integers(2, 10)), v113=int(rng.choice([11, 21, 31, 96, 99])),
                    v116=int(rng.choice([11, 21, 31, 96])), v025=residence,
                    v005=int(rng.integers(500000, 2000000)), v024=state, v021=psu, v001=psu,
                    v002=child//2+1, v022=(psu-1)//10+1,
                    hw70=int(rng.integers(-400, -200) if y else rng.integers(-200, 201))))
    df = pd.DataFrame(rows)
    labels = {
        'v024': {i: f'Synthetic state {i}' for i in range(1, 7)},
        'b4': {1: 'male', 2: 'female'}, 'b9': {0: 'with respondent'},
        'v106': {0: 'no education', 1: 'primary', 2: 'secondary', 3: 'higher'},
        'v025': {1: 'urban', 2: 'rural'},
        'v113': {11: 'piped', 21: 'tube well', 31: 'well', 96: 'other', 99: 'missing'},
        'v116': {11: 'flush', 21: 'pit', 31: 'no facility', 96: 'other'},
        'hw70': {9996: 'height out of plausible limits', 9997: 'age out of plausible limits', 9998: 'flagged'},
    }
    path = Path(path)
    if path.exists():
        raise ValueError('Demo path already exists.')
    path.parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_dta(df, str(path), file_label='SYNTHETIC SOFTWARE TEST ONLY - NOT NFHS DATA', variable_value_labels=labels)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    create_demo(args.output)
    print(f'Created synthetic fixture: {args.output}')
