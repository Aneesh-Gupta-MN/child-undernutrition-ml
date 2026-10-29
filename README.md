# Child Undernutrition ML

Upload an approved India NFHS-5 Children's Recode dataset, validate the cohort, compare reproducible models, and download the selected trained model with its evaluation report.

The complete local application is implemented. The real NFHS dataset has not been downloaded; end-to-end verification uses labelled synthetic records only.

## Use the app

Open **http://127.0.0.1:8501** while the local app is running.

1. Choose **Upload NFHS data** and upload the India NFHS-5 Children's Recode (KR) dataset as `.csv`, original Stata `.dta`, or a KR Stata ZIP archive.
2. Review the eligible-child count. Choose geographic validation (recommended for a national file) or internal PSU validation for a restricted-area file. You can name test states before training; otherwise the fixed seed selects them.
3. Confirm that the file uses original variable codes and unmodified `hw70` units.
4. Click **Train and generate my model**.
5. Download the trained-model package and read its report.

**Try a synthetic demonstration** runs the same workflow without an NFHS download. Every resulting model and report is marked synthetic. Demonstration scores are not evidence of real performance.

To start or restart locally:

```bash
./start.sh
```

On a new machine, use the tested Python 3.14 Linux environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./start.sh
```

The app binds to this computer only. Uploads are processed locally, temporary upload copies are removed after inspection/training, and completed outputs remain under `runs/`. No child-level training records are included in the download. Generated reports and artifacts still need normal access controls.

### CSV uploads

Use UTF-8 (with or without a BOM), one child per row, and comma, semicolon or tab delimiters. Column names are trimmed and matched without case sensitivity. Extra columns are ignored. Child IDs retain leading zeros. The app provides an empty CSV template listing all required fields:

`b19, b4, bord, v106, v136, v113, v116, v025, v005, v021, v022, v024, v001, v002, caseid, bidx, b5, b9, hw70, hw1`

Keep original numeric codes and units (`hw70` = z-score × 100; `v005` = original survey weight). CSV does not retain Stata value labels, so convert special missing-response codes to blank cells using your codebook before upload. The importer does not guess that every 9/99 code is missing. Blanks, `.`, `NA`, `N/A`, `NaN` and `NULL` are accepted as missing values. A CSV exported with words such as “male” instead of numeric category codes must be re-exported with numeric codes. Missing required fields, duplicate headers and malformed rows produce an actionable error before training. CSV codebook limitations are included in the generated report.

The supplied `india_kr.csv` derived extract is supported directly. Its headers include `child_age_months`, `child_sex`, `birth_order`, `mother_education`, `household_size`, `drinking_water_source`, `toilet_type`, `residence_type`, `state_code`, `psu`, `strata`, `sample_weight_x1000000`, and `height_for_age_zscore_x100`. It lacks raw child/household IDs, living status, mother co-residence and separate measurement age. The app uses its supplied child rows aged 6–59 months with valid HAZ, and keeps entire PSUs together. These restrictions are written into the report and must be stated in your academic work.

## What the download contains

- `model.joblib`: the selected trained estimator, fitted preprocessing, probability calibration, threshold and metadata.
- `predict.py`, `pipeline.py`: code to reuse the model with new records.
- `input-schema.json`: the eight expected input fields, original category labels and missing-code handling.
- `prediction-example.json`: a fictional example to test model loading.
- `report.md`, `report.html`: research report, cohort flow, comparison and limitations.
- `evaluation.png`, `confusion-matrix.png`: test-set plots.
- `metrics.json`: weighted/unweighted results, subgroup summaries, approximate uncertainty, settings, input fingerprint and split-group hashes.
- `requirements.txt`, `README.txt`: exact dependency versions and reuse instructions.

Extract the ZIP, install its requirements in a virtual environment, then:

```bash
python predict.py --model model.joblib --input prediction-example.json --output predictions.json
```

Prediction input is a JSON array of objects using original DHS codes. Only child age (6–59 months), sex, birth order, maternal education, household membership count, drinking-water source, toilet facility and rural/urban residence enter the model. Null represents a missing response. Unseen nominal categories are handled by the fitted encoder. The prediction output is a probability of **current stunting**, a research-threshold flag and a synthetic-model flag.

Load only trusted joblib model files: Python serialization can execute code.

## Training design

The conservative cohort requires living children aged 6–59 months, living with their interviewed mother (`b9=0`), valid HAZ (`hw70` from −600 to +600), and identical interview/measurement age (`b19=hw1`). Every sequential exclusion is reported. The target is `hw70 < -200`; missing/flagged outcomes are never negatives. This subset must not be reported as national child prevalence.

The pipeline stops for missing columns, duplicate/blank child identifiers, missing design information, nonpositive weights, inconsistent cluster mappings, apparently transformed outcomes and inadequate split sizes. It requires at least 400 eligible records, and every partition must contain at least 50 records, 10 positive and 10 negative outcomes, and 3 PSUs. These are software feasibility minimums, not a statistical sample-size justification.

Four independent partitions are created before fitting:

1. **Fit**: learns all imputation, scaling, category encoding and model parameters.
2. **Calibration**: learns sigmoid probability calibration for each nonconstant model.
3. **Validation**: selects the lowest weighted Brier score and a threshold meeting the configured sensitivity target on this set.
4. **Test**: evaluates the selected model once. Entire states are held out in geographic mode; PSUs are held out in internal mode.

PSUs and households cannot cross partitions. The model comparison comprises a prevalence baseline, an age-spline logistic baseline, full logistic regression and shallow XGBoost. The default workflow uses fixed hyperparameters, not an exhaustive search or nested cross-validation. A baseline can legitimately win. The exported model is exactly the evaluated model: it is not refit on the held-out test set.

Survey weights enter fitting, calibration and evaluation. Approximate uncertainty resamples PSUs within observed state/stratum; it is conditional on the frozen model and does not include model-selection uncertainty or every aspect of survey design. Small subgroup summaries are suppressed. Review the release's variable labels, sampling appendix, units and cohort restrictions before interpreting real-data results.

## Command-line use

```bash
.venv/bin/python pipeline.py --input data/raw/YOUR_KR_FILE.dta --output runs/my-first-real-run
```

Optional controls include `--test-states 29`, `--split-mode grouped`, `--seed 42`, and `--bootstrap-repeats 200`. State codes must be checked in the actual release. Choose settings before examining outcomes; repeatedly optimizing a test split invalidates its role as a holdout.

Run the software checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover label/age boundaries, special codes, duplicates, invalid weights, household/state separation, training-only preprocessing, missing/unseen categories, ZIP input, artifact round-trip prediction and overwrite protection. The Streamlit controls and complete synthetic training flow are also checked with its application-testing framework.

`audit_data.py` remains available for a separate, preliminary audit. Its provisional cohort differs from the finalized conservative training cohort, so its counts should not be substituted for the training report.

## Data access and limitations

The real dataset is still pending. Obtain **India 2019–21 → Children's Recode (KR) → Stata** through your approved [DHS research account](https://dhsprogram.com/data/dataset/India_Standard-DHS_2020.cfm?flag=0). See `access-request.md` for the proposed study description and the [catalogue access conditions](https://microdata.worldbank.org/catalog/4482).

This is a reproducible research prototype, not a clinically validated screening or forecasting system. A completed training run does not establish useful discrimination, calibration in the college's setting, or improved clinical outcomes. Read the actual report, including negative findings. Scores must not determine eligibility for measurement or care.

Implementation references: [Streamlit file upload](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader), [scikit-learn group splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), and the existing project review. Checked 29 October 2026.
