"""Local upload-to-model application."""
from datetime import datetime
import hashlib
from pathlib import Path
import tempfile
import uuid

import pandas as pd
import streamlit as st

from demo_data import create_demo
from pipeline import Config, read_dataset, clean_cohort, train

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Nursing Research · Model Studio', page_icon='🌱', layout='wide')
st.markdown('''<style>
.block-container{max-width:1150px;padding-top:2.5rem}
[data-testid="stMetric"]{background:#eaf4f1;border-radius:12px;padding:18px}
.stDownloadButton button{border-radius:8px}
</style>''', unsafe_allow_html=True)
st.caption('RV NURSING COLLEGE · RESEARCH WORKSPACE')
st.title('From dataset to trained model')
st.write('Upload your NFHS-5 child dataset. Get a reusable model and a report showing how it performed on children kept out of training.')
st.info('Research task: estimate current stunting among children aged 6–59 months. The model does not predict future malnutrition or replace measured assessment.')
mode = st.radio('Data source', ['Upload NFHS data', 'Try a synthetic demonstration'], horizontal=True)
upload = None
if mode == 'Upload NFHS data':
    upload = st.file_uploader('Children’s Recode file', type=['dta', 'zip'], max_upload_size=1024,
                              help='Use the original Stata KR dataset or its ZIP. Aggregate factsheets and already-transformed files are not supported.')
    if upload is None:
        st.caption('The file stays on this computer when you use this locally running app. No external service receives it.')
        st.stop()
    fingerprint = hashlib.sha256(upload.getbuffer()).hexdigest()
else:
    st.warning('Synthetic demonstration only. Its model and scores are for checking the software, not academic findings.')
    fingerprint = 'synthetic-demo-v1'

if st.session_state.get('data_fingerprint') != fingerprint:
    for key in ['inspection', 'run_result', 'run_path', 'inspection_error']:
        st.session_state.pop(key, None)
    st.session_state.data_fingerprint = fingerprint

if 'inspection' not in st.session_state and 'inspection_error' not in st.session_state:
    try:
        with st.spinner('Checking file structure and eligible children…'):
            with tempfile.TemporaryDirectory(prefix='nfhs-inspect-') as temp:
                if upload is not None:
                    path = Path(temp) / ('upload.zip' if upload.name.lower().endswith('.zip') else 'upload.dta')
                    path.write_bytes(upload.getbuffer())
                else:
                    path = create_demo(Path(temp) / 'SYNTHETIC_KR.dta')
                data, meta = read_dataset(path)
                cohort, _, audit = clean_cohort(data, meta)
                st.session_state.inspection = {'audit': audit, 'states': sorted(cohort.state.unique().tolist()),
                                               'state_labels': meta['value_labels'].get('v024', {}), 'meta': meta}
    except Exception as exc:
        st.session_state.inspection_error = str(exc)
if 'inspection_error' in st.session_state:
    st.error('The dataset cannot be trained yet: ' + st.session_state.inspection_error)
    st.caption('Correct the file and upload it again. No model was created.')
    st.stop()

inspection = st.session_state.inspection
audit = inspection['audit']
a, b, c = st.columns(3)
a.metric('Eligible children', f"{audit['eligible']:,}")
b.metric('States in eligible data', len(inspection['states']))
c.metric('Excluded records', f"{audit['cohort_flow']['input_records'] - audit['eligible']:,}")
with st.expander('Data checks and eligibility'):
    st.write('The cohort includes living children who live with their interviewed mother, have a valid outcome and matching interview/measurement age. Missing design fields or duplicate identifiers stop training.')
    st.dataframe(pd.DataFrame([{'Stage': k.replace('_', ' '), 'Children remaining': v} for k, v in audit['cohort_flow'].items()]), hide_index=True, width='stretch')
    st.write('Predictor values treated as missing')
    st.json(audit['predictor_missing_counts'])
    st.write('Release labels used to interpret the inputs')
    st.json({k: inspection['meta']['value_labels'].get(k, {}) for k in ['b4', 'v106', 'v113', 'v116', 'v025']})

st.subheader('Choose how to evaluate the model')
split_label = st.selectbox('Final test set', ['Hold out entire states', 'Hold out survey communities within the supplied area'])
split_mode = 'geographic' if split_label == 'Hold out entire states' else 'grouped'
selected_states = []
if split_mode == 'geographic':
    selected_states = st.multiselect('States reserved for the final test (optional)', inspection['states'],
        format_func=lambda value: inspection['state_labels'].get(str(float(value)), inspection['state_labels'].get(str(value), f'State {value}')),
        help='Leave empty to reserve 20% of states using the fixed seed before any models are fitted. For a planned local setting, select its state in advance.')
else:
    st.caption('This measures performance in held-out communities from the same supplied area. It is not a geographic transportability test.')
with st.expander('Research settings'):
    seed = st.number_input('Reproducible split seed', min_value=0, max_value=1_000_000, value=42, step=1)
    sensitivity = st.slider('Sensitivity target on validation data', min_value=50, max_value=99, value=80, format='%d%%')
    repeats = st.selectbox('Bootstrap repetitions for uncertainty', [200, 500, 1000], index=0)
    st.caption('The winner has the lowest validation Brier score, which measures probability error. All models use fixed settings. The final test set is not used to choose the model or threshold.')
confirmed = st.checkbox('This is the original NFHS-5 KR file with unmodified variable codes and height-for-age stored as z-score × 100.', value=(mode == 'Try a synthetic demonstration'))
config = Config(seed=int(seed), split_mode=split_mode, test_states=tuple(selected_states),
                target_sensitivity=sensitivity/100, bootstrap_repeats=repeats, synthetic=mode == 'Try a synthetic demonstration')
settings_signature = repr(config)
if 'run_result' in st.session_state and st.session_state.get('run_settings') != settings_signature:
    st.info('Settings have changed. Results below belong to the previous completed run; train again to apply the new settings.')

if st.button('Train and generate my model', type='primary', disabled=not confirmed):
    # An unsuccessful rerun must not display an older run as newly successful.
    for key in ['run_result', 'run_path']:
        st.session_state.pop(key, None)
    output = ROOT / 'runs' / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
    try:
        with st.status('Preparing training', expanded=True) as status:
            def progress(message):
                status.update(label=message)
                st.write(message)
            with tempfile.TemporaryDirectory(prefix='nfhs-upload-') as temp:
                if upload is not None:
                    path = Path(temp) / ('upload.zip' if upload.name.lower().endswith('.zip') else 'upload.dta')
                    path.write_bytes(upload.getbuffer())
                else:
                    path = create_demo(Path(temp) / 'SYNTHETIC_KR.dta')
                result = train(path, output, config, progress)
            status.update(label='Model and report are ready', state='complete', expanded=False)
        st.session_state.run_result = result
        st.session_state.run_path = str(output)
        st.session_state.run_settings = settings_signature
    except Exception as exc:
        st.error('Training stopped: ' + str(exc))
        st.caption('No completed model package was published. Check the dataset or evaluation settings and try again.')

if 'run_result' in st.session_state:
    result = st.session_state.run_result
    output = Path(st.session_state.run_path)
    st.subheader('Your trained model is ready')
    if result['config']['synthetic']:
        st.warning('SYNTHETIC RESULT — this is a demonstration model, not a model trained on NFHS observations.')
    st.write('Selected model: **' + result['selected_model'].replace('_', ' ').title() + '**')
    metrics = result['test_metrics_weighted']
    a, b, c = st.columns(3)
    a.metric('Test ROC-AUC', f"{metrics['roc_auc']:.3f}", help='Ranking performance. 0.5 is chance ranking; 1.0 is perfect.')
    b.metric('Test sensitivity', f"{metrics['sensitivity']:.1%}", help='Fraction of observed stunting cases above the chosen research threshold.')
    c.metric('Test Brier score', f"{metrics['brier_score']:.3f}", help='Mean squared probability error; lower is better.')
    left, right = st.columns(2)
    left.download_button('Download trained model package', (output / 'trained-model-package.zip').read_bytes(),
                         file_name=('SYNTHETIC-' if result['config']['synthetic'] else '') + 'trained-model-package.zip', mime='application/zip', type='primary')
    right.download_button('Download research report', (output / 'report.md').read_text(), file_name='model-report.md', mime='text/markdown')
    st.caption('The package includes the trained model, preprocessing, prediction script, input guide, charts, report and reproducibility information. The uploaded child records are not included.')
    st.image(str(output / 'evaluation.png'))
    with st.expander('Compare models on validation data'):
        table = [{'Model': name, 'Brier score': values['brier_score'], 'ROC-AUC': values['roc_auc']} for name, values in result['validation_metrics'].items()]
        st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
    with st.expander('Read the full report'):
        st.markdown((output / 'report.md').read_text().split('![Evaluation curves]')[0])
