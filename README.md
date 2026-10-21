# Screw Stage Intelligence

A Streamlit torque–angle dashboard with an interactive screw animation and four stage labels: Engage, Runway, Sitting, Tight.

## Files in execution order
1. `requirements.txt` — dependencies
2. `example_curve.csv` — synthetic example for UI demonstration only
3. `core.py` — CSV validation, feature extraction, demo stages and ML inference
4. `train_model.py` — optional training with real labeled runs
5. `app.py` — dashboard, curve, animation and export

## Run on Mac
Open Terminal and go to this folder:

```bash
cd /path/to/screw_stage_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

A browser opens at http://localhost:8501. The example CSV loads automatically; upload another CSV using the sidebar.

## CSV format for prediction

```csv
Angle,Torque
0,0.10
3.6,0.13
7.2,0.15
...
```

The real file must contain **at least 12 rows**, at least 8 distinct angle values, nondecreasing angles and numeric torque values. Each uploaded file is **one run**. Angle is assumed to be degrees and torque N·m; other units must be converted before upload.

## Training an ML model (optional)

Collect multiple independently labeled tightening runs. Use this schema:

```csv
RunID,Angle,Torque,Stage
run001,0,0.1,Engage
run001,3.6,0.13,Engage
...
run002,0,0.09,Engage
...
```

For each RunID include a complete ordered curve, at least 12 rows, and verified labels assigned using independent ground truth. All four labels must occur across the dataset. At least five separate runs are needed by the script; substantially more are required for meaningful validation. Do not label training data using the demo stage percentages: that would only teach the model to copy the demonstration rules.

```bash
python train_model.py labeled_data.csv
streamlit run app.py
```

This creates `screw_stage_model.joblib`, which `app.py` automatically loads. Training and holdout split are separated by RunID to avoid mixing points from the same run. The app's output is still an estimate, not a verified tightening result.

## Publish a link

1. Create a GitHub repository and upload `app.py`, `core.py`, `requirements.txt`, and `example_curve.csv`.
2. If you have trained on real data, include the trusted `screw_stage_model.joblib` only if sharing the model is permitted. Never load untrusted `.joblib` files.
3. Sign in at https://share.streamlit.io/ and choose **Create app** → **Deploy a public app from GitHub**.
4. Select your repository and `app.py` as the entrypoint, then deploy.
5. Share the resulting Streamlit URL. Public hosting is not appropriate for confidential industrial measurements without appropriate controls.

## Important limits

- Without real verified stage labels, the default is a **rule-based visual demonstration**, not ML.
- The demonstration assigns stage ranges at 12%, 72% and 90% of angle progress. These are arbitrary and **not engineering specifications**.
- The animation's downward movement is illustrative and does not derive true axial depth from torque/angle.
- The model does not determine fastener safety, defects, seating quality, target torque compliance or production pass/fail.
- Reliable deployment requires validation against independent runs, tool calibration, fastener specification and ideally additional seating/position sensing.
