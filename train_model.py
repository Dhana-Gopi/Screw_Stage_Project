"""Train with a REAL labeled CSV containing Angle, Torque, Stage, RunID.
Run: python train_model.py labeled_data.csv
Each RunID must identify one independent tightening run.
"""
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from core import normalize, features, FEATURES, STAGES

if len(sys.argv)!=2:
    raise SystemExit('Usage: python train_model.py labeled_data.csv')
raw = pd.read_csv(sys.argv[1])
required = {'Angle','Torque','Stage','RunID'}
if not required.issubset(raw.columns):
    raise SystemExit('Training CSV must contain Angle, Torque, Stage, RunID.')
if raw[list(required)].isna().any().any():
    raise SystemExit('Training data contains missing values.')
if not set(raw.Stage.unique()).issubset(STAGES):
    raise SystemExit('Stage labels must be Engage, Runway, Sitting, Tight.')
if raw.RunID.nunique()<5:
    raise SystemExit('Use at least five independent labeled tightening runs for a basic holdout test.')
frames=[]
for run_id, group in raw.groupby('RunID',sort=False):
    clean = normalize(group[['Angle','Torque']])
    feat = features(clean)
    feat['Stage'] = group.Stage.to_numpy()
    feat['RunID'] = run_id
    frames.append(feat)
data = pd.concat(frames,ignore_index=True)
if set(data.Stage.unique())!=set(STAGES):
    raise SystemExit('Training data must contain all four stage labels.')
split = GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
tr,te = next(split.split(data[FEATURES],data.Stage,groups=data.RunID))
model = RandomForestClassifier(n_estimators=250,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1)
model.fit(data.iloc[tr][FEATURES],data.iloc[tr].Stage)
print('Held-out runs:',list(data.iloc[te].RunID.unique()))
print(classification_report(data.iloc[te].Stage,model.predict(data.iloc[te][FEATURES]),labels=STAGES,zero_division=0))
# Save model trained on ALL labeled runs after the independent-run holdout report.
model.fit(data[FEATURES],data.Stage)
joblib.dump(model,'screw_stage_model.joblib')
print('Saved screw_stage_model.joblib. Do not treat holdout metrics as production certification.')
