import numpy as np
import pandas as pd

STAGES = ['Engage', 'Runway', 'Sitting', 'Tight']
COLORS = {'Engage':'#7C8FA5','Runway':'#E7B93D','Sitting':'#E67E4A','Tight':'#3E9B77'}

def normalize(df):
    names = {str(c).strip().lower().replace(' ','').replace('_',''): c for c in df.columns}
    a = next((names[k] for k in ('angle','angledeg','angledegrees','x') if k in names), None)
    t = next((names[k] for k in ('torque','torquenm','y') if k in names), None)
    if a is None or t is None:
        raise ValueError('CSV must have Angle and Torque columns (or X and Y).')
    out = pd.DataFrame({'Angle': pd.to_numeric(df[a],errors='coerce'), 'Torque':pd.to_numeric(df[t],errors='coerce')})
    if out.isna().any().any():
        raise ValueError('Angle/Torque contains missing or nonnumeric values. Clean the CSV first.')
    if len(out)<12:
        raise ValueError('At least 12 measurement points are required.')
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Angle/Torque contains non-finite values.')
    if (out.Angle.diff().dropna()<0).any():
        raise ValueError('Angle must be nondecreasing in acquisition order. Do not sort a multi-run CSV into one curve.')
    if out.Angle.nunique()<8:
        raise ValueError('At least 8 distinct angle values are needed.')
    return out.reset_index(drop=True)

def features(df):
    d = df.copy()
    a = d.Angle.to_numpy(dtype=float)
    t = d.Torque.to_numpy(dtype=float)
    span = max(float(a.max()-a.min()),1e-9)
    d['AngleProgress'] = (a-a.min())/span
    d['TorqueSmooth'] = pd.Series(t).rolling(5,center=True,min_periods=1).median().to_numpy()
    d['Slope'] = np.gradient(d.TorqueSmooth.to_numpy(),a+np.arange(len(a))*1e-8)
    d['Slope'] = np.clip(np.nan_to_num(d.Slope.to_numpy(),nan=0,posinf=0,neginf=0),-1e6,1e6)
    d['TorqueDelta'] = d.TorqueSmooth.diff().fillna(0)
    d['TorqueProgress'] = (d.TorqueSmooth-d.TorqueSmooth.min())/max(float(d.TorqueSmooth.max()-d.TorqueSmooth.min()),1e-9)
    return d

FEATURES = ['Angle','Torque','AngleProgress','TorqueSmooth','Slope','TorqueDelta','TorqueProgress']

def demo_predict(df):
    # DEMONSTRATION ONLY: stage boundaries are assumed, not learned or validated.
    p = features(df).AngleProgress.to_numpy()
    indices = np.select([p<0.12,p<0.72,p<0.90], [0,1,2], default=3)
    return np.array(STAGES)[indices]

def predict(df, model=None):
    f = features(df)
    if model is None:
        return demo_predict(df), 'Illustrative rule-based stages (NOT ML or engineering-validated)'
    return model.predict(f[FEATURES]), 'Machine-learning estimates (validate on independent real runs)'
