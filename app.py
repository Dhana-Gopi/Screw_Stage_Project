import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from core import normalize, predict, STAGES, COLORS

st.set_page_config(page_title='Screw Stage Intelligence',page_icon='🔩',layout='wide')
st.markdown('''<style>
.stApp {background:#fffdf7;color:#292821} h1,h2,h3 {color:#282820}
[data-testid="stMetric"] {background:white;border:1px solid #f0e4b5;border-radius:12px;padding:15px}
.stButton button {background:#e6b82f;color:#211d12;border-radius:9px;border:0}
</style>''',unsafe_allow_html=True)
st.caption('TORQUE–ANGLE ANALYTICS  /  SCREW FITTING')
st.title('🔩 Screw Stage Intelligence')
st.write('Upload a single tightening run to visualize its torque–angle curve, estimate stages, and replay the screw motion.')

@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)

with st.sidebar:
    st.header('Input')
    uploaded = st.file_uploader('Upload angle–torque CSV',type=['csv'])
    st.caption('Required: Angle, Torque (or X, Y). One run per file. Angle in degrees; torque in N·m.')
    st.download_button('Download example CSV',Path(__file__).with_name('example_curve.csv').read_bytes(),file_name='example_curve.csv',mime='text/csv')
    model_path = Path(__file__).with_name('screw_stage_model.joblib')
    model = None
    if model_path.exists():
        try:
            model = load_model(str(model_path),model_path.stat().st_mtime)
            st.success('Trained model loaded')
        except Exception as e:
            st.error(f'Model could not load: {e}')
    else:
        st.warning('Demo stage estimates only. Train with verified labeled runs to enable ML.')

try:
    df = normalize(pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(Path(__file__).with_name('example_curve.csv')))
    labels,method = predict(df,model)
    df['Stage'] = labels
except Exception as exc:
    st.error(f'Cannot process input: {exc}')
    st.stop()

st.info(method + '. These predictions are not proof of a correctly seated or safely tightened fastener.')
cols=st.columns(4)
for col,stage in zip(cols,STAGES):
    col.metric(stage,f'{(df.Stage==stage).sum()} points')

fig=go.Figure()
fig.add_trace(go.Scatter(x=df.Angle,y=df.Torque,mode='lines',name='Torque curve',line=dict(color='#aaa9a1',width=2),hovertemplate='Angle: %{x:.2f}°<br>Torque: %{y:.3f} N·m<extra></extra>'))
for stage in STAGES:
    d=df[df.Stage==stage]
    fig.add_trace(go.Scatter(x=d.Angle,y=d.Torque,mode='markers',name=stage,marker=dict(color=COLORS[stage],size=7),customdata=np.array([stage]*len(d)),hovertemplate='Angle: %{x:.2f}°<br>Torque: %{y:.3f} N·m<br>Stage: %{customdata}<extra></extra>'))
fig.update_layout(title='Torque–angle curve with estimated stages',xaxis_title='Angle (degrees)',yaxis_title='Torque (N·m)',paper_bgcolor='#fffdf7',plot_bgcolor='#fffdf7',legend=dict(orientation='h',y=-0.25),margin=dict(l=15,r=15,t=50,b=15),height=440)
fig.update_xaxes(showgrid=True,gridcolor='#f0ead9')
fig.update_yaxes(showgrid=True,gridcolor='#f0ead9')
st.plotly_chart(fig,use_container_width=True)

st.subheader('Synchronized screw animation')
st.caption('Press Play or move the slider. The screw motion is schematic; it is not a measured depth or angle-to-displacement simulation.')
# One self-contained HTML component keeps chart marker and SVG screw animation synchronized.
records=[{'angle':round(float(a),5),'torque':round(float(t),5),'stage':str(s)} for a,t,s in zip(df.Angle,df.Torque,df.Stage)]
html='''<!DOCTYPE html><html><head><style>
body{font-family:Arial,sans-serif;background:#fffdf7;color:#282820;margin:0} .wrap{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.panel{background:white;border:1px solid #eee4c6;border-radius:14px;padding:14px;flex:1;min-width:230px}
button{border:0;background:#e6b82f;padding:10px 18px;border-radius:8px;cursor:pointer;font-weight:bold}input{width:100%;accent-color:#d2a62c}
#stage{font-size:22px;font-weight:700} .muted{font-size:13px;color:#777} svg{width:100%;height:245px}
</style></head><body><div class="wrap"><div class="panel"><svg viewBox="0 0 320 250" aria-label="Screw tightening schematic">
<rect x="50" y="168" width="220" height="75" rx="5" fill="#99a6b3"/><rect x="146" y="168" width="28" height="75" fill="#4f6070"/>
<g id="screw"><rect x="151" y="85" width="18" height="87" rx="2" fill="#bcc6d0" stroke="#586b7d" stroke-width="2"/>
<g stroke="#66788a" stroke-width="2">''' + ''.join(f'<path d="M151 {y} l18 7"/>' for y in range(92,164,9)) + '''</g>
<rect x="135" y="72" width="50" height="16" rx="5" fill="#8795a4" stroke="#526374" stroke-width="2"/>
<g id="slot" stroke="#374656" stroke-width="4"><path d="M145 80 h30"/></g></g>
<path d="M213 78 Q245 104 218 125" stroke="#d2a62c" stroke-width="3" fill="none" stroke-dasharray="5 4"/></svg>
<div id="stage">Engage</div><div class="muted" id="details"></div></div>
<div class="panel"><h3>Replay measurements</h3><button id="play">▶ Play</button> <button id="reset">↺ Reset</button>
<p><input type="range" id="scrub" min="0" max="__MAX__" value="0" step="1"></p>
<div id="readout"></div><p class="muted">Moving marker: current point on the torque–angle curve.</p>
<svg viewBox="0 0 320 140" id="mini"><rect x="30" y="10" width="280" height="110" fill="#faf8f0"/>
<path id="curve" fill="none" stroke="#d3aa2b" stroke-width="2.5"/><circle id="dot" r="5" fill="#e77e4a"/></svg></div></div>
<script>
const data=__DATA__,colors=__COLORS__;
const slider=document.getElementById('scrub'),screw=document.getElementById('screw'),slot=document.getElementById('slot');
let timer=null,idx=0;
const a0=data[0].angle,a1=data[data.length-1].angle;
const ts=data.map(d=>d.torque),t0=Math.min(...ts),t1=Math.max(...ts);
const xx=a=>30+280*(a-a0)/Math.max(a1-a0,1e-9),yy=t=>120-110*(t-t0)/Math.max(t1-t0,1e-9);
document.getElementById('curve').setAttribute('d',data.map((d,i)=>(i?'L':'M')+xx(d.angle).toFixed(2)+' '+yy(d.torque).toFixed(2)).join(' '));
function draw(i){idx=i;const d=data[i],p=(d.angle-a0)/Math.max(a1-a0,1e-9);
let depth=d.stage==='Engage'?0:d.stage==='Runway'?Math.min(1,(p-.12)/.60)*55:55;
screw.setAttribute('transform',`translate(0 ${depth})`);
slot.setAttribute('transform',`rotate(${d.angle%360} 160 80)`);
document.getElementById('stage').textContent=d.stage;
document.getElementById('stage').style.color=colors[d.stage];
document.getElementById('details').textContent=({Engage:'Positioned at the hole',Runway:'Rotating into the hole',Sitting:'Screw head seated',Tight:'Final torque application'})[d.stage];
document.getElementById('readout').textContent=`Point ${i+1}/${data.length} · Angle ${d.angle.toFixed(2)}° · Torque ${d.torque.toFixed(3)} N·m`;
document.getElementById('dot').setAttribute('cx',xx(d.angle));document.getElementById('dot').setAttribute('cy',yy(d.torque));slider.value=i;}
function stop(){if(timer){clearInterval(timer);timer=null;}document.getElementById('play').textContent='▶ Play';}
document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(idx>=data.length-1)draw(0);
timer=setInterval(()=>{if(idx>=data.length-1){stop();return;}draw(idx+1)},Math.max(30,Math.round(9000/data.length)));
document.getElementById('play').textContent='❚❚ Pause';};
document.getElementById('reset').onclick=()=>{stop();draw(0)};
slider.oninput=()=>{stop();draw(Number(slider.value))};draw(0);
</script></body></html>'''
html=html.replace('__MAX__',str(len(records)-1)).replace('__DATA__',json.dumps(records)).replace('__COLORS__',json.dumps(COLORS))
components.html(html,height=370,scrolling=False)

st.subheader('Predictions and export')
st.dataframe(df,use_container_width=True,hide_index=True)
st.download_button('Download predictions CSV',df.to_csv(index=False).encode('utf-8'),file_name='screw_stage_predictions.csv',mime='text/csv')
st.caption('Security: upload only trusted data; deploy only trusted model files. No engineering pass/fail or target torque validation is implemented.')
