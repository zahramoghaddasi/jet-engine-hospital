# ============================================================
# app.py – Jet Engine Hospital Dashboard (Streamlit Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Jet Engine Hospital", layout="wide")
st.title("🚀 Jet Engine Hospital")
st.markdown("### Predictive Maintenance Dashboard for NASA C‑MAPSS Turbofans")

# ============================================================
# CONFIGURATION – CHANGE THIS FOR STAGE 2
# ============================================================
SUBSET = st.sidebar.selectbox("Select Dataset", ["FD001", "FD002"])
STAGE = SUBSET

st.sidebar.markdown(f"**Current Stage:** {SUBSET}")

# ----------------------
# 1. Load Models & Preprocessors
# ----------------------
@st.cache_resource
def load_artifacts(subset, stage):
    try:
        feature_list = joblib.load(f'models_{stage}/feature_list.pkl')
        scaler = joblib.load(f'models_{stage}/scaler_filtered.pkl')
        rul_model = joblib.load(f'models_{stage}/best_rul_model_{subset}_capped.pkl')
        uncertainty_model = joblib.load(f'models_{stage}/best_rul_model_uncertainty.pkl')
        interval_width = joblib.load(f'models_{stage}/interval_width.pkl')
        
        classifiers = {}
        for h in [10, 20, 30]:
            classifiers[h] = joblib.load(f'models_{stage}/classifier_{h}.pkl')
        
        calibrators = joblib.load(f'models_{stage}/calibrators.pkl')
        iso_forest = joblib.load(f'models_{stage}/isolation_forest.pkl')
        anomaly_scaler = joblib.load(f'models_{stage}/anomaly_scaler.pkl')
        anomaly_threshold = joblib.load(f'models_{stage}/anomaly_threshold.pkl')
        
        test_feat = pd.read_csv(f'models_{stage}/test_features_{subset}.csv')
        
        return feature_list, scaler, rul_model, uncertainty_model, interval_width, \
               classifiers, calibrators, iso_forest, anomaly_scaler, anomaly_threshold, test_feat
    except Exception as e:
        st.error(f"❌ Error loading artifacts for {subset}: {e}")
        return None

artifacts = load_artifacts(SUBSET, STAGE)
if artifacts is None:
    st.stop()

feature_list, scaler, rul_model, uncertainty_model, interval_width, \
classifiers, calibrators, iso_forest, anomaly_scaler, anomaly_threshold, test_feat = artifacts

# ----------------------
# 2. Engine & Cycle Input
# ----------------------
valid_engines = sorted(test_feat['engine_id'].unique())
engine_cycle_range = {
    eng: (test_feat[test_feat['engine_id'] == eng]['cycle'].min(),
          test_feat[test_feat['engine_id'] == eng]['cycle'].max())
    for eng in valid_engines
}

col1, col2 = st.columns(2)
with col1:
    engine_id = st.number_input("Engine ID", min_value=1, max_value=max(valid_engines), value=1, step=1)
with col2:
    cycle = st.number_input("Cycle", min_value=1, value=1, step=1)

predict_btn = st.button("🔄 Predict", type="primary")

# ----------------------
# 3. Prediction Function
# ----------------------
def predict_engine(engine_id, cycle):
    if engine_id not in valid_engines:
        return f"❌ Engine {engine_id} not found. Available: {valid_engines[0]} to {valid_engines[-1]}"
    
    min_cycle, max_cycle = engine_cycle_range[engine_id]
    if not (min_cycle <= cycle <= max_cycle):
        return f"❌ Cycle {cycle} not valid for Engine {engine_id}. Range: {min_cycle} to {max_cycle}"
    
    row = test_feat[(test_feat['engine_id'] == engine_id) & (test_feat['cycle'] == cycle)]
    if row.empty:
        return f"❌ No data found for Engine {engine_id}, Cycle {cycle}."
    
    X = row[feature_list].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    rul_pred = rul_model.predict(X_scaled)[0]
    true_rul = row['true_RUL'].values[0]
    
    rul_uncertainty = uncertainty_model.predict(X_scaled)[0]
    lower_bound = rul_uncertainty - interval_width
    upper_bound = rul_uncertainty + interval_width
    
    # Risk
    risk_model = {}
    for h in [10, 20, 30]:
        clf = classifiers[h]
        prob_raw = clf.predict_proba(X_scaled)[0, 1]
        prob_calibrated = calibrators[h].predict([prob_raw])[0]
        risk_model[h] = prob_calibrated
    
    risk_true = {}
    for h in [10, 20, 30]:
        if true_rul <= h:
            risk_true[h] = 1.0
        else:
            risk_true[h] = max(0, min(1, 1 - (true_rul / 100)))
    
    if true_rul < 30:
        risk = risk_true
        risk_source = "True RUL (fallback)"
    else:
        risk = risk_model
        risk_source = "Logistic Regression (calibrated)"
    
    # Anomaly
    raw_score = -iso_forest.decision_function(X_scaled)[0]
    anomaly_score = anomaly_scaler.transform([[raw_score]])[0][0]
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
    
    # Decision
    decision = "CONTINUE"
    reasons = []
    if lower_bound < 30:
        decision = "STOP"
        reasons.append("🚨 RUL lower bound < 30 cycles (critical)")
    elif anomaly_score > anomaly_threshold:
        decision = "INSPECT"
        reasons.append(f"⚠️ Anomaly score {anomaly_score:.2f} > threshold {anomaly_threshold:.2f}")
    elif risk[10] > 0.5 or risk[20] > 0.4 or risk[30] > 0.3:
        decision = "INSPECT"
        reasons.append(f"📈 Elevated failure risk ({risk_source})")
    else:
        reasons.append("✅ All indicators within safe range")
    
    # Output
    output = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║              JET ENGINE HOSPITAL – STATUS REPORT             
    ╠══════════════════════════════════════════════════════════════╣
    ║  Engine ID: {engine_id}   |   Cycle: {cycle}                    
    ╠══════════════════════════════════════════════════════════════╣
    ║  📊 True RUL          : {true_rul:.0f} cycles (ground truth)  
    ║  📊 RUL Estimate      : {rul_pred:.1f} cycles                
    ║  📊 95% CI            : [{lower_bound:.1f}, {upper_bound:.1f}]  
    ╠══════════════════════════════════════════════════════════════╣
    ║  🚨 Failure Risk ({risk_source}):                            
    ║     10 cycles : {risk[10]*100:.1f}%                         
    ║     20 cycles : {risk[20]*100:.1f}%                         
    ║     30 cycles : {risk[30]*100:.1f}%                         
    ╠══════════════════════════════════════════════════════════════╣
    ║  🔍 Anomaly Score   : {anomaly_score:.3f} (threshold: {anomaly_threshold:.2f})  
    ╠══════════════════════════════════════════════════════════════╣
    ║  🎯 RECOMMENDATION  : {decision}                             
    ║  📝 Reason(s)       : {', '.join(reasons)}                  
    ╚══════════════════════════════════════════════════════════════╝
    """
    return output

# ----------------------
# 4. Show Result
# ----------------------
if predict_btn:
    with st.spinner("🔄 Predicting..."):
        result = predict_engine(engine_id, cycle)
        st.text(result)