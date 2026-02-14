#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from joblib import load
import numpy as np
import os

base_path = os.path.dirname(__file__)
path_cb = os.path.join(base_path, 'CB_model.joblib')
path_mapie = os.path.join(base_path, 'Mapie_Interval.joblib')
path_scaler = os.path.join(base_path, 'StandardScaler.joblib')

try:
    model_point = load(path_cb)  
    model_interval = load(path_mapie)  
    scaler = load(path_scaler)  
except FileNotFoundError as e:
    st.error(f"文件缺失: {e}\n请确保模型文件存在于应用目录中！")
    st.stop()

st.markdown("""
<style>
div.stButton > button {
    width: 137.5px !important;
    height: calc(2.8em - 8px) !important;
    min-height: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    line-height: 1 !important;
}
.block-container {
    padding-left: 7rem;
    padding-right: 7rem;
}
.main-title {
    max-width: 100% !important;
    font-size: 32px;
    font-weight: bold;
    white-space: normal !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}
@media (max-width: 800px) {
    .main-title {
        font-size: 24px !important;
    }
}
.spacing-col {
    margin-top: 20px;
}
.custom-spacing {
    height: 20px;
}
.button-container {
    display: flex;
    justify-content: flex-start;
    margin-top: 51px;
}
/* 风险框样式 */
.risk-box {
    background-color: #f0f2f6;
    border-left: 5px solid #ff4b4b;
    padding: 15px;
    border-radius: 5px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-container">'
    '<h1 class="main-title">'
    'Prediction of Compressive Strength Loss Rate in Geopolymer Composites after F-T Cycles'
    '</h1>'
    '</div>',
    unsafe_allow_html=True
)

with st.container():
    col1, spacer1, col2, spacer2, col3 = st.columns([1, 0.2, 1, 0.2, 1])
    
    with col1:
        st.markdown('<div class="spacing-col"></div>', unsafe_allow_html=True)
        feature1 = st.number_input('n(SiO₂)/n(Al₂O₃)', step=0.01, format='%.2f')
        st.markdown('<div class="custom-spacing"></div>', unsafe_allow_html=True)
        feature2 = st.number_input('n(SiO₂)/n(CaO)', step=0.01, format='%.2f')
        st.markdown('<div class="custom-spacing"></div>', unsafe_allow_html=True)
        feature3 = st.number_input('Alkali equivalent (%)', step=0.01, format='%.2f')
        st.markdown('<div class="custom-spacing"></div>', unsafe_allow_html=True)
        feature4 = st.number_input('Activator modulus', step=0.01, format='%.2f')
        st.markdown('<div class="custom-spacing"></div>', unsafe_allow_html=True)
        feature5 = st.number_input('Liquid/Solid', step=0.01, format='%.2f')
    
    with col2:
        st.markdown('<div class="spacing-col"></div>', unsafe_allow_html=True)
        feature6 = st.number_input('Fine aggregate/Binder', step=0.01, format='%.2f')
        feature7 = st.number_input('Coarse aggregate/Binder', step=0.01, format='%.2f')
        feature8 = st.number_input('First day curing temperature (℃)', step=1, format='%d')
        feature9 = st.number_input('Subsequent curing temperature (℃)', step=1, format='%d')
        feature10 = st.number_input('Freezing temperatures (℃)', step=1, format='%d')
    
    with col3:
        feature11 = st.number_input('Thawing temperature (℃)', step=1, format='%d')
        st.markdown('<div class="custom-spacing"></div>', unsafe_allow_html=True)
        feature12 = st.number_input('Curing time (d)', step=1, format='%d')
        st.markdown('<div class="custom-spacing"></div>', unsafe_allow_html=True)
        feature13 = st.number_input('F–T cycles', step=1, format='%d')
        feature14 = st.number_input('Compressive strength before F–T (MPa)', step=0.01, format='%.2f')
        
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        predict_button = st.button('Predict')
        st.markdown('</div>', unsafe_allow_html=True)

feature_values = [
    feature1, feature2, feature3, feature4, feature5,
    feature6, feature7, feature8, feature9, feature10,
    feature11, feature12, feature13, feature14
]

if predict_button:
    input_data = np.array([feature_values])
    input_data_scaled = scaler.transform(input_data)
    
    pred_val = model_point.predict(input_data_scaled)[0]
    
    _, y_pis = model_interval.predict(input_data_scaled, alpha=0.1)
    real_lower = max(0.0, y_pis[0, 0, 0])
    real_upper = y_pis[0, 1, 0]
    

    st.success(f'Compressive strength loss rate (%): {pred_val:.2f}%')
    
    st.markdown(f"""
    <div class="risk-box">
        <div style="font-weight: bold; color: #31333F;">
            Uncertainty Analysis (90% Predictive Interval)
        </div>
        <p style="margin: 5px 0; font-size: 14px; color: #555;">
            Based on Mapie calibration (5-Fold CV), the loss rate falls within:
        </p>
        <p style="font-size: 16px; font-weight: bold;">
            [{real_lower:.2f}% — {real_upper:.2f}%]
        </p>
    </div>
    """, unsafe_allow_html=True)

