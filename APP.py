#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from joblib import load
import numpy as np

model = load('ML_model.joblib')
scaler = load('StandardScaler.joblib')


# —— 全局 CSS 覆写 st.button 大小及标题响应式换行缩放 —— 
st.markdown("""
    <style>
      /* 让 calc(2.8em - 15px) 真正生效，需要同时重置这三项 */
      div.stButton > button {
        width: 137.5px !important;
        height: calc(2.8em - 8px) !important;
        min-height: 0 !important;               /* 取消最小高度 */
        padding-top: 0 !important;              /* 去掉上内边距 */
        padding-bottom: 0 !important;           /* 去掉下内边距 */
        line-height: 1 !important;              /* 避免行高撑开 */
      }

      /* 页面左右内边距 */
      .block-container {
          padding-left: 7rem;
          padding-right: 7rem;
      }

      /* 主标题：去除固定宽度，自动换行，响应式字体 */
      .main-title {
          max-width: 100% !important;
          font-size: 32px;
          font-weight: bold;
          white-space: normal !important;
          overflow-wrap: break-word !important;
          word-break: break-word !important;
      }

      /* 在窄屏或小窗口时，将标题字体调小 */
      @media (max-width: 800px) {
        .main-title {
          font-size: 24px !important;
        }
      }

      /* 用于各输入区与按钮的间距 */
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
    </style>
""", unsafe_allow_html=True)


# —— 页面标题 ——  
st.markdown(
    '<div class="title-container">'
    '<h1 class="main-title">'
    'Prediction of Compressive Strength Loss Rate in Geopolymer Composites after F-T Cycles'
    '</h1>'
    '</div>',
    unsafe_allow_html=True
)

# —— 输入区布局 —— 
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

        # 按钮容器
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        predict_button = st.button('Predict')
        st.markdown('</div>', unsafe_allow_html=True)

# —— 收集输入并预测 —— 
feature_values = [
    feature1, feature2, feature3, feature4, feature5,
    feature6, feature7, feature8, feature9, feature10,
    feature11, feature12, feature13, feature14
]

if predict_button:
    input_data = np.array([feature_values])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.success(f'Compressive strength loss rate (%): {prediction[0]:.2f}%')

