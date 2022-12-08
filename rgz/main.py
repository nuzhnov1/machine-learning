# Подключаем необходимые библиотеки
import datetime
import tensorflow as tf
import json
import pickle

import streamlit as st
import pandas as pd

# Импорт предварительно натренированных моделей

model = tf.keras.models.load_model(r'tf.h5')

with open(r'pickleLin.bin', mode='rb') as filestream:
    model_lin = pickle.load(filestream)

with open(r'pickleScaler.bin', mode='rb') as filestream:
    scalerNormX, scalerNormY = pickle.load(filestream)

with open(r'm1.json', mode='r') as filestream:
    m1dict = json.load(filestream)

with open(r'm2.json', mode='r') as filestream:
    m2dict = json.load(filestream)

# Создание объектов переключателей:

val_AT = st.sidebar.slider(
    'Температура окружающей среды',
    min_value   =   0.0,
    max_value   =   60.0,
    value       =   20.0,
    step        =   0.5
)

val_V = st.sidebar.slider(
    'Разряженность выхлопных газов',
    min_value   =   15.0,
    max_value   =   100.0,
    value       =   54.0,
    step        =   0.01
)

val_AP = st.sidebar.slider(
    'Давление окружающей среды',
    min_value   =   980.0,
    max_value   =   1050.0,
    value       =   1013.0,
    step        =   0.01
)

val_RH = st.sidebar.slider(
    'Относительная влажность',
    min_value   =   20.0,
    max_value   =   100.0,
    value       =   73.0,
    step        =   0.01
)

# Компоновка считанных данных в таблицу
dfX_custom = pd.DataFrame(
    data=[[val_AT,
           val_V,
           val_AP,
           val_RH]],
    columns=m1dict["features"]
)

st.write("X в исходных шкалах")
st.write(dfX_custom)

# Вывод нормализованой таблицы
st.write("X нормализованные")
dfX_custom_scaled = pd.DataFrame(
    data=scalerNormX.transform(dfX_custom),
    columns=m2dict["features"]
)
st.write(dfX_custom_scaled)

# Вычисление целевого значения по входным данным и его вывод
col1, col2 = st.columns(2)
with col1:
    st.header("Линейная регрессия")
    y_pred = model_lin.predict(dfX_custom)
    st.write(f"R^2 = {m1dict['R2']}")
    st.write(f"RMSE = {m1dict['RMSE']}")
    st.write("Y в исходной шкале")
    y_pred = pd.DataFrame(
        data    =   y_pred,
        columns =   m1dict["target"]
    )
    st.write(y_pred)
with col2:
    st.header("Нейронная сеть")
    with tf.device('/GPU:0'):
        yNorm_pred = model.predict(dfX_custom_scaled)
    
    st.write(f"R^2 = {m2dict['R2']}")
    st.write(f"RMSE = {m2dict['RMSE']}")
    st.write("Y нормализованый")
    yNorm_pred = pd.DataFrame(
        data    =   yNorm_pred,
        columns =   m2dict["target"]
    )
    st.write(yNorm_pred)
    st.write("Y в исходной шкале")
    yNorm_pred = pd.DataFrame(
        data    =   scalerNormY.inverse_transform(yNorm_pred),
        columns =   m2dict["target"]
    )
    st.write(yNorm_pred)
