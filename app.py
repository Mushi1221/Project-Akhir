import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title('Prediksi Jarak Tempuh Kendaraan')
st.markdown("""
    ### 🧠 Tentang Model Prediksi

    Model yang digunakan untuk memprediksi jarak tempuh pada kendaraan adalah **Regresi Linear (Linear Regression)**.  
            
    Model ini bekerja dengan mengukur seberapa besar pengaruh masing-masing faktor (seperti waktu tempuh, kecepatan rata rata, jenis jalan, kondisi cuaca, kapasitas penumpang, dan umur kendaraan) berdasarkan data yang diinput.

    Semakin besar nilai koefisien dari suatu faktor, maka faktor tersebut cenderung memiliki pengaruh lebih besar terhadap jarak tempuh pada kendaraan.
    """)
# Load dataset dari file CSV
df = pd.read_csv('dataset_jarak_tempuh_kendaraan.csv', sep=';')
st.write("Kolom yang tersedia di dataset:", df.columns.tolist())
st.write(df.head())
df.columns = df.columns.str.strip()

if 'jenis_jalan' not in df.columns or 'cuaca' not in df.columns:
    st.error("Kolom 'jenis_jalan' atau 'cuaca' tidak ditemukan di dataset.")
    st.write("Kolom yang ditemukan:", df.columns.tolist())
    st.stop()
# Encode fitur kategorikal
df['kode_jalan'] = df['jenis_jalan'].map({'Tol': 2, 'Jalan Raya': 1, 'Jalan Kota': 0})
df['kode_cuaca'] = df['cuaca'].map({'Cerah': 2, 'Hujan': 1, 'Berkabut': 0})

# Fitur dan label
X = df[['waktu_tempuh', 'kecepatan', 'kode_jalan', 'kode_cuaca', 'kapasitas_penumpang', 'umur_kendaraan']].values
y = df['jarak_tempuh'].values

# Model
model = LinearRegression()
model.fit(X, y)

# Input user
waktu_tempuh = st.number_input('Waktu Tempuh (jam)', min_value=0.0, max_value=24.0, value=1.0)
kecepatan = st.number_input('Kecepatan Rata-rata (km/jam)', min_value=10, max_value=200, value=60)
jenis_jalan = st.selectbox('Jenis Jalan', ['Tol', 'Jalan Raya', 'Jalan Kota'])
cuaca = st.selectbox('Kondisi Cuaca', ['Cerah', 'Hujan', 'Berkabut'])
kapasitas_penumpang = st.number_input('Kapasitas Penumpang', min_value=1, max_value=10, value=4)
umur_kendaraan = st.number_input('Umur Kendaraan (tahun)', min_value=0, max_value=20, value=3)

kode_jalan = {'Tol': 2, 'Jalan Raya': 1, 'Jalan Kota': 0}[jenis_jalan]
kode_cuaca = {'Cerah': 2, 'Hujan': 1, 'Berkabut': 0}[cuaca]

fitur_input = np.array([[waktu_tempuh, kecepatan, kode_jalan, kode_cuaca, kapasitas_penumpang, umur_kendaraan]])
prediksi = model.predict(fitur_input)

st.write(f'Jarak Tempuh Kendaraan Diperkirakan: **{prediksi[0]:.2f} km**')

st.sidebar.info("Dibuat oleh Mushi Dan TIM ")
st.sidebar.markdown("---")
st.sidebar.header("Tentang Proyek")
st.sidebar.write("Praktikum Data Mining 2025")
st.sidebar.write("Prediksi Jarak Tempuh Kendaraan")

with st.expander("Lihat Dataset"):
    st.dataframe(df)
