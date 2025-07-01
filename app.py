import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title('📈Prediksi Jarak Tempuh Kendaraan')

st.markdown("""
    ### 🧠 Tentang Model Prediksi

    Model yang digunakan untuk memprediksi jarak tempuh pada kendaraan adalah **Regresi Linear (Linear Regression)**.  
            
    Model ini bekerja dengan mengukur seberapa besar pengaruh masing-masing faktor (seperti waktu tempuh, kecepatan rata rata, jenis jalan, kondisi cuaca, kapasitas penumpang, dan umur kendaraan) berdasarkan data yang diinput.

    Semakin besar nilai koefisien dari suatu faktor, maka faktor tersebut cenderung memiliki pengaruh lebih besar terhadap jarak tempuh pada kendaraan.
    """)

# Input fitur
waktu_tempuh = st.number_input('Waktu Tempuh (jam)', min_value=0.0, max_value=24.0, value=1.0)
kecepatan = st.number_input('Kecepatan Rata-rata (km/jam)', min_value=10, max_value=200, value=60)
jenis_jalan = st.selectbox('Jenis Jalan', ['Tol', 'Jalan Raya', 'Jalan Kota'])
cuaca = st.selectbox('Kondisi Cuaca', ['Cerah', 'Hujan', 'Berkabut'])
kapasitas_penumpang = st.number_input('Kapasitas Penumpang', min_value=1, max_value=10, value=4)
umur_kendaraan = st.number_input('Umur Kendaraan (tahun)', min_value=0, max_value=20, value=3)

# Encode fitur kategorikal
kode_jalan = {'Tol': 2, 'Jalan Raya': 1, 'Jalan Kota': 0}[jenis_jalan]
kode_cuaca = {'Cerah': 2, 'Hujan': 1, 'Berkabut': 0}[cuaca]

# Dummy data pelatihan
# Fitur: [waktu_tempuh, kecepatan, kode_jalan, kode_cuaca, kapasitas_penumpang, umur_kendaraan]
X = np.array([
    [1, 60, 2, 2, 4, 3],
    [2, 80, 2, 2, 2, 1],
    [1, 40, 1, 1, 4, 5],
    [2, 50, 1, 0, 3, 7],
    [1, 30, 0, 1, 2, 10],
    [2, 40, 0, 0, 5, 12],
    [1.5, 70, 2, 1, 4, 2],
    [2, 90, 2, 2, 3, 1]
])
# Label: jarak tempuh (km)
y = np.array([60, 160, 40, 100, 30, 80, 105, 180])

# Model regresi linear
model = LinearRegression()
model.fit(X, y)

# Prediksi
fitur_input = np.array([[waktu_tempuh, kecepatan, kode_jalan, kode_cuaca, kapasitas_penumpang, umur_kendaraan]])
prediksi = model.predict(fitur_input)

st.write(f'Prediksi Jarak Tempuh: {prediksi[0]:.2f} km')

st.sidebar.info("Dibuat oleh Mushi Dan TIM ")
st.sidebar.markdown("---")
st.sidebar.header("Tentang Proyek")
st.sidebar.write("Praktikum Data Mining 2025")
st.sidebar.write("Prediksi Jarak Tempuh Kendaraan")
