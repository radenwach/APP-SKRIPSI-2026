import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import pandas as pd
import plotly.express as px
import os
from streamlit_gsheets import GSheetsConnection
from tensorflow.keras.preprocessing import image as keras_image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="APLIKASI SKRIPSI",
    page_icon="🔍",
    layout="wide"
)

# --- STATE MANAGEMENT ---
if 'menu' not in st.session_state:
    st.session_state.menu = 'Beranda'

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    .result-text { font-size: 24px; font-weight: bold; text-align: center; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #007BFF; background-color: #f0f8ff; border-radius: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_efficientnet_adamw.keras')

try:
    model = load_my_model()
except:
    st.error("File model tidak ditemukan. Pastikan 'model_efficientnet_adamw.keras' ada di folder yang benar.")

# --- FUNGSI PREDIKSI ---
def predict(img, model):
    if img.mode != "RGB": img = img.convert("RGB")
    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array, verbose=0)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigasi Sistem")
    if st.button("BERANDA", use_container_width=True): st.session_state.menu = "Beranda"
    if st.button("PREDIKSI", use_container_width=True): st.session_state.menu = "Prediksi"
    if st.button("STATISTIK", use_container_width=True): st.session_state.menu = "Statistik"
    if st.button("TENTANG", use_container_width=True): st.session_state.menu = "Tentang"

# --- KONTEN HALAMAN ---
if st.session_state.menu == "Beranda":
    st.title("Sistem Deteksi AI-Generated Portrait")
    st.write("Aplikasi untuk mendeteksi potret asli vs AI menggunakan EfficientNetB0.")
    
elif st.session_state.menu == "Prediksi":
    st.header("Deteksi Citra")
    uploaded_file = st.file_uploader("Unggah Potret (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image_uploaded = Image.open(uploaded_file)
        st.image(image_uploaded, width=230)
        if st.button("Mulai Prediksi"):
            with st.spinner('Menganalisis...'):
                res = predict(image_uploaded, model)
                prob_real = float(res[0][0])
                label = "ASLI (REAL)" if prob_real > 0.7 else "AI (FAKE)"
                
                st.markdown(f"<div class='result-text'>PREDIKSI: {label}</div>", unsafe_allow_html=True)
                
                # Simpan ke Google Sheets
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    df_baru = pd.DataFrame([{"Nama Berkas": uploaded_file.name, "Hasil": label, "Akurasi (%)": prob_real*100, "Waktu (s)": 0.1}])
                    df_lama = conn.read(worksheet="Sheet1")
                    conn.update(worksheet="Sheet1", data=pd.concat([df_lama, df_baru], ignore_index=True))
                except Exception as e:
                    st.warning("Gagal menyimpan ke Sheets.")

elif st.session_state.menu == "Statistik":
    st.header("Dasbor Statistik")
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Sheet1", ttl=0).dropna(how="all")
        if not df.empty:
            st.metric("Total Data", len(df))
            fig = px.pie(df, names='Hasil')
            st.plotly_chart(fig)
        else:
            st.info("Riwayat kosong.")
    except Exception as e:
        st.error(f"Error memuat data: {e}")

elif st.session_state.menu == "Tentang":
    st.header("Tentang Penelitian")
    st.write("Peneliti: R. Muhammad Wachyu Fajar Sidik")
