import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import pandas as pd
import plotly.express as px
from streamlit_gsheets import GSheetsConnection
from tensorflow.keras.preprocessing import image as keras_image
import cv2  # Pustaka untuk Gatekeeper (OpenCV)
import os
import urllib.request

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Deteksi Potret AI",
    page_icon="🔍",
    layout="wide"
)

# --- STATE MANAGEMENT UNTUK MENU NAVIGATION ---
if 'menu' not in st.session_state:
    st.session_state.menu = 'Beranda'

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Mengatur jarak (padding) blok utama agar konten lebih ke atas */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main {
        background-color: #f5f7f9;
    }
    /* Memastikan desain tombol kotak biru tetap konsisten */
    .stButton>button {
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 0;
    }
    
    /* --- GAYA BARU UNTUK BOX UPLOAD --- */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #007BFF; /* Garis putus-putus warna biru */
        background-color: #f0f8ff;  /* Latar belakang biru sangat muda */
        border-radius: 15px;        /* Sudut membulat */
        padding: 20px;
        transition: all 0.3s ease;  /* Efek transisi halus */
    }
    
    /* Efek saat mouse diarahkan ke kotak upload */
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #e1effe;
        border-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL UTAMA (EfficientNetB0) ---
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('model_efficientnet_adamw.keras')
    return model

model = load_my_model()

# --- FUNGSI GATEKEEPER (PRA-SELEKSI OOD DATA + DEBUG MODE) ---
def cek_validitas_potret(img):
    """
    Memvalidasi apakah gambar yang diunggah memiliki 
    struktur anatomi manusia menggunakan OpenCV Haar Cascade lokal.
    Dilengkapi dengan notifikasi debug otomatis untuk pelacakan server.
    """
    try:
        import cv2
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Mengambil path absolut dari folder tempat app.py berada
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        
        # CEK 1: Apakah file XML benar-benar ada di server?
        if not os.path.exists(cascade_path):
            st.toast("⚠️ LOG: File XML tidak ditemukan di direktori server! Sistem melakukan bypass.", icon="⚠️")
            return True
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # CEK 2: Apakah OpenCV gagal memuat file XML tersebut?
        if face_cascade.empty():
            st.toast("⚠️ LOG: OpenCV gagal memuat Haar Cascade! Sistem melakukan bypass.", icon="⚠️")
            return True
            
        # PROSES DETEKSI: Parameter diperketat secara maksimal
        # minNeighbors dinaikkan ke 12 (Sangat Ketat)
        # minSize dinaikkan ke 80x80 (Mengabaikan pola kecil seperti tekstur bulu hewan)
        fitur_anatomi = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=12, 
            minSize=(80, 80)
        )
        
        # CEK 3: Hasil Deteksi
        if len(fitur_anatomi) > 0:
            st.toast(f"🎯 LOG: Struktur potret tervalidasi ({len(fitur_anatomi)} objek ditemukan). Diteruskan ke EfficientNetB0.", icon="✅")
            return True
        else:
            st.toast("🚫 LOG: Objek bukan potret manusia! Proses dihentikan.", icon="🛑")
            return False
            
    except Exception as e:
        # Jika terjadi error library di level server
        st.toast(f"❌ LOG ERROR OPENCV: {str(e)}. Sistem melakukan bypass keamanan.", icon="❌")
        return True
        
# --- FUNGSI PREDIKSI ---
def predict(img, model):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img_resized = img.resize((224, 224), Image.NEAREST)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    return prediction

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigasi Sistem")
    
    # REVISI: Mengganti use_container_width=True menjadi width="stretch"
    if st.button("BERANDA", width="stretch"):
        st.session_state.menu = "Beranda"
    if st.button("PREDIKSI", width="stretch"):
        st.session_state.menu = "Prediksi"
    if st.button("STATISTIK", width="stretch"):
        st.session_state.menu = "Statistik"
    if st.button("TENTANG", width="stretch"):
        st.session_state.menu = "Tentang"

menu = st.session_state.menu

# --- BERANDA ---
if menu == "Beranda":
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    col_kiri, col_kanan = st.columns([1.5, 1])
    
    with col_kiri:
        st.title("Sistem Deteksi AI-Generated Portrait")
        st.write("""
        Aplikasi ini digunakan untuk mendeteksi apakah suatu citra merupakan potret manusia asli atau hasil sintesis generasi AI (*deepfake*). 
        Menggunakan model *Convolutional Neural Network* jenis **EfficientNetB0**, sistem menganalisis anomali piksel, tekstur, hingga konteks pencahayaan gambar secara menyeluruh.
        
        Gunakan menu di sidebar untuk mulai melakukan prediksi.
        """)
        st.info("💡 Sistem ini dirancang untuk membaca konteks pencahayaan, tekstur, dan latar belakang potret secara utuh.")

    with col_kanan:
        try:
            # st.image tetap menggunakan use_container_width karena parameternya beda
            st.image("ilustrasi.png", use_container_width=True)
        except:
            st.empty() 

# --- PREDIKSI ---
elif menu == "Prediksi":
    st.header("Deteksi Citra")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Unggah Potret Manusia (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_uploaded = Image.open(uploaded_file)
            st.image(image_uploaded, caption="Citra yang Diunggah", width=230)
            
    with col2:
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        st.subheader("Hasil Prediksi")
        if uploaded_file is not None:
            # REVISI: Mengganti use_container_width=True menjadi width="stretch"
            if st.button("Mulai Prediksi", width="stretch"):
                
                # --- 1. PROSES GATEKEEPER ---
                with st.spinner('Memverifikasi struktur citra...'):
                    is_potret_valid = cek_validitas_potret(image_uploaded)
                
                # --- 2. LOGIKA PERCABANGAN ---
                if not is_potret_valid:
                    # JIKA GAGAL: Gambar bukan potret manusia
                    st.error("Sistem mendeteksi bahwa gambar ini BUKAN citra potret manusia. Silakan unggah gambar yang relevan untuk menghindari kesalahan deteksi.")
                
                else:
                    # JIKA LOLOS: Eksekusi model utama EfficientNetB0
                    with st.spinner('Menganalisis fitur citra dengan EfficientNetB0...'):
                        start_time = time.time()
                        result = predict(image_uploaded, model)
                        end_time = time.time()
                        
                        waktu_inferensi = end_time - start_time
                        prob_real = float(result[0][0])
                        prob_fake = 1 - prob_real
                        
                        # --- IMPLEMENTASI THRESHOLD ---
                        THRESHOLD = 0.6
                        if prob_real > THRESHOLD:
                            hasil_label = "ASLI (REAL)"
                            prob_final = prob_real
                            warna_teks = '#155724' 
                            bg_color = '#d4edda'   
                        else:
                            hasil_label = "AI (FAKE)"
                            prob_final = prob_fake
                            warna_teks = '#FF0000' 
                            bg_color = '#f8d7da'   
                        
                        st.markdown(f"""
                            <div style='background-color: {bg_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid {warna_teks}40;'>
                                <div class='result-text' style='color: {warna_teks};'>
                                    PREDIKSI: {hasil_label}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(f"Waktu Pemrosesan: {waktu_inferensi:.2f} detik")
                        
                        # --- SIMPAN PERMANEN KE GOOGLE SHEETS ---
                        try:
                            conn = st.connection("gsheets", type=GSheetsConnection)
                            df_lama = conn.read(worksheet="Sheet1", usecols=list(range(4)))
                            
                            df_baru = pd.DataFrame([{
                                "Nama Berkas": uploaded_file.name,
                                "Hasil": hasil_label,
                                "Akurasi (%)": prob_final * 100,
                                "Waktu (s)": waktu_inferensi
                            }])
                            
                            df_lama = df_lama.dropna(how="all")
                            df_update = pd.concat([df_lama, df_baru], ignore_index=True)
                            
                            conn.update(worksheet="Sheet1", data=df_update)
                        except Exception as e:
                            st.warning(f"Terjadi kesalahan saat menyimpan ke Google Sheets: {e}")
                            
        else:
            st.info("Silakan unggah potret terlebih dahulu untuk memulai prediksi.")

# --- STATISTIK ---
elif menu == "Statistik":
    st.header("Dasbor Statistik")
    st.write("Panel pemantauan analitik penggunaan sistem secara keseluruhan.")
    
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_stat = conn.read(worksheet="Sheet1", usecols=list(range(4)), ttl=0)
        df_stat = df_stat.dropna(how="all")
        
        if len(df_stat) > 0:
            total_data = len(df_stat)
            rata_waktu = df_stat["Waktu (s)"].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Total Citra Diproses", value=f"{total_data} Citra")
            col2.metric(label="Akurasi Model Yang Digunakan", value="87.0%") 
            col3.metric(label="Rata-rata Waktu Inferensi", value=f"{rata_waktu:.2f} s")
            
            st.markdown("---")
            
            st.subheader("Distribusi Klasifikasi")
            distribusi = df_stat["Hasil"].value_counts().reset_index()
            distribusi.columns = ['Klasifikasi', 'Jumlah']
            
            fig = px.pie(distribusi, values='Jumlah', names='Klasifikasi', hole=0.4, 
                         color_discrete_map={'AI (FAKE)':'#ff9999', 'ASLI (REAL)':'#66b3ff'})
            # REVISI: st.plotly_chart tetap pakai use_container_width (ini beda dengan st.button)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Tampilkan Detail Riwayat Prediksi"):
                df_history = df_stat.copy()
                df_display = df_history.drop(columns=["Akurasi (%)", "Tingkat Keyakinan"], errors='ignore')
                df_display.index = df_display.index + 1 
                # REVISI: st.dataframe tetap pakai use_container_width
                st.dataframe(df_display, use_container_width=True)
                
                # REVISI: Mengganti use_container_width=True menjadi width="stretch"
                if st.button("Hapus Semua Riwayat", width="stretch"):
                    df_kosong = pd.DataFrame(columns=["Nama Berkas", "Hasil", "Akurasi (%)", "Waktu (s)"])
                    conn.update(worksheet="Sheet1", data=df_kosong)
                    st.rerun()
        else:
            st.info("Riwayat prediksi masih kosong. Silakan lakukan deteksi terlebih dahulu.")
            
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Sheets. Error: {e}")

# --- TENTANG ---
elif menu == "Tentang":
    st.header("Tentang Penelitian")
    st.write("**Peneliti:** R. Muhammad Wachyu Fajar Sidik")
    st.write("**Instansi:** Universitas PGRI Kediri (UNP Kediri)")
    st.write("**Topik:** Deteksi Manipulasi AI pada Citra Potret Menggunakan Arsitektur Lightweight EfficientNetB0 dengan Optimizer AdamW.")
    st.write("Sistem ini dibangun sebagai perwujudan purwarupa dari penelitian akademis untuk meningkatkan keamanan digital dalam mendeteksi ancaman potret sintesis (*deepfake*).")
