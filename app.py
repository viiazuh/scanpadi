import os
# ================== KONFIGURASI ENV ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import json
import h5py
import requests

# ================== IMPORT TENSORFLOW ==================
import tensorflow as tf
keras = tf.keras
models = keras.models
layers = keras.layers
utils = keras.utils
preprocessing = keras.preprocessing

# ================== KONFIGURASI HALAMAN & TEMA ==================
st.set_page_config(
    page_title="Scan Padi AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS INJECTION UNTUK TEMA HIJAU PUTIH ---
st.markdown("""
    <style>
    /* 1. Background Utama Putih */
    .stApp {
        background-color: #FFFFFF;
        color: #1E1E1E;
    }

    /* 2. Sidebar Hijau Muda Lembut */
    [data-testid="stSidebar"] {
        background-color: #F1F8E9;
        border-right: 1px solid #C8E6C9;
    }

    /* 3. Header & Judul Hijau Tua */
    h1, h2, h3 {
        color: #2E7D32 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* 4. Tombol Utama (Primary Button) */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* 5. Kotak Info & Success */
    .stSuccess {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    .stInfo {
        background-color: #E3F2FD;
        color: #0D47A1;
    }
    
    /* 6. Hapus Padding Atas Berlebih */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ================== FUNGSI DOWNLOAD MODEL ==================
def download_model_from_github():
    url = "https://github.com/viiazuh/scanpadi/raw/main/best_model.h5"
    save_path = "downloaded_model.h5"
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000000:
        return save_path
    
    st.info("📥 Sedang mendownload model asli dari GitHub... (Mohon tunggu)")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Download selesai!")
        return save_path
    except Exception as e:
        st.error(f"Gagal download model: {e}")
        return None

# ================== TRANSLATOR KERAS 3 KE KERAS 2 ==================
def translate_config(config):
    if isinstance(config, dict):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config['batch_shape']
            del config['batch_shape']
        if 'dtype' in config:
            if isinstance(config['dtype'], dict) or config['dtype'] is None:
                config['dtype'] = 'float32'
        ignore_keys = ['time_major', 'ragged', 'build_config', 'compile_config']
        for k in ignore_keys:
            if k in config: del config[k]
        for key, value in config.items():
            if value is not None: translate_config(value)
    elif isinstance(config, list):
        for item in config:
            if item is not None: translate_config(item)

@st.cache_resource
def load_ai_model():
    model_path = download_model_from_github()
    
    if not model_path:
        st.error("❌ Gagal mendapatkan file model.")
        return None

    try:
        with h5py.File(model_path, 'r') as f:
            if 'model_config' not in f.attrs: raise ValueError("No config found")
            config_str = f.attrs.get('model_config')
            if isinstance(config_str, bytes): config_str = config_str.decode('utf-8')
            model_config = json.loads(config_str)

        translate_config(model_config)

        with utils.custom_object_scope({}):
            model = models.model_from_json(json.dumps(model_config))
        
        model.load_weights(model_path)
        return model

    except Exception as e:
        st.error(f"⚠️ Error Load Model: {e}")
        st.warning("Mengaktifkan Mode Simulasi.")
        return None

model = load_ai_model()

# ================== DATABASE INFO ==================
MODEL_LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

DISEASE_KB = {
    "Blas": {"title": "Penyakit Blas", "cause": "Jamur Pyricularia oryzae", "prevention": ["Varietas tahan", "Kurangi Nitrogen"], "treatment": ["Fungisida Tricyclazole"], "color": "#FFEBEE"}, # Merah muda banget
    "Hawar Daun": {"title": "Hawar Daun (Kresek)", "cause": "Bakteri Xanthomonas", "prevention": ["Atur air", "Kurangi Urea"], "treatment": ["Bakterisida tembaga"], "color": "#FFF3E0"}, # Oranye muda
    "Tungro": {"title": "Penyakit Tungro", "cause": "Virus Wereng Hijau", "prevention": ["Tanam serempak"], "treatment": ["Insektisida sistemik"], "color": "#FFEBEE"},
    "Sehat": {"title": "Tanaman Sehat", "cause": "Kondisi Optimal", "prevention": ["Rawat rutin"], "treatment": ["-"], "color": "#E8F5E9"}, # Hijau muda
    "Brown Spot": {"title": "Bercak Coklat", "cause": "Jamur Helminthosporium", "prevention": ["Pupuk Kalium"], "treatment": ["Fungisida"], "color": "#EFEBE9"},
    "Rice Hispa": {"title": "Hama Putih Palsu", "cause": "Kumbang Hispa", "prevention": ["Pangkas daun"], "treatment": ["Insektisida"], "color": "#F5F5F5"}
}

# ================== PREDIKSI ==================
def predict_image(image):
    if model is None: return None
    if image.mode != "RGB": image = image.convert("RGB")
    img = image.resize((224, 224))
    try: x = utils.img_to_array(img)
    except AttributeError: x = preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    
    try:
        pred = model.predict(x)
        idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0])) * 100
        label = MODEL_LABELS[idx] if idx < len(MODEL_LABELS) else "Sehat"
        return {"hasil": label, "confidence": round(confidence, 1), "detail": DISEASE_KB.get(label, DISEASE_KB["Sehat"])}
    except Exception as e:
        return None

# ================== UI CONTROL ==================
if "page" not in st.session_state: st.session_state.page = "home"
if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = None

with st.sidebar:
    st.header("🛠 Mode Developer")
    sim_disease = st.selectbox("Pilih Penyakit", list(DISEASE_KB.keys()))
    if st.button("Tampilkan Info Dummy"):
        st.session_state.result = {"hasil": sim_disease, "confidence": 100.0, "detail": DISEASE_KB[sim_disease], "image": Image.new('RGB', (200, 200), color=(200, 255, 200))}
        st.session_state.page = "result"; st.rerun()

# ================== HALAMAN ==================
def home_page():
    # Header Banner Sederhana
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌾 Dashboard Scan Padi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Deteksi dini penyakit padi menggunakan Artificial Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([3,1])
    with col1:
        st.info("👋 Halo Petani! Siap memindai tanaman hari ini?")
    with col2:
        if st.button("➕ Scan Baru", use_container_width=True): st.session_state.page = "scan"
    
    st.subheader("📋 Riwayat Scan")
    if not st.session_state.history:
        st.markdown("""
            <div style='background-color: #F9F9F9; padding: 20px; border-radius: 10px; text-align: center; border: 1px dashed #CCC;'>
                <p style='color: #888;'>Belum ada data scan. Klik tombol <b>+ Scan Baru</b> di atas.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for i, h in enumerate(st.session_state.history):
            with st.container():
                c1, c2 = st.columns([1,3])
                with c1: st.image(h["image"], use_column_width=True)
                with c2:
                    st.markdown(f"<h4 style='margin:0; color:#2E7D32;'>{h['title']}</h4>", unsafe_allow_html=True)
                    st.caption(f"Akurasi: {h['confidence']}% | {h['date']}")
                    if st.button("Hapus", key=f"del_{i}"): st.session_state.history.pop(i); st.rerun()
            st.divider()

def scan_page():
    st.markdown("## 📸 Scan Tanaman")
    st.write("Ambil foto daun padi secara close-up atau upload dari galeri.")
    
    if model is None: st.warning("⚠️ Model sedang didownload atau offline.")
    
    col1, col2 = st.columns(2)
    with col1: img_file = st.camera_input("Ambil Foto")
    with col2: upl_file = st.file_uploader("Upload File", type=["jpg","png","jpeg"])
    
    image = img_file if img_file else upl_file
    if image:
        image = Image.open(image)
        st.image(image, caption="Preview", use_column_width=True)
        
        st.markdown("---")
        if st.button("🔍 Analisis AI Sekarang", use_container_width=True, disabled=(model is None)):
            with st.spinner("Sedang menganalisis struktur daun..."):
                res = predict_image(image)
                if res:
                    res["image"] = image
                    st.session_state.result = res
                    st.session_state.page = "result"; st.rerun()
    
    if st.button("⬅ Kembali ke Home"): st.session_state.page = "home"

def result_page():
    if not st.session_state.result: 
        st.session_state.page = "home"
        st.rerun()
        return

    r = st.session_state.result
    info = r["detail"]
    
    # Header Hasil
    st.markdown(f"<h2 style='text-align:center; color:#2E7D32;'>Hasil Analisis</h2>", unsafe_allow_html=True)
    
    col_img, col_stat = st.columns([1, 1])
    with col_img:
        st.image(r["image"], use_column_width=True, caption="Foto Daun")
    with col_stat:
        st.markdown(f"### {info['title']}")
        
        # Indikator Confidence
        if r['confidence'] > 80:
            st.success(f"✅ Tingkat Keyakinan: **{r['confidence']}%** (Sangat Yakin)")
        else:
            st.warning(f"⚠️ Tingkat Keyakinan: **{r['confidence']}%** (Kurang Yakin)")
            
        st.info(f"**Penyebab:** {info['cause']}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: 
        st.markdown("#### 🛡 Pencegahan")
        for p in info["prevention"]:
            st.write(f"• {p}")
            
    with c2: 
        st.markdown("#### 💊 Pengobatan")
        for t in info["treatment"]:
            st.write(f"• {t}")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("💾 Simpan ke Riwayat", use_container_width=True):
            st.session_state.history.insert(0, {"title": info["title"], "confidence": r["confidence"], "image": r["image"], "date": datetime.now().strftime("%d-%m %H:%M")})
            st.session_state.page = "home"; st.rerun()
            
    with col_btn2:
        if st.button("🔄 Scan Lagi", use_container_width=True): 
            st.session_state.page = "scan"
            st.rerun()

# Routing
if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
