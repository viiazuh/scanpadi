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

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Scan Padi AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS TEMA HIJAU & FIX KAMERA ==================
st.markdown("""
    <style>
    /* 1. Background Putih & Teks Gelap */
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    
    /* 2. Sidebar Hijau Lembut */
    [data-testid="stSidebar"] { background-color: #F1F8E9; border-right: 1px solid #C8E6C9; }
    
    /* 3. Header Hijau */
    h1, h2, h3 { color: #2E7D32 !important; }
    
    /* 4. Tombol Utama Hijau */
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 10px; border: none;
        padding: 0.5rem 1rem; font-weight: bold; transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #388E3C; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }

    /* 5. FIX TAMPILAN KAMERA (Solusi Layar Hitam) */
    /* Mengubah background area kamera jadi hijau transparan */
    [data-testid="stCamera"] {
        background-color: #1b5e20 !important; /* Hijau Tua Gelap biar kontras */
        border-radius: 15px;
        padding: 10px;
    }
    
    /* Memaksa Tombol Kamera (Take Photo) jadi Putih/Terlihat */
    button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border: 2px solid white !important;
    }
    
    /* 6. Pesan Error/Warning */
    .stWarning { background-color: #FFF3E0; color: #E65100; }
    .stSuccess { background-color: #E8F5E9; color: #2E7D32; }
    
    /* Hapus padding atas */
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ================== DOWNLOAD & LOAD MODEL ==================
def download_model_from_github():
    url = "https://github.com/viiazuh/scanpadi/raw/main/best_model.h5"
    save_path = "downloaded_model.h5"
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000000: return save_path
    
    st.info("📥 Sedang mendownload model... (Mohon tunggu)")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        st.success("✅ Download selesai!")
        return save_path
    except Exception as e:
        st.error(f"Gagal download: {e}"); return None

def translate_config(config):
    if isinstance(config, dict):
        if 'batch_shape' in config: config['batch_input_shape'] = config['batch_shape']; del config['batch_shape']
        if 'dtype' in config: config['dtype'] = 'float32'
        for k in ['time_major', 'ragged', 'build_config', 'compile_config']:
            if k in config: del config[k]
        for v in config.values(): translate_config(v)
    elif isinstance(config, list):
        for i in config: translate_config(i)

@st.cache_resource
def load_ai_model():
    path = download_model_from_github()
    if not path: return None
    try:
        with h5py.File(path, 'r') as f:
            cfg = json.loads(f.attrs.get('model_config').decode('utf-8') if isinstance(f.attrs.get('model_config'), bytes) else f.attrs.get('model_config'))
        translate_config(cfg)
        with utils.custom_object_scope({}): model = models.model_from_json(json.dumps(cfg))
        model.load_weights(path)
        return model
    except Exception as e:
        st.error(f"Error Model: {e}"); return None

model = load_ai_model()

# ================== DATABASE INFO ==================
MODEL_LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

DISEASE_KB = {
    "Blas": {"title": "Penyakit Blas", "cause": "Jamur Pyricularia oryzae", "prevention": ["Gunakan varietas tahan", "Kurangi Nitrogen"], "treatment": ["Fungisida Tricyclazole"], "color": "#FFEBEE"},
    "Hawar Daun": {"title": "Hawar Daun (Kresek)", "cause": "Bakteri Xanthomonas", "prevention": ["Atur air", "Kurangi Urea"], "treatment": ["Bakterisida tembaga"], "color": "#FFF3E0"},
    "Tungro": {"title": "Penyakit Tungro", "cause": "Virus Wereng Hijau", "prevention": ["Tanam serempak"], "treatment": ["Insektisida sistemik"], "color": "#FFEBEE"},
    "Sehat": {"title": "Tanaman Sehat", "cause": "Kondisi Optimal", "prevention": ["Rawat rutin"], "treatment": ["-"], "color": "#E8F5E9"},
    # INFO KHUSUS BUKAN PADI
    "Bukan Padi": {"title": "Objek Tidak Dikenali", "cause": "Foto yang Anda scan sepertinya bukan daun padi, atau gambarnya terlalu buram/gelap.", "prevention": ["Pastikan foto fokus ke daun", "Cari pencahayaan yang cukup"], "treatment": ["Scan ulang daun padi yang benar"], "color": "#ECEFF1"}
}

# ================== PREDIKSI CERDAS (LOGIKA BARU) ==================
def predict_image(image):
    if model is None: return None
    
    # 1. Preprocess Gambar
    if image.mode != "RGB": image = image.convert("RGB")
    img = image.resize((224, 224))
    try: x = utils.img_to_array(img)
    except: x = preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    
    try:
        # 2. Prediksi AI
        pred = model.predict(x)
        idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0])) * 100
        
        # 3. LOGIKA FILTER "BUKAN PADI"
        # Jika AI yakinnya di bawah 65%, kita anggap itu bukan padi (objek asing)
        # Angka 65 bisa kamu naik turunkan sesuai selera
        THRESHOLD = 65.0 
        
        if confidence < THRESHOLD:
            label = "Bukan Padi"
            # Kita set confidence 0 agar user waspada
            confidence = 0 
        else:
            label = MODEL_LABELS[idx] if idx < len(MODEL_LABELS) else "Sehat"
            
        return {
            "hasil": label, 
            "confidence": round(confidence, 1), 
            "detail": DISEASE_KB.get(label, DISEASE_KB["Bukan Padi"])
        }
    except Exception:
        return None

# ================== UI PAGES ==================
if "page" not in st.session_state: st.session_state.page = "home"
if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = None

with st.sidebar:
    st.header("🛠 Mode Developer")
    if st.button("Reset Aplikasi"): st.session_state.clear(); st.rerun()

def home_page():
    st.markdown("<h1 style='text-align: center;'>🌾 Dashboard Scan Padi</h1>", unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns([3,1])
    with col1: st.info("👋 Halo Petani! Gunakan pencahayaan yang cukup saat mengambil foto.")
    with col2: 
        if st.button("➕ Scan Baru", use_container_width=True): st.session_state.page = "scan"

    st.subheader("📋 Riwayat")
    if not st.session_state.history:
        st.markdown("<div style='text-align: center; color: #888;'>Belum ada riwayat.</div>", unsafe_allow_html=True)
    else:
        for i, h in enumerate(st.session_state.history):
            with st.container():
                c1, c2 = st.columns([1,3])
                with c1: st.image(h["image"], use_column_width=True)
                with c2:
                    st.markdown(f"**{h['title']}**")
                    if h['title'] == "Objek Tidak Dikenali":
                        st.caption("⚠️ Bukan Padi / Gambar Buram")
                    else:
                        st.caption(f"Akurasi: {h['confidence']}%")
                    if st.button("Hapus", key=f"del_{i}"): st.session_state.history.pop(i); st.rerun()
            st.divider()

def scan_page():
    st.markdown("## 📸 Scan Tanaman")
    if model is None: st.warning("⚠️ Sedang menyiapkan AI, harap tunggu...")
    
    # CSS di atas sudah menangani warna background kamera
    col1, col2 = st.columns(2)
    with col1: img_file = st.camera_input("Buka Kamera")
    with col2: upl_file = st.file_uploader("Upload Foto", type=["jpg","png","jpeg"])
    
    image = img_file if img_file else upl_file
    if image:
        img_open = Image.open(image)
        st.image(img_open, caption="Preview Foto", use_column_width=True)
        
        st.markdown("---")
        if st.button("🔍 Analisis AI Sekarang", use_container_width=True, disabled=(model is None)):
            with st.spinner("Sedang menganalisis..."):
                res = predict_image(img_open)
                if res:
                    res["image"] = img_open
                    st.session_state.result = res
                    st.session_state.page = "result"; st.rerun()
    
    if st.button("⬅ Kembali"): st.session_state.page = "home"

def result_page():
    if not st.session_state.result: st.session_state.page = "home"; st.rerun(); return
    r = st.session_state.result
    info = r["detail"]
    
    st.markdown(f"<h2 style='text-align:center;'>Hasil Analisis</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1,1])
    with c1: st.image(r["image"], use_column_width=True)
    with c2:
        st.markdown(f"### {info['title']}")
        
        # LOGIKA TAMPILAN JIKA BUKAN PADI
        if r['hasil'] == "Bukan Padi":
            st.error("⚠️ **Peringatan:** Objek ini kemungkin besar **BUKAN PADI** atau gambar tidak jelas.")
        else:
            if r['confidence'] > 80: st.success(f"✅ Yakin: **{r['confidence']}%**")
            else: st.warning(f"⚠️ Agak Ragu: **{r['confidence']}%**")
            
        st.info(f"**Info:** {info['cause']}")

    # Jika bukan padi, sembunyikan pengobatan
    if r['hasil'] != "Bukan Padi":
        st.markdown("---")
        kp, ko = st.columns(2)
        with kp: 
            st.markdown("#### 🛡 Pencegahan")
            for p in info["prevention"]: st.write(f"• {p}")
        with ko: 
            st.markdown("#### 💊 Pengobatan")
            for t in info["treatment"]: st.write(f"• {t}")

    st.markdown("---")
    if st.button("💾 Simpan", use_container_width=True):
        st.session_state.history.insert(0, {"title": info["title"], "confidence": r["confidence"], "image": r["image"], "date": datetime.now().strftime("%d-%m")})
        st.session_state.page = "home"; st.rerun()
    if st.button("🔄 Scan Lagi", use_container_width=True): st.session_state.page = "scan"; st.rerun()

if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
