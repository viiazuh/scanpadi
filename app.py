import os
# ================== KONFIGURASI ENV ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import json
import h5py

# ================== IMPORT TENSORFLOW ==================
import tensorflow as tf
# Shortcut agar code lebih ringkas dan aman
keras = tf.keras
models = keras.models
layers = keras.layers
utils = keras.utils
preprocessing = keras.preprocessing

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="AI Kesehatan Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== DEBUGGING (Cari File) ==================
def find_model_file():
    if os.path.exists("best_model.h5"): return os.path.abspath("best_model.h5")
    app_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(app_dir, "best_model.h5")
    if os.path.exists(file_path): return file_path
    return None

# ================== ULTRA-SMART LOADER ==================
def clean_config(config):
    """
    Membersihkan konfigurasi Keras 3 secara agresif namun aman (anti-crash).
    """
    if isinstance(config, dict):
        # 1. Daftar kunci Keras 3 yang tidak dikenali Keras 2
        keys_to_remove = [
            'time_major', 'ragged', 'batch_shape', 'batch_input_shape', 
            'build_config', 'compile_config', 'registered_name'
        ]
        
        for key in keys_to_remove:
            if key in config:
                del config[key]
        
        # 2. Perbaiki dtype (Keras 3 pakai dict, Keras 2 pakai string)
        if 'dtype' in config:
            if isinstance(config['dtype'], dict):
                config['dtype'] = 'float32'
            elif config['dtype'] is None: # Fix untuk error NoneType
                config['dtype'] = 'float32'

        # 3. Rekursif aman (Cek apakah value tidak None sebelum diproses)
        for key, value in config.items():
            if value is not None:
                clean_config(value)
            
    elif isinstance(config, list):
        for item in config:
            if item is not None:
                clean_config(item)

@st.cache_resource
def load_ai_model():
    model_path = find_model_file()
    
    if not model_path:
        st.error("❌ File 'best_model.h5' tidak ditemukan.")
        return None

    # Tampilkan info file di sidebar untuk memastikan file terbaca
    st.sidebar.success(f"File loaded: {os.path.basename(model_path)}")

    with st.spinner("⏳ Memproses arsitektur model..."):
        try:
            # 1. Buka File H5
            with h5py.File(model_path, 'r') as f:
                if 'model_config' not in f.attrs:
                    st.error("File rusak: Tidak ada 'model_config' di dalam h5.")
                    return None
                
                # Ambil config JSON
                config_str = f.attrs.get('model_config')
                if config_str is None:
                    st.error("Config model kosong (None).")
                    return None
                    
                model_config = json.loads(config_str)

            # 2. BERSIHKAN CONFIG (Penyebab utama error)
            clean_config(model_config)

            # 3. BANGUN ULANG MODEL DARI JSON BERSIH
            # Gunakan custom_object_scope agar layer aneh diabaikan
            with utils.custom_object_scope({}):
                model = models.model_from_json(json.dumps(model_config))
            
            # 4. ISI BOBOT (WEIGHTS)
            model.load_weights(model_path)
            
            return model

        except Exception as e:
            # Tampilkan error detail tapi jangan hentikan aplikasi total
            st.error(f"Gagal memuat model: {e}")
            st.warning("⚠️ Mengaktifkan Mode Darurat: Aplikasi berjalan, tapi AI dimatikan sementara.")
            return None

model = load_ai_model()

# ================== DATABASE INFO ==================
MODEL_LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

DISEASE_KB = {
    "Blas": {"title": "Penyakit Blas", "cause": "Jamur Pyricularia oryzae", "prevention": ["Varietas tahan", "Kurangi Nitrogen"], "treatment": ["Fungisida Tricyclazole"], "color": "red"},
    "Hawar Daun": {"title": "Hawar Daun (Kresek)", "cause": "Bakteri Xanthomonas", "prevention": ["Atur air", "Kurangi Urea"], "treatment": ["Bakterisida tembaga"], "color": "orange"},
    "Tungro": {"title": "Penyakit Tungro", "cause": "Virus Wereng Hijau", "prevention": ["Tanam serempak"], "treatment": ["Insektisida sistemik"], "color": "red"},
    "Sehat": {"title": "Tanaman Sehat", "cause": "Kondisi Optimal", "prevention": ["Rawat rutin"], "treatment": ["-"], "color": "green"},
    "Brown Spot": {"title": "Bercak Coklat", "cause": "Jamur Helminthosporium", "prevention": ["Pupuk Kalium"], "treatment": ["Fungisida"], "color": "brown"},
    "Rice Hispa": {"title": "Hama Putih Palsu", "cause": "Kumbang Hispa", "prevention": ["Pangkas daun"], "treatment": ["Insektisida"], "color": "gray"}
}

# ================== FUNGSI PREDIKSI ==================
def predict_image(image):
    if model is None: return None
    
    if image.mode != "RGB": image = image.convert("RGB")
    img = image.resize((224, 224))
    
    try: x = utils.img_to_array(img)
    except AttributeError: x = preprocessing.image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    try:
        pred = model.predict(x)
        idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0])) * 100
        label = MODEL_LABELS[idx] if idx < len(MODEL_LABELS) else "Sehat"
        return {"hasil": label, "confidence": round(confidence, 1), "detail": DISEASE_KB.get(label, DISEASE_KB["Sehat"])}
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        return None

# ================== SESSION STATE ==================
if "page" not in st.session_state: st.session_state.page = "home"
if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = None

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("🛠 Mode Developer")
    st.info("Simulasi data jika model error.")
    sim_disease = st.selectbox("Pilih Penyakit", list(DISEASE_KB.keys()))
    if st.button("Tampilkan Info Dummy"):
        st.session_state.result = {
            "hasil": sim_disease, "confidence": 100.0,
            "detail": DISEASE_KB[sim_disease],
            "image": Image.new('RGB', (200, 200), color=DISEASE_KB[sim_disease]['color'])
        }
        st.session_state.page = "result"
        st.rerun()

# ================== HALAMAN ==================
def home_page():
    st.markdown("## 🌾 Dashboard Kesehatan Padi")
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("➕ Scan Baru", use_container_width=True): st.session_state.page = "scan"
    
    if not st.session_state.history: st.info("Belum ada riwayat.")
    else:
        for i, h in enumerate(st.session_state.history):
            with st.container():
                c1, c2 = st.columns([1,3])
                with c1: st.image(h["image"], use_column_width=True)
                with c2:
                    st.markdown(f"**{h['title']}**")
                    st.caption(f"{h['confidence']}%")
                    if st.button("Hapus", key=f"del_{i}"):
                        st.session_state.history.pop(i)
                        st.rerun()
            st.divider()

def scan_page():
    st.markdown("## 📸 Scan Tanaman")
    img_file = st.camera_input("Ambil Foto")
    upl_file = st.file_uploader("Upload Foto", type=["jpg","png","jpeg"])
    
    image = None
    if img_file: image = Image.open(img_file)
    elif upl_file: image = Image.open(upl_file)
    
    if image:
        st.image(image, caption="Preview", use_column_width=True)
        if st.button("🔍 Analisis AI", use_container_width=True):
            if model is None:
                st.error("Model Error. Gunakan Mode Developer.")
            else:
                with st.spinner("Menganalisis..."):
                    res = predict_image(image)
                    if res:
                        res["image"] = image
                        st.session_state.result = res
                        st.session_state.page = "result"
                        st.rerun()
    if st.button("⬅ Kembali"): st.session_state.page = "home"

def result_page():
    if not st.session_state.result: st.session_state.page = "home"; st.rerun(); return
    r = st.session_state.result
    info = r["detail"]
    st.image(r["image"], use_column_width=True)
    st.markdown(f"## {info['title']}")
    st.success(f"Confidence: {r['confidence']}%") if r['confidence'] > 80 else st.warning(f"Confidence: {r['confidence']}%")
    st.info(f"Penyebab: {info['cause']}")
    c1, c2 = st.columns(2)
    with c1: 
        st.write("### 🛡 Pencegahan"); [st.write(f"- {p}") for p in info["prevention"]]
    with c2:
        st.write("### 💊 Pengobatan"); [st.write(f"- {t}") for t in info["treatment"]]
    if st.button("Simpan", use_container_width=True):
        st.session_state.history.insert(0, {"title": info["title"], "confidence": r["confidence"], "image": r["image"], "date": datetime.now().strftime("%d-%m %H:%M")})
        st.session_state.page = "home"; st.rerun()
    if st.button("Scan Lagi"): st.session_state.page = "scan"; st.rerun()

if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
