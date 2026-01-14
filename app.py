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
    page_title="AI Kesehatan Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    "Blas": {"title": "Penyakit Blas", "cause": "Jamur Pyricularia oryzae", "prevention": ["Varietas tahan", "Kurangi Nitrogen"], "treatment": ["Fungisida Tricyclazole"], "color": "red"},
    "Hawar Daun": {"title": "Hawar Daun (Kresek)", "cause": "Bakteri Xanthomonas", "prevention": ["Atur air", "Kurangi Urea"], "treatment": ["Bakterisida tembaga"], "color": "orange"},
    "Tungro": {"title": "Penyakit Tungro", "cause": "Virus Wereng Hijau", "prevention": ["Tanam serempak"], "treatment": ["Insektisida sistemik"], "color": "red"},
    "Sehat": {"title": "Tanaman Sehat", "cause": "Kondisi Optimal", "prevention": ["Rawat rutin"], "treatment": ["-"], "color": "green"},
    "Brown Spot": {"title": "Bercak Coklat", "cause": "Jamur Helminthosporium", "prevention": ["Pupuk Kalium"], "treatment": ["Fungisida"], "color": "brown"},
    "Rice Hispa": {"title": "Hama Putih Palsu", "cause": "Kumbang Hispa", "prevention": ["Pangkas daun"], "treatment": ["Insektisida"], "color": "gray"}
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

# ================== UI ==================
if "page" not in st.session_state: st.session_state.page = "home"
if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = None

with st.sidebar:
    st.header("🛠 Mode Developer")
    sim_disease = st.selectbox("Pilih Penyakit", list(DISEASE_KB.keys()))
    if st.button("Tampilkan Info Dummy"):
        st.session_state.result = {"hasil": sim_disease, "confidence": 100.0, "detail": DISEASE_KB[sim_disease], "image": Image.new('RGB', (200, 200), color=DISEASE_KB[sim_disease]['color'])}
        st.session_state.page = "result"; st.rerun()

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
                    if st.button("Hapus", key=f"del_{i}"): st.session_state.history.pop(i); st.rerun()
            st.divider()

def scan_page():
    st.markdown("## 📸 Scan Tanaman")
    if model is None: st.warning("⚠️ Model sedang didownload atau offline.")
    img_file = st.camera_input("Ambil Foto")
    upl_file = st.file_uploader("Upload Foto", type=["jpg","png","jpeg"])
    image = img_file if img_file else upl_file
    if image:
        image = Image.open(image)
        st.image(image, caption="Preview", use_column_width=True)
        if st.button("🔍 Analisis AI", use_container_width=True, disabled=(model is None)):
            with st.spinner("Menganalisis..."):
                res = predict_image(image)
                if res:
                    res["image"] = image
                    st.session_state.result = res
                    st.session_state.page = "result"; st.rerun()
    if st.button("⬅ Kembali"): st.session_state.page = "home"

def result_page():
    if not st.session_state.result: 
        st.session_state.page = "home"
        st.rerun()
        return

    r = st.session_state.result
    info = r["detail"]
    
    st.image(r["image"], use_column_width=True)
    st.markdown(f"## {info['title']}")
    
    # PERBAIKAN UTAMA: Menggunakan IF biasa, bukan one-liner
    if r['confidence'] > 80:
        st.success(f"Confidence: {r['confidence']}%")
    else:
        st.warning(f"Confidence: {r['confidence']}%")
        
    st.info(f"Penyebab: {info['cause']}")
    
    c1, c2 = st.columns(2)
    # PERBAIKAN: Menggunakan loop biasa, bukan list comprehension
    with c1: 
        st.write("### 🛡 Pencegahan")
        for p in info["prevention"]:
            st.write(f"- {p}")
            
    with c2: 
        st.write("### 💊 Pengobatan")
        for t in info["treatment"]:
            st.write(f"- {t}")

    if st.button("Simpan", use_container_width=True):
        st.session_state.history.insert(0, {"title": info["title"], "confidence": r["confidence"], "image": r["image"], "date": datetime.now().strftime("%d-%m %H:%M")})
        st.session_state.page = "home"; st.rerun()
        
    if st.button("Scan Lagi"): 
        st.session_state.page = "scan"
        st.rerun()

if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
