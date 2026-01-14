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
# Shortcut
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

# ================== TRANSLATOR KERAS 3 KE KERAS 2 ==================
def translate_config(config):
    """
    Menerjemahkan config Keras 3 agar bisa dibaca Keras 2.
    Bukan dihapus, tapi disesuaikan.
    """
    if isinstance(config, dict):
        # 1. TRANSLATE: batch_shape (Keras 3) -> batch_input_shape (Keras 2)
        if 'batch_shape' in config:
            config['batch_input_shape'] = config['batch_shape']
            del config['batch_shape']
        
        # 2. FIX: dtype policy dictionary -> string simple
        if 'dtype' in config:
            # Jika dtype berbentuk dict {'class_name':...}, ambil 'float32' aja
            if isinstance(config['dtype'], dict):
                config['dtype'] = 'float32'
            # Jika dtype None, paksa jadi float32
            elif config['dtype'] is None:
                config['dtype'] = 'float32'

        # 3. HAPUS: Fitur Keras 3 yang tidak ada di Keras 2
        ignore_keys = ['time_major', 'ragged', 'build_config', 'compile_config']
        for k in ignore_keys:
            if k in config:
                del config[k]

        # 4. REKURSIF (Lanjut ke anak-anaknya)
        for key, value in config.items():
            translate_config(value)
            
    elif isinstance(config, list):
        for item in config:
            translate_config(item)

@st.cache_resource
def load_ai_model():
    model_path = find_model_file()
    
    if not model_path:
        st.error("❌ File 'best_model.h5' tidak ditemukan di server.")
        return None

    st.sidebar.success(f"File ditemukan: {os.path.basename(model_path)}")

    # GLOBAL TRY-EXCEPT: Agar aplikasi tidak pernah crash total
    try:
        with st.spinner("⏳ Menerjemahkan Model Keras 3 ke Keras 2..."):
            
            # 1. Buka File H5 & Ambil JSON Config
            with h5py.File(model_path, 'r') as f:
                if 'model_config' not in f.attrs:
                    raise ValueError("File h5 tidak memiliki 'model_config'.")
                
                config_str = f.attrs.get('model_config')
                if config_str is None:
                    raise ValueError("Config model bernilai None.")
                    
                # Decode jika bytes
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                    
                model_config = json.loads(config_str)

            # 2. LAKUKAN PENERJEMAHAN (TRANSLATE)
            translate_config(model_config)

            # 3. BANGUN ULANG MODEL DARI JSON YANG SUDAH DITERJEMAHKAN
            # Gunakan scope kosong untuk menghindari error custom object
            with utils.custom_object_scope({}):
                model = models.model_from_json(json.dumps(model_config))
            
            # 4. ISI BOBOT (WEIGHTS)
            model.load_weights(model_path)
            
            return model

    except Exception as e:
        # TANGKAP ERROR AGAR TIDAK CRASH
        st.error(f"⚠️ Gagal memuat model AI: {e}")
        st.warning("Aplikasi beralih ke 'Mode Simulasi' agar tetap bisa digunakan.")
        return None

# Load model dengan aman
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
    # Jika model None, return None (biar UI yang handle)
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
    
    # Notifikasi status model
    if model is None:
        st.warning("⚠️ Model AI sedang offline (Mode Simulasi Aktif). Silakan gunakan 'Mode Developer' di sidebar.")
    
    img_file = st.camera_input("Ambil Foto")
    upl_file = st.file_uploader("Upload Foto", type=["jpg","png","jpeg"])
    
    image = None
    if img_file: image = Image.open(img_file)
    elif upl_file: image = Image.open(upl_file)
    
    if image:
        st.image(image, caption="Preview", use_column_width=True)
        # Disable tombol jika model error
        if st.button("🔍 Analisis AI", use_container_width=True, disabled=(model is None)):
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

# Routing sederhana
if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
